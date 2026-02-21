"""
Piano Keyboard Image Coregistration

Two-phase feature matching + Thin Plate Spline (TPS) warping to produce
dense per-pixel mappings from keyboard2→keyboard1 and keyboard3→keyboard1.

Phase 1: SIFT + BFMatcher crossCheck + RANSAC homography (rough alignment)
Phase 2: Re-match on warped image with spatial proximity constraint (refined)
Phase 3: TPS via scipy RBFInterpolator for smooth non-linear dense mapping
"""

import os
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "keyboard_image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "intermidiate_results", "coregistration")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_image(path):
    """Load RGBA image and convert to BGR for OpenCV processing."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def detect_features(img_gray, n_features=3000):
    """Detect SIFT keypoints and compute descriptors."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    kps, des = sift.detectAndCompute(img_gray, None)
    return kps, des


def match_features_phase1(kps1, des1, kps2, des2):
    """Phase 1: BFMatcher with crossCheck for initial correspondences."""
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2, matches


def compute_homography(pts_src, pts_dst, reproj_thresh=5.0):
    """Compute homography with RANSAC, return H matrix and inlier mask."""
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, reproj_thresh)
    mask = mask.ravel().astype(bool)
    return H, mask


def filter_outlier_matches(pts_ref, pts_warp, k=6, threshold=2.5):
    """
    Remove outlier matches whose displacement deviates significantly from
    their k nearest neighbors. Uses median absolute deviation (MAD).
    """
    from scipy.spatial import cKDTree

    if len(pts_ref) < k + 1:
        return pts_ref, pts_warp

    displacements = pts_warp - pts_ref  # (N, 2)
    tree = cKDTree(pts_ref)
    _, indices = tree.query(pts_ref, k=k + 1)  # includes self
    indices = indices[:, 1:]  # exclude self

    keep = []
    for i in range(len(pts_ref)):
        neighbor_disp = displacements[indices[i]]  # (k, 2)
        median_disp = np.median(neighbor_disp, axis=0)
        mad = np.median(np.abs(neighbor_disp - median_disp), axis=0)
        mad = np.maximum(mad, 1.0)  # floor to avoid division by zero
        dev = np.abs(displacements[i] - median_disp) / mad
        if np.all(dev < threshold):
            keep.append(i)

    keep = np.array(keep)
    if len(keep) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return pts_ref[keep], pts_warp[keep]


def match_features_phase2(ref_gray, warped_gray, max_dist=30.0, n_features=5000):
    """
    Phase 2: Re-detect features on reference and warped-to-reference images,
    then keep only spatially close matches (within max_dist pixels).
    Applies outlier filtering based on local displacement consistency.
    """
    kps_ref, des_ref = detect_features(ref_gray, n_features)
    kps_warp, des_warp = detect_features(warped_gray, n_features)

    if des_ref is None or des_warp is None:
        return np.empty((0, 2)), np.empty((0, 2))

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_ref, des_warp)

    pts_ref, pts_warp = [], []
    for m in matches:
        p_ref = np.array(kps_ref[m.queryIdx].pt)
        p_warp = np.array(kps_warp[m.trainIdx].pt)
        if np.linalg.norm(p_ref - p_warp) < max_dist:
            pts_ref.append(p_ref)
            pts_warp.append(p_warp)

    if len(pts_ref) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    pts_ref = np.float32(pts_ref)
    pts_warp = np.float32(pts_warp)

    # Filter outliers based on local displacement smoothness
    n_before = len(pts_ref)
    pts_ref, pts_warp = filter_outlier_matches(pts_ref, pts_warp)
    n_after = len(pts_ref)
    if n_before != n_after:
        print(f"      Outlier filter: {n_before} -> {n_after} matches")

    return pts_ref, pts_warp


def back_project_points(pts, H_inv):
    """Back-project 2D points through inverse homography."""
    n = len(pts)
    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])  # (N, 3)
    proj = (H_inv @ pts_h.T).T  # (N, 3)
    proj = proj[:, :2] / proj[:, 2:3]
    return proj.astype(np.float32)


def generate_boundary_anchors(ref_shape, H_inv, n_edge=20):
    """
    Generate anchor point pairs along the reference image boundary and a
    sparse interior grid, using the homography to compute corresponding
    moving-image coordinates. These anchor the TPS at the edges to prevent
    wild extrapolation.
    """
    h, w = ref_shape[:2]

    # Edge points along all 4 borders
    pts = []
    for t in np.linspace(0, 1, n_edge, endpoint=True):
        pts.append([t * (w - 1), 0])               # top
        pts.append([t * (w - 1), h - 1])            # bottom
        pts.append([0, t * (h - 1)])                # left
        pts.append([w - 1, t * (h - 1)])            # right

    # Sparse interior grid (5 rows x 10 cols)
    for yy in np.linspace(0.1, 0.9, 5):
        for xx in np.linspace(0.05, 0.95, 10):
            pts.append([xx * (w - 1), yy * (h - 1)])

    pts_ref = np.array(pts, dtype=np.float64)

    # Remove duplicates
    pts_ref = np.unique(pts_ref, axis=0)

    # Project to moving image via inverse homography
    pts_mov = back_project_points(pts_ref, H_inv)
    return pts_ref.astype(np.float32), pts_mov


def build_tps_mapping(pts_ref, pts_moving, ref_shape, H_inv, smoothing=1.0):
    """
    Build dense per-pixel mapping using Thin Plate Spline (RBFInterpolator).

    Adds homography-based boundary/grid anchors to prevent TPS divergence
    outside the convex hull of matched feature points. The TPS smoothly
    interpolates between accurate local correspondences (features) and the
    global homography (anchors).

    pts_ref: (N, 2) matched points in reference image
    pts_moving: (N, 2) corresponding points in moving image
    ref_shape: (H, W) of reference image
    H_inv: 3x3 inverse homography (reference -> moving)
    smoothing: RBF smoothing parameter (>0 for regularization)
    Returns: map_x, map_y arrays of shape (H, W)
    """
    h, w = ref_shape[:2]

    # Add boundary/grid anchor points from homography
    anchor_ref, anchor_mov = generate_boundary_anchors(ref_shape, H_inv)

    # Remove anchors that are too close to feature matches (features take priority)
    from scipy.spatial import cKDTree

    if len(pts_ref) > 0:
        tree = cKDTree(pts_ref)
        dists, _ = tree.query(anchor_ref)
        far_mask = dists > 10.0
        anchor_ref = anchor_ref[far_mask]
        anchor_mov = anchor_mov[far_mask]

    all_ref = np.vstack([pts_ref, anchor_ref])
    all_mov = np.vstack([pts_moving, anchor_mov])

    # Remove any remaining near-duplicate reference points
    tree_all = cKDTree(all_ref)
    pairs = tree_all.query_pairs(r=0.5)
    if pairs:
        remove_idx = set()
        for i, j in pairs:
            remove_idx.add(max(i, j))  # remove the later one (anchor over feature)
        keep = [i for i in range(len(all_ref)) if i not in remove_idx]
        all_ref = all_ref[keep]
        all_mov = all_mov[keep]

    print(f"      TPS points: {len(pts_ref)} features + {len(anchor_ref)} anchors = {len(all_ref)} total")

    # Fit two separate RBF interpolators: one for x-mapping, one for y-mapping
    tps_x = RBFInterpolator(
        all_ref, all_mov[:, 0], kernel="thin_plate_spline", smoothing=smoothing
    )
    tps_y = RBFInterpolator(
        all_ref, all_mov[:, 1], kernel="thin_plate_spline", smoothing=smoothing
    )

    # Build dense grid of reference pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Evaluate TPS in chunks to manage memory
    chunk_size = 100_000
    map_x_flat = np.empty(grid_pts.shape[0], dtype=np.float32)
    map_y_flat = np.empty(grid_pts.shape[0], dtype=np.float32)

    for i in range(0, grid_pts.shape[0], chunk_size):
        chunk = grid_pts[i : i + chunk_size]
        map_x_flat[i : i + chunk_size] = tps_x(chunk).astype(np.float32)
        map_y_flat[i : i + chunk_size] = tps_y(chunk).astype(np.float32)

    map_x = map_x_flat.reshape(h, w)
    map_y = map_y_flat.reshape(h, w)
    return map_x, map_y


def build_homography_mapping(H, ref_shape):
    """Fallback: build dense mapping from a homography matrix (moving→reference).
    H maps moving→reference, so we need H_inv to go reference→moving."""
    h, w = ref_shape[:2]
    H_inv = np.linalg.inv(H)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ones = np.ones((h, w), dtype=np.float32)
    coords = np.stack([grid_x, grid_y, ones], axis=-1)  # (H, W, 3)
    coords_flat = coords.reshape(-1, 3)
    proj = (H_inv @ coords_flat.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    map_x = proj[:, 0].reshape(h, w).astype(np.float32)
    map_y = proj[:, 1].reshape(h, w).astype(np.float32)
    return map_x, map_y


def warp_image(img, map_x, map_y):
    """Warp image using dense coordinate maps via cv2.remap."""
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ---------------------------------------------------------------------------
# Full registration pipeline
# ---------------------------------------------------------------------------
def register_image_pair(img_ref, img_moving, pair_name=""):
    """
    Full two-phase registration pipeline.

    Returns:
        map_x, map_y: dense per-pixel mappings (reference→moving coordinates)
        warped: warped moving image in reference frame
        H: rough homography matrix
        info: dict with diagnostic data
    """
    ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    mov_gray = cv2.cvtColor(img_moving, cv2.COLOR_BGR2GRAY)

    # --- Phase 1: rough alignment ---
    print(f"  [{pair_name}] Phase 1: SIFT detection + rough matching...")
    kps_ref, des_ref = detect_features(ref_gray)
    kps_mov, des_mov = detect_features(mov_gray)
    print(f"    Reference features: {len(kps_ref)}, Moving features: {len(kps_mov)}")

    pts_ref_p1, pts_mov_p1, raw_matches = match_features_phase1(
        kps_ref, des_ref, kps_mov, des_mov
    )
    print(f"    Cross-check matches: {len(pts_ref_p1)}")

    H, inlier_mask = compute_homography(pts_mov_p1, pts_ref_p1)
    n_inliers = int(inlier_mask.sum())
    print(f"    RANSAC inliers: {n_inliers}/{len(pts_ref_p1)} ({100*n_inliers/max(1,len(pts_ref_p1)):.0f}%)")

    # Warp moving image to reference space using homography
    h_ref, w_ref = img_ref.shape[:2]
    warped_rough = cv2.warpPerspective(img_moving, H, (w_ref, h_ref))
    warped_rough_gray = cv2.cvtColor(warped_rough, cv2.COLOR_BGR2GRAY)

    # --- Phase 2: refined matching ---
    print(f"  [{pair_name}] Phase 2: Refined matching on warped image...")
    pts_ref_p2, pts_warp_p2 = match_features_phase2(ref_gray, warped_rough_gray)
    n_refined = len(pts_ref_p2)
    print(f"    Spatially consistent matches: {n_refined}")

    use_tps = n_refined >= 15
    info = {
        "n_phase1_matches": len(pts_ref_p1),
        "n_inliers": n_inliers,
        "n_phase2_matches": n_refined,
        "method": "TPS" if use_tps else "homography",
        "H": H,
    }

    H_inv = np.linalg.inv(H)

    if use_tps:
        # Back-project phase-2 warped points to original moving image coords
        pts_mov_orig = back_project_points(pts_warp_p2, H_inv)

        info["pts_ref"] = pts_ref_p2
        info["pts_mov"] = pts_mov_orig

        # --- Phase 3: TPS dense mapping ---
        # Adaptive smoothing: more matches = less smoothing needed
        smoothing = max(1.0, 50.0 / n_refined)
        print(f"  [{pair_name}] Phase 3: Building TPS dense mapping ({w_ref}x{h_ref}, smoothing={smoothing:.1f})...")
        map_x, map_y = build_tps_mapping(
            pts_ref_p2, pts_mov_orig, (h_ref, w_ref), H_inv, smoothing=smoothing
        )
    else:
        print(f"  [{pair_name}] Fallback: Using homography-only dense mapping.")
        map_x, map_y = build_homography_mapping(H, (h_ref, w_ref))
        info["pts_ref"] = pts_ref_p1[inlier_mask]
        info["pts_mov"] = pts_mov_p1[inlier_mask]

    warped = warp_image(img_moving, map_x, map_y)
    return map_x, map_y, warped, info


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_matches(img_ref, img_moving, pts_ref, pts_mov, save_path):
    """Draw correspondences between reference and moving image."""
    h1, w1 = img_ref.shape[:2]
    h2, w2 = img_moving.shape[:2]
    h_out = max(h1, h2)
    canvas = np.zeros((h_out, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img_ref
    canvas[:h2, w1:] = img_moving

    for p1, p2 in zip(pts_ref, pts_mov):
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]) + w1, int(p2[1])
        color = (0, 255, 0)
        cv2.circle(canvas, (x1, y1), 3, color, -1)
        cv2.circle(canvas, (x2, y2), 3, color, -1)
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

    cv2.imwrite(save_path, canvas)
    print(f"    Saved match visualization: {save_path}")


def visualize_overlay(img_ref, warped, save_path, block_size=40):
    """Checkerboard overlay of reference and warped image."""
    h, w = img_ref.shape[:2]
    result = img_ref.copy()
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            by = (y // block_size) % 2
            bx = (x // block_size) % 2
            if (by + bx) % 2 == 1:
                y2 = min(y + block_size, h)
                x2 = min(x + block_size, w)
                result[y:y2, x:x2] = warped[y:y2, x:x2]

    cv2.imwrite(save_path, result)
    print(f"    Saved overlay: {save_path}")


def visualize_displacement(map_x, map_y, save_path):
    """Visualize displacement field magnitude as a heatmap."""
    h, w = map_x.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx = map_x - grid_x
    dy = map_y - grid_y
    magnitude = np.sqrt(dx**2 + dy**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    im0 = axes[0].imshow(dx, cmap="RdBu_r")
    axes[0].set_title("X displacement")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(dy, cmap="RdBu_r")
    axes[1].set_title("Y displacement")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(magnitude, cmap="hot")
    axes[2].set_title("Displacement magnitude")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved displacement field: {save_path}")


def compute_quality_metrics(img_ref, warped):
    """Compute SSIM and NCC between reference and warped images."""
    ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    warp_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Create mask where warped image has content
    mask = warp_gray > 0

    # NCC (normalized cross-correlation) on valid region
    if mask.sum() > 0:
        ref_vals = ref_gray[mask].astype(np.float64)
        warp_vals = warp_gray[mask].astype(np.float64)
        ref_vals -= ref_vals.mean()
        warp_vals -= warp_vals.mean()
        denom = np.sqrt((ref_vals**2).sum() * (warp_vals**2).sum())
        ncc = (ref_vals * warp_vals).sum() / max(denom, 1e-10)
    else:
        ncc = 0.0

    # SSIM on valid (non-black) region only
    try:
        from skimage.metrics import structural_similarity

        # Crop to bounding box of valid warped content
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]
            ref_crop = ref_gray[r_min:r_max + 1, c_min:c_max + 1]
            warp_crop = warp_gray[r_min:r_max + 1, c_min:c_max + 1]
            ssim = structural_similarity(ref_crop, warp_crop, data_range=255)
        else:
            ssim = 0.0
    except ImportError:
        ssim = float("nan")

    return {"ssim": ssim, "ncc": ncc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load images
    print("Loading images...")
    img1 = load_image(os.path.join(IMAGE_DIR, "keyboard1.png"))
    img2 = load_image(os.path.join(IMAGE_DIR, "keyboard2.png"))
    img3 = load_image(os.path.join(IMAGE_DIR, "keyboard3.png"))
    print(f"  keyboard1: {img1.shape}")
    print(f"  keyboard2: {img2.shape}")
    print(f"  keyboard3: {img3.shape}")

    pairs = [
        ("keyboard2", img2),
        ("keyboard3", img3),
    ]

    for name, img_mov in pairs:
        print(f"\n{'='*60}")
        print(f"Registering {name} -> keyboard1")
        print(f"{'='*60}")

        map_x, map_y, warped, info = register_image_pair(img1, img_mov, pair_name=name)

        prefix = os.path.join(OUTPUT_DIR, name)

        # Save dense mappings
        np.save(f"{prefix}_map_x.npy", map_x)
        np.save(f"{prefix}_map_y.npy", map_y)
        print(f"    Saved map_x: {prefix}_map_x.npy  shape={map_x.shape}")
        print(f"    Saved map_y: {prefix}_map_y.npy  shape={map_y.shape}")

        # Save warped image
        cv2.imwrite(f"{prefix}_registered.png", warped)
        print(f"    Saved registered image: {prefix}_registered.png")

        # Save homography
        np.save(f"{prefix}_homography.npy", info["H"])

        # Visualizations
        visualize_matches(img1, img_mov, info["pts_ref"], info["pts_mov"], f"{prefix}_matches.png")
        visualize_overlay(img1, warped, f"{prefix}_overlay.png")
        visualize_displacement(map_x, map_y, f"{prefix}_displacement.png")

        # Quality metrics
        metrics = compute_quality_metrics(img1, warped)
        print(f"    Quality - SSIM: {metrics['ssim']:.4f}, NCC: {metrics['ncc']:.4f}")
        print(f"    Method: {info['method']} ({info['n_phase2_matches']} refined matches)")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
