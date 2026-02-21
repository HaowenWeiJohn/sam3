import cv2
import matplotlib.pyplot as plt



# use cv2 to visualize an image and I can zoom in on the image
image_path = r"\\192.168.1.104\home\piano\data\overhead_camera_images\last_frames\028-12-fx30_2_0323.png"
image = cv2.imread(image_path)
plt.imshow(image)
plt.show()


