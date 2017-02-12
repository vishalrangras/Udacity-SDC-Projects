#Canny to Detect Lane Lines

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread('11.0-exit-ramp.jpg')
plt.imshow(image)
plt.show()

#pip install opencv-python
#bringing in OpenCV libraries
import cv2
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grascale conversion
plt.imshow(gray,cmap='gray')
plt.show()
mpimg.imsave("11.1-exit-ramp-gray.jpg", gray,cmap='gray')


#Define a kernel size for Gaussian smoothing / blurring
#Note: this step is optional as cv2.Canny() applies a 5 x 5 Gaussian internally
kernel_size = 1 #Must be an odd number (3, 5, 7 ...)
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

#Define parameters for Canny and run it
low_threshold = 100
high_threshold = 200
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

#Display the image
plt.imshow(edges, cmap='Greys_r')
plt.show()
mpimg.imsave("11.2-exit-ramp-canny.jpg", edges,cmap='Greys_r')
