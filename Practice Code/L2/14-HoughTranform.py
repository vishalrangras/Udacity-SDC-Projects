#Hough Tranform to find Lane Lines

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#Read in and grayscale the image
image = mpimg.imread('14.0-exit-ramp.jpg')

plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
plt.imshow(gray,cmap='Greys_r')
plt.show()
mpimg.imsave('14.0-exit-ramp-gray.jpg', gray, cmap='Greys_r')
#Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size),0)

#Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

plt.imshow(edges,cmap='Greys_r')
plt.show()
mpimg.imsave('14.1-exit-ramp-edge.jpg', edges, cmap='Greys_r')

#Define the Hough transform parameters
#Make a blank the same size as our image to draw on

rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 10
max_line_gap = 1
line_image = np.copy(image) * 0 #Creating a blank to draw lines on

#Run Hough on edge detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

#Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

plt.imshow(line_image)
plt.show()
mpimg.imsave('14.2-exit-ramp-line.jpg', line_image, cmap='Greys_r')
#Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))

#Draw the lines on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1,0)
plt.imshow(combo)
plt.show()
mpimg.imsave('14.3-exit-ramp-combo.jpg', combo, cmap='Greys_r')
