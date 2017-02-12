import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# Read in and grayscale the image
# Note: in the previous example we were reading a .jpg 
# Here we read a .png and convert to 0,255 bytescale
image = mpimg.imread('15.0-exit-ramp.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

plt.imshow(image)
#plt.show()

plt.imshow(gray, cmap='Greys_r')
#plt.show()
mpimg.imsave('15.1-exit-ramp-gray.jpg', gray, cmap='Greys_r')

# Define a kernel size and apply Gaussian smoothing
kernel_size = 9
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

plt.imshow(edges, cmap='Greys_r')
#plt.show()
mpimg.imsave('15.3-exit-ramp-edges.jpg', edges, cmap='Greys_r')

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255

# This time we are defining a four sided polygon to mask
#Shape of image is accessed by img.shape.
#It returns a tuple of number of rows, columns and channels (if image is color)
imshape = image.shape
print (imshape)
#vertices = np.array([[(80,imshape[0]),(420, 300), (520, 300), (900,imshape[0])]], dtype=np.int32)
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)

#Python: cv2.fillPoly(img, pts, color[, lineType[, shift[, offset]]]) → None
cv2.fillPoly(mask, vertices, ignore_mask_color)
plt.imshow(mask, cmap='Greys_r')
#plt.show()

masked_edges = cv2.bitwise_and(edges, mask)

plt.imshow(masked_edges, cmap='Greys_r')
#plt.show()
mpimg.imsave('15.5-exit-ramp-masked_edges.jpg', masked_edges, cmap='Greys_r')

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1*(np.pi/180) # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

#np.copy - Creates a copy of image. Then it is multiplied by 0.

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

plt.imshow(line_image)
#plt.show()
mpimg.imsave('15.6-exit-ramp-line_image.jpg', line_image)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
#Python: cv.AddWeighted(src1, alpha, src2, beta, gamma, dst) → None
lines_edges = cv2.addWeighted(color_edges, 0.5, line_image, 1, 0)
color_lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
plt.imshow(lines_edges)
plt.show()
plt.imshow(color_lines_edges)
#plt.show()
mpimg.imsave('15.8-exit-ramp-line_edges.jpg', lines_edges)
mpimg.imsave('15.9-exit-ramp-color_line_edges.jpg', color_lines_edges)
