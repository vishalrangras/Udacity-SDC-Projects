#Color and Region mask Combined to detect lane lines

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#Read in the image
image = mpimg.imread('test.jpg')

#Grab the x and y sizes and make two copies of the image
#With one copy we'll extract only the pixels that meet our selection,
#then we'll paint those pixels red in the original image to see our selection
#overlaid on the original.
ysize = image.shape[0]
xsize = image.shape[1]

#Debug
print ('The ysize and xsize of the image are: ',ysize,', ',xsize)
color_select = np.copy(image)
line_image= np.copy(image)

#Define our color criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

#Define a triangle region of interest
#Note: the origin (x=0. y=0) is in the upper left in image processing
left_bottom = [0, 539]
right_bottom = [900, 539]
apex = [475, 320]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]),1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

#Mask pixels below the threshold
#(image[:,:,0]), (image[:,:,1]), (image[:,:,2]) represents all pixels on x-axis and y-axis for R, Y, B channels respectively

color_thresholds = (image[:,:,0] < rgb_threshold[0]) | (image[:,:,1] < rgb_threshold[1]) | (image[:,:,2]<rgb_threshold[2])

#Find the region inside the lines
#The logic of this part is not clear to me yet
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0]+ fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

#Mask color selection and region selection
color_select[color_thresholds | ~region_thresholds] = [0,0,0]

#Color the pixels in red where both color and region selection met
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

#plt.ion()

#Display our two output images
plt.imshow(color_select)
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
plt.show()
plt.imshow(line_image)
plt.show()
mpimg.imsave("05-Mask_color_selection.jpg", color_select)
mpimg.imsave("05-Image_colored_red_in_region.jpg", line_image)
