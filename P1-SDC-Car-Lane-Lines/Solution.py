#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)
#plt.show()
#if you wanted to show a single color channel image called 'gray', for example,
#call as plt.imshow(gray, cmap='gray')

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, sigma):
    """Applies the Canny transform"""
    img_median = np.median(img)
    low_threshold = int(max(0, (1.0 - sigma) * img_median))
    high_threshold = int(min(255, (1.0 + sigma) * img_median))
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    left_lane_max, right_lane_max = 0, 0
    left_slope, right_slope = -0.1, 0.1
    left_lane, right_lane = (0,0,0,0), (0,0,0,0)    
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            lane_length = math.hypot(x2-x1, y2-y1)

            #For left lane
            if(slope < -0.5):
               if(lane_length > left_lane_max):
                   left_lane = (x1,y1,x2,y2)
                   left_slope = slope
                   left_lane_max = lane_length
                   
            #For right lane
            elif(slope > 0.5):
                if(lane_length > right_lane_max):
                    right_lane = (x1,y1,x2,y2)
                    right_slope = slope
                    right_lane_max = lane_length

    #Intercept: c = y - mx
    left_lane_intercept = left_lane[1] - (left_slope * left_lane[0])
    right_lane_intercept = right_lane[1] - (right_slope * right_lane[0])

    #Variables for a complete extrapolated line
    Y1 = 540
    Y2 = 330
    
    #Equation for extrapolated line
    #x = (y - c) / m
    LX1 = int((Y1 - left_lane_intercept)/left_slope)
    LX2 = int((Y2 - left_lane_intercept)/left_slope)
    RX1 = int((Y1 - right_lane_intercept)/right_slope)
    RX2 = int((Y2 - right_lane_intercept)/right_slope)

    #Adding Left Lane Line
    cv2.line(img, (LX1,Y1), (LX2,Y2), color, thickness)

    #Adding Right Lane Line
    cv2.line(img, (RX1,Y1), (RX2,Y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=1, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def detect_lanes(img):
    gray =  grayscale(img)
    blur = gaussian_blur(gray,3)
    edges = canny(blur, 0.33)
    imshape = img.shape
    vertices = np.array([[(0,imshape[0]),(470, 330), (490, 330), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_image =  region_of_interest(edges, vertices)
    line_img = hough_lines(masked_image, 1, np.pi/180, 30, 50, 150)
    annotated_img = weighted_img(line_img, img)
    return annotated_img
    
import os
#current_dir = os.getcwd() + '/test_images'
print (os.listdir("test_images/"))

for file in os.listdir("test_images/"):
    imageFile = mpimg.imread("test_images/"+file)
    annotated_image = detect_lanes(imageFile)
    mpimg.imsave("test_images/out_"+file,annotated_image)

from moviepy.editor import VideoFileClip

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

    
