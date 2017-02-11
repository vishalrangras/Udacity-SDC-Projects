## Self Driving Car Nanodegree - Project 1 - Detecting Car Lane Lines

**Notes**

1. I have found out an amazing article on automatic threshold detection for Canny Algorithm which I have implemented in the code. What the logic does is it takes median from the pixels of the provided image and then calculates threshold values using 1:3 ratio by putting sigma = 0.33

	Attribution: http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

2. For extraplotation of Line Segments, I simply iterated all the lines available after Hough Transformation and obtained maximum slopes, segment lengths and Y-Intercepts. Based on this values, a single line was created for each side of the lane within region of interest. Then these two lines were added into original image through linear blending function provided.

3. Rest of the things were similar to the process which was taught in classroom lectures. To conclude, the pipeline takes images one after another and process them in a following manner:

    a. Color to Grayscale Conversion
	
    b. Gaussian Blur for smoothing of edges and removing noise - Input parameter to be provided: Kernel size
    
	c. Canny Edge Detection - In order to detect only edges from a Gaussian Blurred Grayscale image.
        The Edge detection Algorithm works on the principle of calculating gradient at different points in image.
        The gradients at edge will be higher due to change in intensity of bright and dark pixel.
    
	d. Region of Interest Mask - To consider only portion where preferably Lane Lines will be present.
        If we don't do that, many unwanted edges will be detected.
    
	e. Hough Transformation - To find out the lines based on votes we get for a particular grid in Hough Space.
            The higher the votes, the more intersection of lines in Hough Space is denoted. 
            And the higher number of intersection points indicates a connected line in Image Space.
    
	f. Extrapolation of Line - Already explained above.
    
	g. Linear blending: of Lines generated after extrapolation of HoughLines with Original Source image.

	Attributions:

		1. http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
		2. http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
		3. https://alyssaq.github.io/2014/understanding-hough-transform/

**Shortcomings and Possible improvements**

1. I haven't worked for the logic to extrapolate lane if there are curves on the road (i.e.: Optional Challenge Video). Improvements can be done to take smaller line segments and integrate them to find curves instead of linear lines.

2. Different luminosity factors can be considered to design the algorithm in a way that it works for brighter as well as darker lane lines irrespective of time of the day.

3. Since I am from India and I have seen here that there are no proper lanes on the road in small towns, it is quite impossible to use this logic of lane detection over here. Something can be thought of in that direction to enable autonomous vehicles to drive.

4. A logic can be implemented for transient conditions like when the car is changing its lane.
