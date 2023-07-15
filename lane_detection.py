import cv2 as cv
import numpy as np
import math

# purpose: computes the slope and y-intercept of a line
# parameters: numpy.ndarray(shape: (2), type: numpy.intc) 
# return values: float slope, float yIntercept
def computeLineParameters(line):
    pt1 = (line[0], line[1]) # type: Tuple(2, int)
    pt2 = (line[2], line[3]) # type: Tuple(2, int)
    slope = float(pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    yIntercept = float(line[1]) - slope * float(line[0])
    return slope, yIntercept

# purpose: adjustYRange and ExtendLine are use to get the min and max y-values (assumes lane is vertical)
# and use them to extrapolate the filtered lines we have (that doesn't contain duplicates)
# This technique helps to get the line for the full lane in cases where line detection
# ends up detecting broken segments for a single line instead of the full lane line
# parameters: numpy.ndarray(shape: (2), type: numpy.intc), Tuple(2, int) yRange
# return values: Tuple(2, int) newYRange
def adjustYRange(line, yRange):
    #print(yRange[0], yRange[1], line[1], line[3])
    return (min(yRange[0], min(line[1], line[3])), max(yRange[1], max(line[1], line[3])))

# parameters: Tuple(4, int), Tuple(2, int)
# no return values
def ExtendLine(line, yRange):
    slope, yIntercept = computeLineParameters(line) # type: float
    line[0] = (yRange[0] - yIntercept) /  slope
    line[1] = yRange[0]
    line[2] = (yRange[1] - yIntercept) /  slope
    line[3] = yRange[1]

# purpose: use the left and right lane edges to get a lane marker along the center of the lane
# parameters: Tuple(4, int), Tuple(4, int)
# return values: Tuple(4, int)
def GetLaneCenter(laneEdge1, laneEdge2):
    return (int((laneEdge1[0] + laneEdge2[0]) / 2),  int((laneEdge1[1] + laneEdge2[1]) / 2),
                int((laneEdge1[2] + laneEdge2[2]) / 2), int((laneEdge1[3] + laneEdge2[3]) / 2))


# Open video file
cap = cv.VideoCapture("./sample_videos/test_video.mp4")
if (not cap.isOpened()):
    print("Error opening video")
    exit()

# Main program loop
while (True):
    # open the next frame from the video file, or exit the loop if its the end
    ret, frame = cap.read()
    if (not ret):
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #print(frame.shape)

    ######## Stage 1 : Edge Detection #########/
    # Apply Gaussian Blur before Canny
    gaussianKernelSize = (5, 5) # type: Tuple(2, int)
    frameFiltered = cv.GaussianBlur(frame, gaussianKernelSize, 0, None, 0) # type: numpy.ndarray

    # Get optimal parameters for Canny edge detection using the median of the grayscale frame
    gray_img = cv.cvtColor(frameFiltered, cv.COLOR_BGR2GRAY) # type: numpy.ndarray
    v = np.median(gray_img) # type: float
    sigma = 0.33 # type: float
    lower_thresh = max(0, int((1.0 - sigma) * v)) # type: int
    upper_thresh = min(255, int((1.0 + sigma) * v)) # type: int
    #print(lower_thresh, " ", upper_thresh)

    # Use Canny to get edge map
    edgeMap = cv.Canny(frameFiltered, lower_thresh, upper_thresh) # type: numpy.ndarray

    ######## Stage 2 : Line Detection #########
    rhoRes=1.0              # type: const float
    thetaRes=3*math.pi/180.0   # type: const float
    Threshold=155              # type: const int
    minLineLength = 100    # 30; type: const float 
    maxLineGap = 10         # type: const float

    linesP = cv.HoughLinesP(edgeMap, rhoRes, thetaRes, Threshold, None, minLineLength, maxLineGap) # type: numpy.ndarray(shape: (x, 1, 2), type: numpy.intc)

    ######## Stage 3 : Remove Unwanted and Duplicate Lines #########
    # Remove unwanted lines based on their slope
    minSlope = 0.75 # 0.5; type: const float 
    maxSlope = 1.25 # 2.0; type: const float 
    # Remove duplicate lines using slope and y-intercept percentage thresholds
    slopeThreshold = 1.0 # type: const float
    yInterceptThreshold = 1.0 # type: const float

    filteredLines = [] # used to store lines after removing unwanted and duplicates; type: list of numpy.ndarray(shape: (2), type: numpy.intc)
    #yRanges; # y Ranges to extrapolate lines; type: list of Tuple(2, int)
    yRange = (frame.shape[1], 0) # y Range to extrapolate lines; # Tuple(2, int)
    for i in range(1, len(linesP)):
        # compute slope and y-intercept
        line = linesP[i][0] # type: numpy.ndarray(shape: (2), type: numpy.intc)
        slope1, yIntercept1 = computeLineParameters(line) # type: float

        # detect if unwanted: if not in slope range, it is unwanted so we skip this line
        if (abs(slope1) < minSlope or abs(slope1) > maxSlope):
            continue

        # detect if duplicate
        isDuplicate = False # type: bool
        for j in range(len(filteredLines)):
            # compute slope and y-intercept (similar to any line in list of filtered lines)
            slope2, yIntercept2 = computeLineParameters(filteredLines[j]) # type: float
            #print(slope1, slope2, yIntercept1, yIntercept2)
            # if line and filteredLines[j] are within slope and y-intercept threshold, line is a duplicate of filteredLines[j]
            if (abs((slope1 - slope2) / slope2) < slopeThreshold and abs((yIntercept1 - yIntercept2) / yIntercept2) < yInterceptThreshold):
                isDuplicate = True
                #adjustYRange(linesP[i][0], yRanges[j])
                break

        # if not a duplicate, add it to list of filtered lines
        if not isDuplicate:
            filteredLines.append(line) # need to make a copy???
            #yRanges.append(min(line[1], line[3]), max(line[1], line[3]))

        # adjust yRange
        yRange = adjustYRange(linesP[i][0], yRange)

    ######## Stage 4 : Add center lane marker and display frame #########
    showUnwantedAndDuplicatelines = False # for debugging, type: const bool 

    # Display frame
    #edgeMapBGR = cv.cvtColor(edgeMap, cv.COLOR_BGR2GRAY) # type: numpy.ndarray

    thickness = 8 if showUnwantedAndDuplicatelines else 2 # type: int
    for i in range(len(filteredLines)):
        ExtendLine(filteredLines[i], yRange)
        pt1 = (filteredLines[i][0], filteredLines[i][1]) # type: Tuple(2, int)
        pt2 = (filteredLines[i][2], filteredLines[i][3]) # type: Tuple(2, int)
        cv.line(frame, pt1, pt2, (0, 255, 0), thickness, cv.LINE_AA)

    # to draw unwanted and duplicate lines for debugging
    if (showUnwantedAndDuplicatelines):
        for i in range(len(linesP)):
            pt1 = (linesP[i][0][0], linesP[i][0][1]) # type: Tuple(2, int)
            pt2 = (linesP[i][0][2], linesP[i][0][3]) # type: Tuple(2, int)

            # compute slope and y-intercept
            slope1, yIntercept1 = computeLineParameters(linesP[i][0]) # type: float

            # remove unwanted lines using slope threshold
            if (abs(slope1) < minSlope or abs(slope1) > maxSlope):
                cv.line(frame, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)
            else:
                cv.line(frame, pt1, pt2, (255, 0, 0), 2, cv.LINE_AA)

    # if there are 2 lines, compute center of lane marker and draw it
    if (len(filteredLines) == 2):
        laneMarker = GetLaneCenter(filteredLines[0], filteredLines[1]) # type: Tuple(4, int)
        pt1 = (laneMarker[0], laneMarker[1]) # type: Tuple(2, int)
        pt2 = (laneMarker[2], laneMarker[3]) # type: Tuple(2, int)
        cv.line(frame, pt1, pt2, (0, 255, 0), 2, cv.LINE_AA)

    #print(len(filteredLines))
    cv.imshow("Frame with Lane Marker", frame)
    if cv.waitKey(10) == ord('q'): 
        break
    #while(waitKey(10)!='a') # to examine frame by frame for debugging

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
