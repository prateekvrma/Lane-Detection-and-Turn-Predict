import cv2
import numpy as np

def performCannyEdgeDetection(frame):
    # Converts frame to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line[0].reshape((4,1))
        print(x1, y1, x2, y2)

        slope = (y2 - y1)/ (x2 - x1)
        y_intercept = y1 - (slope*x1)
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append([slope, y_intercept])
        else:
            right.append([slope, y_intercept])
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize

# The video feed is read in as a VideoCapture object
cap = cv2.VideoCapture("whiteline.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frameSize = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
# VideoWriter Objects are created to record videos
canny_out = cv2.VideoWriter('canny_video_out.mp4', fourcc, 25, frameSize,0)
roi_out = cv2.VideoWriter('roi_video_out.mp4', fourcc, 25, frameSize,0)
final_out = cv2.VideoWriter('lane_video_out.mp4', fourcc, 25, frameSize)


while (cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    if ret is True:
        # frame = cv2.flip(frame, flipCode=1)
        canny = performCannyEdgeDetection(frame)
        cv2.imshow("canny", canny)
        canny_out.write(canny)
        height, width  = canny.shape

        # Creates a triangular polygon for the mask defined by three (x, y) coordinates
        polygons = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)

        # Creates an image filled with zero intensities with the same dimensions as the frame
        mask = np.zeros_like(canny)
        # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
        cv2.fillPoly(mask, polygons, 255)
        # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
        segment = cv2.bitwise_and(canny, mask)
        roi_out.write(segment)

        # Perform Hough Tramsform
        hough = cv2.HoughLinesP(segment, 0.75, np.pi / 180, 30, np.array([]), minLineLength = 5, maxLineGap = 10000)

        left_seg = np.copy(segment[int(height/2):height, :int(width/2)])
        right_seg = np.copy(segment[int(height/2):height, int(width/2):])

        left_count = np.count_nonzero(left_seg)
        right_count = np.count_nonzero(right_seg)
        # print(left_count,right_count)

        counter_left = 0
        counter_right = 0

        for line in hough:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]

            slope = (y2 - y1)/ (x2 - x1)
            if slope == 0:
                continue

            y_intercept = y1 - (slope*x1)

            if y1>y2:
                y_point = y1
                x_point = x1
            else:
                y_point = y2
                x_point = x2

            x_limit = int(((int(height/2)+50) - y_intercept)/slope)
            y_limit = int(height/2) + 50

            # Highlight Green and Red lines over the detected solid and dashed lanes
            if right_count >= left_count:
                if (slope < 0) and (counter_left == 0):
                    cv2.line(frame, (x_limit,y_limit), (x_point,y_point), (0,0,255),3)
                    counter_left += 1 
                elif (slope >= 0) and (counter_right == 0):
                    cv2.line(frame, (x_limit,y_limit), (x_point,y_point), (0,255,0),3)
                    counter_right += 1
            else:
                if (slope < 0) and (counter_left == 0):
                    cv2.line(frame, (x_limit,y_limit), (x_point,y_point), (0,255,0),3)
                    counter_left += 1 
                elif (slope >= 0) and (counter_right == 0):
                    cv2.line(frame, (x_limit,y_limit), (x_point,y_point), (0,0,255),3)
                    counter_right += 1

        # Opens a new window and displays the output frame
        cv2.imshow("output", frame)
        final_out.write(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
# The following frees up resources and closes all windows
cap.release()
cv2.destroyAllWindows()