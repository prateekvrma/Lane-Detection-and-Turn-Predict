#!/usr/env python
import numpy as np
import cv2
# import matplotlib.pyplot as plt

def histogram(image):
    his = np.mean(image[470:, :],axis=0)
    return his

def get_curvature(img, left_x, right_x):
    ploty = np.linspace(0,img.shape[1] - 1, img.shape[1])
    y_eval = np.max(ploty)
    y_mperpix = 30.5/720
    x_mperpix = 3.7/1280

    p_l_cr = np.polyfit(ploty*y_mperpix, left_x*x_mperpix, 2)
    p_r_cr = np.polyfit(ploty*y_mperpix, right_x*x_mperpix, 2)

    left_c_rad_status = True
    right_c_rad_status = True
    left_c_rad = ((1+(2*p_l_cr[0]*y_eval*y_mperpix + p_l_cr[1]**2)**1.5)/abs(2*p_l_cr[0]))
    right_c_rad = ((1+(2*p_r_cr[0]*y_eval*y_mperpix + p_r_cr[1]**2)**1.5)/abs(2*p_r_cr[0]))
    if left_c_rad == np.nan:
        left_c_rad_status = False
    if right_c_rad == np.nan:
        right_c_rad_status = True

    pos = img.shape[0]/2
    l_fit_x_int = p_l_cr[0]*img.shape[1]**2 + p_l_cr[1]*img.shape[0] + p_l_cr[2]
    r_fit_x_int = p_r_cr[0]*img.shape[1]**2 + p_r_cr[1]*img.shape[0] + p_r_cr[2]
    lane_center = (l_fit_x_int + r_fit_x_int)/2
    center = (pos - lane_center)*x_mperpix/10
    return left_c_rad, right_c_rad, center, left_c_rad_status, right_c_rad_status

def draw_lanes(img, left_fit, right_fit, rad, turn_dir):
    ploty = np.linspace(0,img.shape[1] - 1, img.shape[1])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit,ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit,ploty])))])

    color_img = cv2.polylines(color_img, np.int_(left), False, (255,0,0), 50)
    color_img = cv2.polylines(color_img, np.int_(right), False, (255,0,0),50)
    pts = np.hstack((left, right))
    cv2.fillPoly(color_img, np.int_(pts), (0, 0, 255))
    dst_pts = np.array([[564, 470],[766, 470],[0.897*frame_width, frame_height],[0.125*frame_width, frame_height]]).astype(np.int32)
    src_pts = np.array([[0.125*frame_width, 0],[0.897*frame_width, 0],[0.897*frame_width, frame_height],[0.125*frame_width, frame_height]])
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    dst_2 = cv2.warpPerspective(color_img, H, (frame_width,int(frame_height)))          
    dst_2 = cv2.addWeighted(img, 1, dst_2, 0.7, 0)
    # if rad >=0:
    #     dst_2 = cv2.putText(dst_2, "Radius of Curvature (in meters): " + str(rad), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0, 100), 3, cv2.LINE_AA)
    # else:
    #     dst_2 = cv2.putText(dst_2, "Radius of Curvature not found", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0, 100), 3, cv2.LINE_AA)
    
    if turn_dir == 0:
        dst_2 = cv2.putText(dst_2, "Right Turn", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0, 100), 3, cv2.LINE_AA)
    elif turn_dir == 1:
        dst_2 = cv2.putText(dst_2, "Left Turn", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0, 100), 3, cv2.LINE_AA)
    elif turn_dir == 2:
        dst_2 = cv2.putText(dst_2, "Stay Striaght", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0, 100), 3, cv2.LINE_AA)
    return dst_2

if __name__ == '__main__':
    # Create a VideoCapture object
    # cap = cv2.VideoCapture('challenge.mp4') # input video
    cap = cv2.VideoCapture('whiteline.mp4') # input video
    
    
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read input video feed - Check the source input.")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frameSize = (2*frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    # VideoWriter Objects are created to record videos
    out_video = cv2.VideoWriter('Prob3_video_whiteline.mp4', fourcc, 25, frameSize)
    prev_p_l = None
    prev_p_r = None
    while(True):
        ret, frame = cap.read()
        if(ret == True):
            # To check for Horizontally flipped video,can uncomment this line below
            # frame = cv2.flip(frame, flipCode=1)
            
            roi_pts = np.array([[564, 470],[766, 470],[0.897*frame_width, frame_height],[0.125*frame_width, frame_height]]).astype(np.int32)
            dst_pts = np.array([[0.125*frame_width, 0],[0.897*frame_width, 0],[0.897*frame_width, frame_height],[0.125*frame_width, frame_height]])
            H, _ = cv2.findHomography(roi_pts, dst_pts, cv2.RANSAC, 5.0)
            dst = cv2.warpPerspective(frame, H, (frame_width,int(frame_height)))
            hsv_2 = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

            # Threshold the HSV image to get only white colors
            lower_white = np.array([0,0,240], dtype=np.uint8)
            upper_white = np.array([145,145,255], dtype=np.uint8)
            mask_white_2 = cv2.inRange(hsv_2, lower_white, upper_white)
            
            lower_y = np.array([20, 100, 100], dtype=np.uint8)
            upper_y = np.array([30,255,255], dtype=np.uint8)
            
            mask_y_2 = cv2.inRange(hsv_2, lower_y, upper_y)
            mask_2 = mask_white_2 + mask_y_2
            modified_frame = cv2.bitwise_and(dst, dst, mask=mask_2)
            bi_2 = cv2.bilateralFilter(cv2.cvtColor(modified_frame, cv2.COLOR_BGR2GRAY), 5, 75, 75)
            h = histogram(bi_2)
            midpt = int(h.shape[0]/2)
            leftx_base = np.argmax(h[:midpt])
            rightx_base = np.argmax(h[midpt:]) + midpt
            win_height =  np.int(modified_frame.shape[1]/9)

            nonzero = modified_frame.nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])
            leftx_curr = leftx_base
            rightx_curr = rightx_base

            left_lane_ind = []
            right_lane_ind = []

            left_a, left_b, left_c=[],[],[]
            right_a, right_b, right_c=[],[],[]
            p_l_m =np.empty(3)
            p_r_m = np.empty(3)
            
            margin = 150

            for window in range(9):
                win_y_low = modified_frame.shape[1] - (window + 1)*win_height
                win_y_high = modified_frame.shape[1] - (window)*win_height
                win_xleft_low = leftx_curr - margin
                win_xleft_high = leftx_curr + margin
                win_xright_low = rightx_curr - margin
                win_xright_high = rightx_curr + margin
                cv2.rectangle(modified_frame,(win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 3)
                cv2.rectangle(modified_frame,(win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 3)
                good_l_ind = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
                good_r_ind = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

                left_lane_ind.append(good_l_ind)
                right_lane_ind.append(good_r_ind)

                if len(good_l_ind) > 100:
                    leftx_curr = np.int(np.mean(nonzero_x[good_l_ind]))
                if len(good_r_ind) > 100:
                    rightx_curr = np.int(np.mean(nonzero_x[good_r_ind]))
            left_lane_ind = np.concatenate(left_lane_ind)
            right_lane_ind = np.concatenate(right_lane_ind)
            left_x = nonzero_x[left_lane_ind]
            left_y = nonzero_y[left_lane_ind]
            right_x = nonzero_x[right_lane_ind]
            right_y = nonzero_y[right_lane_ind]                
            if (len(left_x) == 0) or (len(left_y) == 0) or (len(right_x) == 0) or (len(right_y) == 0):
                p_left = prev_p_l
                p_right = prev_p_r
            else:
                p_left = np.polyfit(left_y, left_x, 2)
                p_right = np.polyfit(right_y, right_x, 2)
                if (p_left == np.nan).any() or (p_right == np.nan).any() or p_left is None or p_right is None:
                    p_left = prev_p_l
                    p_right = prev_p_r
                else:
                    prev_p_l = p_left
                    prev_p_r = p_right

            left_a.append(p_left[0])
            left_b.append(p_left[1])
            left_c.append(p_left[2])
            right_a.append(p_right[0])
            right_b.append(p_right[1])
            right_c.append(p_right[2])

            p_l_m[0] = np.mean(left_a[-10:])
            p_l_m[1] = np.mean(left_b[-10:])
            p_l_m[2] = np.mean(left_c[-10:])
            p_r_m[0] = np.mean(right_a[-10:])
            p_r_m[1] = np.mean(right_b[-10:])
            p_r_m[2] = np.mean(right_c[-10:])

            ploty = np.linspace(0,modified_frame.shape[1]-1, modified_frame.shape[1])
            left_fit_x = p_l_m[0]*ploty**2 + p_l_m[1]*ploty + p_l_m[2]
            right_fit_x = p_r_m[0]*ploty**2 + p_r_m[1]*ploty + p_r_m[2]

            if ((p_l_m[0] +p_r_m[0])/2 > 0):
                turn_dir = 0
            elif ((p_l_m[0] +p_r_m[0])/2 < 0):
                turn_dir = 1
            else:
                turn_dir = 2

            modified_frame[nonzero_y[left_lane_ind], nonzero_x[left_lane_ind]] = [255,0,255]
            modified_frame[nonzero_y[right_lane_ind], nonzero_x[right_lane_ind]] = [255,0,255]
            
            curved = get_curvature(modified_frame, left_fit_x, right_fit_x)
            if curved[3] is not np.nan and curved[4] is not np.nan:
                mean_rad = (curved[0] + curved[1])//2
            else:
                mean_rad = -1
            

            frame_1 = draw_lanes(frame, left_fit_x, right_fit_x, mean_rad, turn_dir)   
            frame_2 = modified_frame

            res_comb = np.zeros((frame_height, 2*frame_width, 3), np.uint8)
            res_comb[:,:frame_width]= frame_1
            res_comb[:,frame_width:]= modified_frame
            
            cv2.imshow('Combined Output', res_comb)
            
            out_video.write(res_comb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()
    out_video.release()