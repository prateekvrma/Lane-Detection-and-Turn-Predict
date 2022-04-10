#!/usr/env python
import numpy as np
import cv2
import glob

# Histogram Equalization
def histogram_equalization(frame):
    height,width=frame.shape
    pixel_count = height*width
    histrogram = np.zeros((256,1))

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            histrogram[frame[i][j]] += 1
    
    CDF = np.zeros((256,1))
    for p in range(256):
        CDF[p] = np.sum(histrogram[0:p+1])/pixel_count
    
    equalized_frame = np.zeros(frame.shape, dtype=np.uint8)
    for q in range(height):
        for r in range(width):
            equalized_frame[q,r] = CDF[frame[q][r]] * 255

    return equalized_frame.astype(np.uint8)

# Adaptive Histogram Equalization
def adaptive_histogram_equalize(frame):
    # Dividing image into 8 x 8 = 64 tiles
    tile_height = int(frame.shape[0]/8)
    tile_width = int(frame.shape[1]/8)
    adapt_y = frame.copy()
    for i in range(0, frame.shape[0], tile_height):
        for j in range(0, frame.shape[1], tile_width):
            if i+tile_height <= frame.shape[0] and j+tile_width <= frame.shape[1]:
                adapt_y[i:i+tile_height, j:j+tile_width] = histogram_equalization(frame[i:i+tile_height,j:j+tile_width])
    return adapt_y

if __name__ == '__main__':
    cv_img = []
    for img in sorted(glob.glob("./adaptive_hist_data/*.png")):
        n = cv2.imread(img)
        cv_img.append(n)
    frameSize = (cv_img[0][:,:,0].shape[1], cv_img[0][:,:,0].shape[0])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_eq = cv2.VideoWriter('Histogram_Equalization.mp4',fourcc, 2, frameSize)
    out_adapt = cv2.VideoWriter('Adaptive_Histogram_Equalization.mp4',fourcc, 2, frameSize)
    for img in cv_img:
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        channel_y = ycrcb_img[:, :, 0].copy()
        
        # Standard Histogram Equalization
        ycrcb_img[:,:,0] = histogram_equalization(channel_y)
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        
        # Adaptive Histogram Equalization
        new_ycrcb = ycrcb_img.copy()
        new_ycrcb[:,:,0] = adaptive_histogram_equalize(channel_y)
        adapt_eq_img = cv2.cvtColor(new_ycrcb, cv2.COLOR_YCrCb2BGR)

        out_eq.write((equalized_img).astype(np.uint8))
        out_adapt.write((adapt_eq_img).astype(np.uint8))
    out_eq.release()
    out_adapt.release()