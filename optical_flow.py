import skvideo.io
import cv2
import numpy as np
import math
import time

def combineTrackers(oldPoints, newPoints, status): # Create new feature list based on old and new points
    threshold = 5
    extraFeatures = []
    for i in range(len(newPoints)):
        new = True # Assume point is new 
        for j in range(len(oldPoints)): # Check proximty to all old points
            if status[j] == 1: # If point is still in frame
                if math.hypot(newPoints[i][0]-oldPoints[j][0], newPoints[i][1]-oldPoints[j][1]) < threshold:
                    new = False # If too close, then not new
        if new == True:
            extraFeatures += [newPoints[i].tolist()] # Otherwise add to key point list
    if len(extraFeatures) != 0:
        featureList = np.concatenate((oldPoints, np.asarray(extraFeatures)))
    else:
        featureList = oldPoints
    return featureList

video = skvideo.io.vread('frames_short_small.mp4')
t0 = time.time()
# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 200, qualityLevel = 0.3, minDistance = 2, blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Variable for color to draw optical flow track
colourList = []
for i in range(5000):
    colourList += [np.random.random(size=3) * 256]

video_bw = video[:, :, :, 0].astype('uint8')
prev_img = video_bw[0]
prev_feat = cv2.goodFeaturesToTrack(prev_img, mask = None, **feature_params)
mask = np.zeros_like(video[0])

writer = skvideo.io.FFmpegWriter("tracked_short (line).mp4") # Prepare to write video

for i in range(len(video)):
    frame = video[i]
    frame_bw = video_bw[i]

    nxt, status, error = cv2.calcOpticalFlowPyrLK(prev_img, frame_bw, prev_feat, None, **lk_params)
    # Selects good feature points for previous position
    good_old = prev_feat[:, 0, :]
    # Selects good feature points for next position
    good_new = nxt[:, 0, :]

    for j in range(len(good_new)):
        new_coord = tuple(good_new[j])
        old_coord = tuple(good_old[j])
        #mask = cv2.line(mask, new_coord, old_coord, (0,255,0), 2)
        frame = cv2.circle(frame, new_coord, 3, colourList[j], -1)

    mask = cv2.line(mask, tuple(good_new[14]), tuple(good_old[14]), (0,255,0), 2)
    
    if (i%20 == 0) and (i!=0):
        new_trackers = cv2.goodFeaturesToTrack(prev_img, mask = None, **feature_params)
        good_new = combineTrackers(np.rint(good_new), new_trackers[:, 0, :], status)
        good_new = np.float32(good_new)
        
    output = cv2.add(frame, mask)
    cv2.imshow("sparse optical flow", output)
    cv2.waitKey(1)
    prev_feat = good_new.reshape(-1, 1, 2)
    prev_img = frame_bw.copy()
    writer.writeFrame(output)
print(len(prev_feat))
t1 = time.time()
print(t1-t0)
writer.close()
