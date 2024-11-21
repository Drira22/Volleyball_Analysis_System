import cv2 
import numpy as np 

def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    frames=[]

    while 1:
        ret,frame=cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def read_video_masked(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply mask to focus on court area
        height, width, _ = frame.shape
        mask = np.zeros_like(frame)
        # Adjust these values to fit your court position
        mask[int(height * 0.3):int(height * 0.8), :] = 255
        masked_frame = cv2.bitwise_and(frame, mask)
        frames.append(masked_frame)

    return frames


def save_video(output_video_frame,output_video_path):
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path,fourcc,24,(output_video_frame[0].shape[1],output_video_frame[0].shape[0]))
    for frame in output_video_frame:
        out.write(frame)
    out.release() 

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]