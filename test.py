from utils import read_video, save_video
from trackers import Tracker
import cv2 
import numpy as np 



def main():

    video_frames = read_video('./input/videoplayback-1.mp4')


    tracker = Tracker('./models/people/best.pt', './models/court/best.pt')

    results = tracker.detect_court(video_frames)

    print(results[0].boxes.data[0].cpu().numpy())



if __name__ == '__main__':
    main()