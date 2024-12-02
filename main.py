from utils import read_video, save_video
from trackers import Tracker
import cv2 
import numpy as np 



def main():
    # Read video
    # video_frames = read_video('./input/rally_men.mp4')
    video_frames = read_video('./input/videoplayback-1.mp4')


    # Initialize the tracker with player and court models
    tracker = Tracker('training/runs/detect/train10/weights/best.pt')

    # Get tracks
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path="stubs/track_stubs.pkl")

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output/output_video.avi')




if __name__ == '__main__':
    main()