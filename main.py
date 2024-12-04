from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from court import Court
from ultralytics import YOLO
from ball import Ball
from time import sleep
import gc

def main():
    # Read video in chunks or reduce resolution if memory is tight
    video_frames = read_video('./input/videoplayback-1.mp4')
    
    # Draw Court on the frames
    court = Court(model_path="./models/court/best.pt", frames=video_frames, output_video_path="./output/court.avi")
    video_frames = court.process_video()
    del court
    gc.collect()
    
    sleep(5)

    # Process ball tracking
    ball = Ball(model_path='models/ball/best.pt', frames=video_frames, output_video_path="./output/ball.avi")
    video_frames = ball.process_video()
    del ball
    gc.collect()

    # Initialize the tracker
    tracker = Tracker('models/players/best.pt')

    # Get tracks
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path="stubs/track_stubs.pkl")

    for track_id,player in tracks['player'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]
        cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        cv2.imwrite(f'./developement _and_analysis/cropped_image.jpg',cropped_image)
        break

    # Draw output and save
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, 'output/output_video_1.avi')

    del tracker, tracks, output_video_frames
    gc.collect()

if __name__ == '__main__':
    main()
