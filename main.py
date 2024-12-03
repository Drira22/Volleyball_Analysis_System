from utils import read_video, save_video
from trackers import Tracker
import cv2 
import numpy as np 
from court import Court




def main():
    # Read video
    video_frames = read_video('./input/rally_men.mp4')
    # video_frames = read_video('./input/videoplayback-1.mp4')

    #Draw Court on the frames 
    court = Court(model_path="./models/court/best.pt", frames=video_frames, output_video_path="./output/court.avi" )
    video_frames  = court.process_video()

    # Initialize the tracker with player and court models
    tracker = Tracker('models/people/best.pt')

    # Get tracks
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path="stubs/track_stubs.pkl")

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output/output_video.avi')




if __name__ == '__main__':
    main()