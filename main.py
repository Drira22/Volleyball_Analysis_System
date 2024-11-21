from utils import read_video, save_video, read_video_masked
from trackers import Tracker
import cv2 
import numpy as np 



def main():
    #read video 
    #video_frames = read_video('./input/videoplayback-1.mp4')
    video_frames = read_video('./input/rally_men.mp4')
    video_frames_masked = read_video_masked('./input/rally_men.mp4')

    tracker = Tracker('./models/people/best.pt')

    tracks=tracker.get_object_tracks(video_frames_masked, read_from_stub=True, stub_path="stubs/track_stubs.pkl")
    print(tracks["people"][0])


    #draw output
    output_videos_frames=tracker.draw_annotations(video_frames_masked,tracks)

    #save video 
    save_video(output_videos_frames,'output/output_video.avi')



if __name__ == '__main__':
    main()