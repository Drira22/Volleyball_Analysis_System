from ultralytics import YOLO
import supervision as sv 
import pickle
import os 
import cv2
import numpy as np
import pandas as pd 
#in order to use functions we defined in utils we need to get back one step behind in the racine thenimport the functions 
import sys
sys.path.append('../')
from utils import get_center_of_bbox,get_bbox_width

class Tracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        
        #interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() #bfill is for the first second frames who doesn't have back history to track and interpolate

        ball_positions= [{1:{"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions


    def detect_frames(self,frames):
        
        batch_size=20 
        detections=[]
        
        for i in range(0,len(frames),batch_size):
        
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.3)
            detections+=detections_batch
        
        return detections
    

    def get_object_tracks(self,frames,read_from_stub=False, stub_path=None):

        #read the tracks already saved in the stub_path no need to do all the work all over again 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks= pickle.load(f)
            return tracks
        

        #extract the frames and make them in the object detections
        detections=self.detect_frames(frames)
        print(detections[0])

        #tracks is the final result we want to detect at each fram the bboxs with appropriate label
        tracks = {
            "player":[],
        }

        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            cls_name_inv = {v:k for k,v in cls_names.items()}

            #convert to supervision format 
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #track objects 
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            
            tracks["player"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id = frame_detection[4]
                
                #no need to track the volleyball because it s one ball so the boundary box is teh same 
                #add the boundary box of a player in this frame number 
                if cls_id == cls_name_inv["player"]:
                    tracks["player"][frame_num][track_id] = {"bbox":bbox}
                    # tracks={"people":[
                    #                   { 0:{"bbox":{[0,0,0,0]}} , 14:{"bbox":{[0,0,0,0]}} , 21:{"bbox":{[0,0,0,0]}} }, this with index 0 refers to frame_num 0 
                    #                   { 10:{"bbox":{[0,0,0,0]}} , 15:{"bbox":{[0,0,0,0]}} , 24:{"bbox":{[0,0,0,0]}} } ,and so on with frame numbers 
                    #                   ]}

            


        #save the results to stub_path as pickle 
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks


    def draw_ellipse(self,frame,bbox,color,track_id=None):
        
        #we need to draw elipse around the player so we need the lowest under y 
        y2 = int(bbox[3])
        #in order to draw ellipse we need the center of bbox
        #so we defined functions in utils under bbox_utils.py to do so 
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0, 
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        #now the rectangle which appeared the fix number for each player
        rectangle_width = 40
        rectangle_height = 20 
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2-rectangle_height//2)+15 #15 is a random buffer 
        y2_rect = (y2+rectangle_height//2)+15 

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED
                          )
            x1_text =x1_rect+12    
            if track_id>99:
                x1_text-=10
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0,0,0),
                2

            )
        
        return frame


    def draw_triangle(self,frame,bbox,color):
        y=int(bbox[1])
        x,_=get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)

        return frame

    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame


    def draw_annotations(self,video_frames,tracks):
        output_video_frames =[]
        for frame_num,frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["player"][frame_num]


            #Draw Players
            for track_id,people in player_dict.items():
               
                frame = self.draw_ellipse(frame,people["bbox"],(0,0,255),track_id)


            output_video_frames.append(frame)

        return output_video_frames