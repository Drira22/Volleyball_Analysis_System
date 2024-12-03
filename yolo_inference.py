from ultralytics import YOLO
import torch
from utils import read_video,save_video
import cv2
import os


def draw_court_box(frame, bbox):
    image = frame.copy()
    x_min, y_min, x_max, y_max, confidence, class_id = bbox

    
    # Draw the bounding box
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
    label = f"volleyball-court {confidence:.0%}"
    color = (255, 255, 0)

    # Draw the rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)

    # Add label
    font_scale = 1
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(image, (x_min, y_min - text_h - 10), (x_min + text_w, y_min), color, -1)
    cv2.putText(image, label, (x_min, y_min - 5), font, font_scale, (0, 0, 0), font_thickness)
    
    return image


if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO('models/court/best.pt')

    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model is using device: {device}")

    input_video = "input/rally_men.mp4"
    output_image = "processed_frame.jpg"



    # Read video frames
    video_frames = read_video(input_video)
    frames=[]
    for i,frame in enumerate(video_frames):
        # Run YOLO model on the frame
        results = model.predict(frame, device=device)

        # Process results for the first frame only
        if len(results[0].boxes): 
            for box in results[0].boxes.data.cpu().numpy():
                frame = draw_court_box(frame, box)
        
        frames.append(frame)
        print(f"Saved processed frame number {i}")
    
    save_video(frames,'output/court.avi')
    


