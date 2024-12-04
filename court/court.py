from ultralytics import YOLO
import torch
from utils import read_video, save_video
import cv2
import os


class Court:
    def __init__(self, model_path, frames, output_video_path):
        self.model_path = model_path
        self.frames = frames
        self.output_video_path = output_video_path

        # Initialize YOLO model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.model_path)
        self.model.to(self.device)


    def draw_court_box(self, frame, bbox):
        image = frame.copy()
        x_min, y_min, x_max, y_max, confidence, class_id = bbox
        if confidence>0.5:
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

    def process_video(self):
        processed_frames = []

        for i, frame in enumerate(self.frames):
            # Run YOLO model on the frame
            results = self.model.predict(frame, device=self.device)

            # Process results for each frame
            if len(results[0].boxes): 
                for box in results[0].boxes.data.cpu().numpy():
                    frame = self.draw_court_box(frame, box)

            processed_frames.append(frame)

        # Save processed video
        os.makedirs(os.path.dirname(self.output_video_path), exist_ok=True)
        save_video(processed_frames, self.output_video_path)
        print(f"Processed video saved to {self.output_video_path}")

        return processed_frames


##### to use this class here is an example 

# if __name__ == '__main__':
#     # Initialize Court class with paths
#     court_processor = Court(
#         model_path='models/court/best.pt',
#         frames,
#         output_video='output/court.avi'
#     )

#     # Process the video
#     court_processor.process_video()