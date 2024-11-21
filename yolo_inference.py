from ultralytics import YOLO 
import torch 

# Load the YOLOv8 model
model = YOLO('yolov8l.pt')  

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Model is using device: {next(model.model.parameters()).device}")

# Specify the output directory
output_dir = 'output'

# Analyze the video and save predictions
results = model.predict(
    source='image.png',  # Path to the input video
    save=True,                          # Save the output video
)

# Inspect the results
print(results[0])
print('##################################')
for box in results[0].boxes:
    print(box)
    break


