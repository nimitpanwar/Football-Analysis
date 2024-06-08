from ultralytics import YOLO

model = YOLO('yolov8m') # github.com/ultralytics/ultralytics

results = model.predict('input_videos/08fd33_4.mp4', save=True)

print(results[0]) # first frame

print('___________________________')
for box in results[0].boxes:    
    print(box)