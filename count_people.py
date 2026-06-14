from ultralytics import YOLO

# Load the pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run detection on the image
results = model("test.jpg")

# Count how many "person" detections (class 0 in COCO)
person_count = 0
for box in results[0].boxes:
    if int(box.cls[0]) == 0:  # class 0 = person
        person_count += 1

print(f"Number of people detected: {person_count}")

# Save the image with boxes drawn around detected people
results[0].save(filename="output.jpg")