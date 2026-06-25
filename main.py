from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    zone_x1 = width // 3
    zone_y1 = height // 2
    zone_x2 = 2 * width // 3
    zone_y2 = height

    cv2.rectangle(
    frame,
    (zone_x1, zone_y1),
    (zone_x2, zone_y2),
    (0, 0, 255),
    2)
    if not ret:
        break

    results = model(frame,verbose=False)
    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])

            class_name = model.names[cls]

            if class_name in ["cat", "dog"]:
                print("Animal Detected:", class_name)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            overlap_x = max(0, min(x2, zone_x2) - max(x1, zone_x1))
            overlap_y = max(0, min(y2, zone_y2) - max(y1, zone_y1))
            overlap_area = overlap_x * overlap_y
            animal_area = (x2 - x1) * (y2 - y1)

            if animal_area == 0:
                continue

            overlap_ratio = overlap_area / animal_area

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if overlap_ratio > 0.6:
                risk = "HIGH"
            elif overlap_ratio > 0.3:
                risk = "MEDIUM"

        else:
            risk = "LOW"
        annotated_frame = results[0].plot()

    cv2.imshow("SafeMove Detector", annotated_frame)
    cv2.putText(
    annotated_frame,
    f"Risk: {risk}",
    (x1, y1 - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 255),
    2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()