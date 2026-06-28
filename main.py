from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    risk = "SAFE"

    height, width, _ = frame.shape

    # Smaller danger zone (bottom center)
    zone_width = width // 4
    zone_height = height // 4

    zone_x1 = (width - zone_width) // 2
    zone_y1 = height - zone_height
    zone_x2 = zone_x1 + zone_width
    zone_y2 = height

    annotated_frame = frame.copy()

    # Draw danger zone
    

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name not in ["cat", "dog"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw only the animal bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                class_name,
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Animal center position
            center_y = (y1 + y2) // 2
            print(f"Animal center Y: {center_y}")
# Frame-based risk
            if center_y > height * 0.75:
                risk = "HIGH"
            elif center_y > height * 0.55:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            if risk == "HIGH":
                color = (0, 0, 255)
            elif risk == "MEDIUM":
                color = (0, 165, 255)
            elif risk == "LOW":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.putText(
                annotated_frame,
                f"Risk: {risk}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.putText(
        annotated_frame,
        f"Overall Risk: {risk}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("SafeMove Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()