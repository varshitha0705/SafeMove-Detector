from ultralytics import YOLO
import cv2
import winsound

# ==========================
# LOAD YOLO MODEL
# ==========================
model = YOLO("models/yolov8n.pt")

# ==========================
# OPEN CAMERA
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# ==========================
# ANIMAL CLASSES
# ==========================
animal_classes = [
    "cat",
    "dog",
    "bird"
]

# ==========================
# ALARM STATUS
# ==========================
alarm_on = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    annotated_frame = frame.copy()

    animal_detected = False

    status = "SAFE"

    results = model(frame, verbose=False)

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            class_name = model.names[cls]
            confidence = float(box.conf[0])

            # Ignore low confidence detections
            if confidence < 0.50:
                continue

            # Detect only required animals
            if class_name not in animal_classes:
                continue

            animal_detected = True
            status = f"{class_name.upper()} DETECTED"

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 0, 255)

            # Bounding Box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Label
            label = f"{class_name} {confidence * 100:.1f}%"

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                annotated_frame,
                f"{class_name.upper()} DETECTED",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ==========================
    # ALARM
    # ==========================
    if animal_detected:

        if not alarm_on:

            winsound.PlaySound(
                "sounds/alarm sound.wav",
                winsound.SND_ASYNC | winsound.SND_LOOP
            )

            alarm_on = True

    else:

        if alarm_on:

            winsound.PlaySound(
                None,
                winsound.SND_PURGE
            )

            alarm_on = False

    # ==========================
    # STATUS BAR
    # ==========================
    if animal_detected:
        status_color = (0, 0, 255)
    else:
        status_color = (0, 255, 0)

    cv2.rectangle(
        annotated_frame,
        (0, 0),
        (640, 60),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        annotated_frame,
        "SafeMove Detector",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Status : {status}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    # ==========================
    # WARNING SCREEN
    # ==========================
    if animal_detected:

        height, width = annotated_frame.shape[:2]

        cv2.rectangle(
            annotated_frame,
            (0, 0),
            (width - 1, height - 1),
            (0, 0, 255),
            6
        )

        cv2.putText(
            annotated_frame,
            "WARNING!",
            (170, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            3
        )

        cv2.putText(
            annotated_frame,
            "Animal Detected Under Vehicle",
            (60, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    # ==========================
    # DISPLAY
    # ==========================
    cv2.imshow(
        "SafeMove Detector",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# EXIT
# ==========================
winsound.PlaySound(
    None,
    winsound.SND_PURGE
)

cap.release()
cv2.destroyAllWindows()