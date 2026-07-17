#  SafeMove Detector

### AI-Based Under-Vehicle Animal Detection and Driver Alert System

SafeMove Detector is an AI-powered computer vision system that detects animals underneath parked vehicles in real time and alerts the driver before the vehicle is moved. The project aims to reduce accidental harm to animals that seek shelter beneath parked vehicles.

Using **YOLOv8**, **OpenCV**, and **Python**, the system continuously monitors the area beneath a vehicle through a camera. Whenever an animal is detected, it immediately displays a visual warning and triggers an audible alarm, allowing the driver to take appropriate action before starting the vehicle.

---

##  Problem Statement

Small animals such as cats, dogs, and birds often seek shelter underneath parked vehicles because they provide warmth, shade, and protection. Drivers are usually unaware of their presence, which can lead to accidental injuries when the vehicle is moved.

SafeMove Detector addresses this problem by providing a real-time AI-based monitoring system that alerts the driver whenever an animal is detected beneath the vehicle.

---

##  Features

- Real-time animal detection
-  Detects Cats, Dogs and Birds using YOLOv8
-  Live webcam monitoring
-  Audible alarm for driver notification
-  Visual warning interface
-  Bounding boxes with confidence score
-  Lightweight and easy to run

---

##  Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Computer Vision
- Winsound

---

##  Project Structure

```text
SafeMove-Detector/
│
├── screenshots/
│   ├── demo.mp4
│   ├── safe_state.png
│   ├── cat_detection.png
│   ├── dog_detection.png
│
├── models/
│   └── yolov8n.pt
│
├── sounds/
│   └── alarm sound.wav
│
├── screenshots/
│
├── main.py
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

##  Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SafeMove-Detector.git
```

Navigate into the project directory

```bash
cd SafeMove-Detector
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
python main.py
```

---

##  Project Output

###  Safe State

![Safe State](assets/safe_state.png)

---

###  Cat Detection

![Cat Detection](assets/cat_detection.png)

---

###  Dog Detection

![Dog Detection](assets/dog_detection.png)

---
## Download YOLOv8 Model

Download the pretrained model from Ultralytics:

https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

Place the downloaded file inside:

models/yolov8n.pt


##  Project Demonstration

The complete working demonstration of the project is available in the repository.

 **Demo Video:** `assets/demo.mp4`

The demonstration showcases:

- System startup
- Live camera monitoring
- Animal detection
- Driver warning
- Audible alarm
- Safe state restoration

---

##  System Workflow

```text
          Camera
             │
             ▼
     Capture Live Frame
             │
             ▼
     YOLOv8 Animal Detection
             │
             ▼
      Animal Detected?
        │          │
       No         Yes
        │          │
 Continue        Display Warning
Monitoring          │
                    ▼
              Trigger Alarm
                    │
                    ▼
               Alert Driver
```

---

##  Future Enhancements

- Raspberry Pi deployment
- Camera installation beneath the vehicle
- Night vision camera support
- Mobile notification system
- Custom-trained model for additional small animals (rabbit, squirrel, rat, etc.)
- Detection logging with date and time
- Voice-based warning system

---

##  License

This project is licensed under the MIT License.

---

##  Author

**R. Varshitha**

B.Tech Computer Science and Engineering

Vardhaman College of Engineering

GitHub: https://github.com/varshitha0705

LinkedIn: https://www.linkedin.com/in/varshitha-r-9387632a6/

---

### ⭐ If you found this project interesting, consider giving it a star on GitHub!
