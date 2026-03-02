import cv2
import numpy as np
from ultralytics import YOLO
from simple_pid import PID

# ---------------------------
# PID CONTROLLERS
# ---------------------------
pid_yaw = PID(1.2, 0.0, 0.25, setpoint=0)
pid_yaw.output_limits = (-1, 1)

pid_dist = PID(0.8, 0.0, 0.2, setpoint=0)
pid_dist.output_limits = (-1, 1)

# ---------------------------
# CAMERA + MODEL
# ---------------------------
cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")

tracker = None
tracking = False
frame_count = 0

# ---------------------------
# DISTANCE CALIBRATION VALUES
# ---------------------------
REAL_PERSON_HEIGHT = 1.70   # meters (your height)
FOCAL_LENGTH = 260          # <-- replace after calibration
TARGET_DISTANCE = 2.0       # desired follow distance (meters)

# ---------------------------
# KALMAN FILTER
# ---------------------------
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.02
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3


def detect_person(frame):
    results = model(frame, verbose=False)
    best_box = None
    best_conf = 0

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                conf = float(box.conf[0])
                if conf > 0.5 and conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_box = (x1, y1, x2, y2)
    return best_box


def person_exists(frame):
    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                return True
    return False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_h, frame_w = frame.shape[:2]

    if not tracking:
        box = detect_person(frame)

        if box is not None:
            x1, y1, x2, y2 = box

            tracker = cv2.TrackerCSRT_create()

            w = x2 - x1
            h = y2 - y1

            tracker.init(frame, (x1, y1, w, int(h * 0.6)))
            initial_height = int(h * 0.6)
            tracking = True
            print("Person locked")

    else:
        success, box = tracker.update(frame)

        if not success:
            tracking = False
            continue

        if frame_count % 15 == 0:
            if not person_exists(frame):
                tracking = False
                continue

        x, y, w_box, h_box = [int(v) for v in box]

        if h_box <= 0:
            tracking = False
            continue

        # ---------------------------
        # TORSO ANCHOR
        # ---------------------------
        center_x = x + w_box / 2
        center_y = y + h_box * 0.35

        measurement = np.array([[np.float32(center_x)],
                                [np.float32(center_y)]])

        kalman.correct(measurement)
        prediction = kalman.predict()

        pred_x = prediction[0][0]
        pred_y = prediction[1][0]

        # ---------------------------
        # YAW ERROR
        # ---------------------------
        error_x = (pred_x - frame_w/2) / (frame_w/2)

        # ---------------------------
        # GEOMETRIC DISTANCE
        # ---------------------------
        height_change = (h_box - initial_height) / initial_height
        dist_error = height_change

        # ---------------------------
        # PID CONTROL
        # ---------------------------
        yaw_cmd = pid_yaw(error_x)
        dist_cmd = pid_dist(dist_error)

        if abs(yaw_cmd) < 0.05:
            yaw_cmd = 0
        if abs(dist_cmd) < 0.05:
            dist_cmd = 0

        # ---------------------------
        # DRAW
        # ---------------------------
        cv2.rectangle(frame, (x, y),
                      (x + w_box, y + h_box),
                      (0, 255, 0), 2)

        cv2.circle(frame,
                   (int(pred_x), int(pred_y)),
                   6, (0, 0, 255), -1)

        cv2.circle(frame,
                   (frame_w//2, frame_h//2),
                   6, (255, 0, 0), -1)

        print("Yaw:", round(yaw_cmd,2),
              "Forward:", round(dist_cmd,2),
              "Dist(m):", round(current_distance,2))

    cv2.imshow("Geometric Distance Follow", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()