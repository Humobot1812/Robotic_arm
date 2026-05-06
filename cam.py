import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# ================= SERIAL =================
ser = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)

# ================= CAMERAS =================
esp_url = "http://10.156.7.109:81/stream"
control_cam = cv2.VideoCapture(0)
esp_cam = cv2.VideoCapture(esp_url)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

# ================= STATE =================
tracking_enabled = False
origin = None
prev = [0, 0, 0]
alpha = 0.3

# ================= P CONTROLLER (ADDED) =================
kp = 1.0
kp_step = 0.1
kp_min = 0.1
kp_max = 5.0

# ================= LOOP =================
while True:

    ret1, frame = control_cam.read()
    ret2, esp_frame = esp_cam.read()

    if not ret1:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and tracking_enabled:

        hand = result.multi_hand_landmarks[0]

        # ===== GET LANDMARKS =====
        lm = hand.landmark

        # fingertips + palm
        ids = [0, 5, 9, 13, 17, 4, 8, 12, 16, 20]

        x_vals = [lm[i].x for i in ids]
        y_vals = [lm[i].y for i in ids]

        cx = int(np.mean(x_vals) * w)
        cy = int(np.mean(y_vals) * h)

        # ===== HAND SIZE (DEPTH) =====
        wrist = lm[0]
        middle = lm[9]

        size = np.sqrt((wrist.x - middle.x)**2 + (wrist.y - middle.y)**2)

        # ===== BOUNDING BOX =====
        x_list = [int(l.x * w) for l in lm]
        y_list = [int(l.y * h) for l in lm]

        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)

        # ===== CALIBRATION =====
        if origin is None:
            origin = (cx, cy, size)
            print("CALIBRATED")
            continue

        dx = cx - origin[0]
        dy = origin[1] - cy
        dz = size - origin[2]

        # ===== NORMALIZE =====
        x = dx / (w/2) * 90
        y = dy / (h/2) * 90
        z = dz * 300

        # ===== P CONTROLLER (ADDED) =====
        x = kp * x
        y = kp * y
        z = kp * z

        # ===== SMOOTHING =====
        x = alpha*x + (1-alpha)*prev[0]
        y = alpha*y + (1-alpha)*prev[1]
        z = alpha*z + (1-alpha)*prev[2]

        prev = [x, y, z]

        # ===== MAP TO SERVOS =====
        base = int(90 + x)
        shoulder = int(90 + y)
        elbow = int(90 - z)

        base = np.clip(base, 0, 180)
        shoulder = np.clip(shoulder, 0, 180)
        elbow = np.clip(elbow, 0, 180)

        wrist_servo = 90

        # ===== PINCH (GRIPPER) =====
        thumb = lm[4]
        index = lm[8]

        pinch = np.hypot(thumb.x - index.x, thumb.y - index.y)

        if pinch < 0.04:
            gripper = 0
        else:
            gripper = 180

        # ===== SEND =====
        cmd = f"{base},{shoulder},{elbow},{wrist_servo},{gripper}\n"
        ser.write(cmd.encode())

        # ===== DRAW =====
        cv2.circle(frame, (cx, cy), 10, (0,255,0), -1)

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # ===== DISPLAY =====
    cv2.putText(frame, f"KP: {kp:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)  # ADDED
    
    status_text = "RUNNING" if tracking_enabled else "PAUSED"

    cv2.putText(frame, f"STATUS: {status_text}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0) if tracking_enabled else (0,0,255), 2)

    cv2.imshow("CONTROL CAM", frame)

    if ret2:
        cv2.imshow("ROBOT VIEW", esp_frame)  

    key = cv2.waitKey(1) & 0xFF
    
    # ===== TRACKING CONTROL =====
    if key == ord('p'):
        tracking_enabled = False
        print("Tracking PAUSED")

    elif key == ord('r'):
        tracking_enabled = True
        print("Tracking RESUMED")
    
    # ===== KP CONTROL (ADDED) =====
    if key == ord('+') or key == ord('='):
        kp = min(kp + kp_step, kp_max)
        print(f"KP increased → {kp:.2f}")

    elif key == ord('-') or key == ord('_'):
        kp = max(kp - kp_step, kp_min)
        print(f"KP decreased → {kp:.2f}")

    # ===== EXIT =====
    if key == 27 or key == ord('q'):
        break

control_cam.release()
esp_cam.release()
cv2.destroyAllWindows()