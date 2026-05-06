import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import urllib.request

# ================= SERIAL =================
ser = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)

# ================= ESP STREAM =================
esp_url = "http://10.156.7.109:81/stream"

bytes_data = b''
stream = None
esp_frame = None   # store last valid frame

try:
    print("Connecting to ESP32 stream...")
    stream = urllib.request.urlopen(esp_url, timeout=5)
    print("ESP32 stream connected!")
except Exception as e:
    print("ESP STREAM ERROR:", e)

# ================= CAMERA =================
control_cam = cv2.VideoCapture(0)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# ================= STATE =================
tracking_enabled = False
origin = None
prev = [0, 0, 0]
alpha = 0.3

# ================= P CONTROLLER =================
kp = 0.001
kp_step = 0.05
kp_min = 0.001
kp_max = 5.0

# ================= LOOP =================
while True:

    ret1, frame = control_cam.read()

    # ===== ESP STREAM READ (FIXED) =====
    if stream is not None:
        try:
            bytes_data += stream.read(8192)

            while True:
                start = bytes_data.find(b'\xff\xd8')
                end = bytes_data.find(b'\xff\xd9')

                if start != -1 and end != -1 and end > start:
                    jpg = bytes_data[start:end+2]
                    bytes_data = bytes_data[end+2:]

                    if len(jpg) > 1000:
                        img = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )

                        if img is not None:
                            esp_frame = img
                            break
                else:
                    break

        except:
            stream = None

    if not ret1:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (960, 720))

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and tracking_enabled:

        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        # ===== BOUNDING BOX (ADDED BACK) =====
        x_list = [int(l.x * w) for l in lm]
        y_list = [int(l.y * h) for l in lm]

        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        # ===== CENTROID =====
        ids = [0, 5, 9, 13, 17, 4, 8, 12, 16, 20]
        cx = int(np.mean([lm[i].x for i in ids]) * w)
        cy = int(np.mean([lm[i].y for i in ids]) * h)

        # ===== DEPTH =====
        wrist = lm[0]
        middle = lm[9]
        size = np.sqrt((wrist.x - middle.x)**2 + (wrist.y - middle.y)**2)

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

        # ===== P CONTROL =====
        x, y, z = kp*x, kp*y, kp*z

        # ===== SMOOTH =====
        x = alpha*x + (1-alpha)*prev[0]
        y = alpha*y + (1-alpha)*prev[1]
        z = alpha*z + (1-alpha)*prev[2]
        prev = [x, y, z]

        # ===== SERVO =====
        base = np.clip(int(90 + x), 0, 180)
        shoulder = np.clip(int(90 + y), 0, 180)
        elbow = np.clip(int(90 - z), 0, 180)
        wrist_servo = 90

        # ===== PINCH (IMPROVED) =====
        thumb = lm[4]
        index = lm[8]
        pinch = np.hypot(thumb.x - index.x, thumb.y - index.y)

        if pinch < 0.035:
            gripper = 0
        elif pinch > 0.05:
            gripper = 180

        cmd = f"{base},{shoulder},{elbow},{wrist_servo},{gripper}\n"
        ser.write(cmd.encode())

        cv2.circle(frame, (cx, cy), 10, (0,255,255), -1)
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # ================= UI PANEL =================
    panel = np.zeros((720, 300, 3), dtype=np.uint8)

    cyan = (255, 255, 0)
    magenta = (255, 0, 255)
    green = (0,255,0)
    red = (0,0,255)

    status_text = "RUNNING" if tracking_enabled else "PAUSED"

    cv2.putText(panel, "ROBOT CONTROL", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, magenta, 2)

    cv2.putText(panel, f"KP: {kp:.3f}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cyan, 2)

    cv2.putText(panel, "STATUS:", (20,200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cyan, 2)

    cv2.putText(panel, status_text, (20,240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                green if tracking_enabled else red, 2)

    cv2.putText(panel, "+ / - : KP", (20,350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cyan, 1)

    cv2.putText(panel, "R : START", (20,400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cyan, 1)

    cv2.putText(panel, "P : PAUSE", (20,450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cyan, 1)

    cv2.putText(panel, "Q : EXIT", (20,500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cyan, 1)

    # ================= ESP DISPLAY =================
    if esp_frame is not None:
        try:
            esp_small = cv2.resize(esp_frame, (240,180))
            frame[10:190, w-250:w-10] = esp_small
        except:
            pass

    # ================= MERGE =================
    final_ui = np.hstack((frame, panel))
    cv2.imshow("ROBOT CONTROL SYSTEM", final_ui)

    # ================= KEYS =================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        tracking_enabled = False
    elif key == ord('r'):
        tracking_enabled = True

    if key == ord('+') or key == ord('='):
        kp = min(kp + kp_step, kp_max)
    elif key == ord('-') or key == ord('_'):
        kp = max(kp - kp_step, kp_min)

    if key == 27 or key == ord('q'):
        break

control_cam.release()
cv2.destroyAllWindows()