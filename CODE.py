import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# -------------------- MediaPipe Pose --------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- Camera --------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)


# -------------------- Pose Connections --------------------
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 31), (28, 32)
]

# -------------------- 3D Plot --------------------
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")

# -------------------- State --------------------
start_time = time.time()
frame_count = 0
trail_points = []
max_trail_length = 8

# -------------------- Helpers --------------------
def get_gradient_color(t):
    hue = (t * 120) % 360
    c = 1
    x = c * (1 - abs((hue / 60) % 2 - 1))

    if hue < 60:
        r, g, b = c, x, 0
    elif hue < 120:
        r, g, b = x, c, 0
    elif hue < 180:
        r, g, b = 0, c, x
    elif hue < 240:
        r, g, b = 0, x, c
    elif hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return int(r * 255), int(g * 255), int(b * 255)

def draw_smooth_line(img, p1, p2, color, thickness):
    cv2.line(img, p1, p2, color, thickness)
    cv2.line(img, p1, p2, tuple(c // 2 for c in color), thickness + 2)

def draw_smooth_circle(img, center, radius, color):
    cv2.circle(img, center, radius, color, -1)
    cv2.circle(img, center, radius // 2, (255, 255, 255), -1)


# -------------------- Main Loop --------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    now = time.time()
    wave_time = now - start_time

    base_color = get_gradient_color(wave_time)
    pulse = 0.8 + 0.2 * math.sin(wave_time * 6)
    neon_color = tuple(int(c * pulse) for c in base_color)

    if results.pose_landmarks:
        h, w, _ = frame.shape

        landmarks_2d = [
            (int((1 - lm.x) * w), int(lm.y * h))
            for lm in results.pose_landmarks.landmark
        ]
        landmarks_3d = [
            (1 - lm.x, lm.y, lm.z)
            for lm in results.pose_landmarks.landmark
        ]

        glow = np.zeros_like(frame, dtype=np.uint8)

        for i, j in POSE_CONNECTIONS:
            draw_smooth_line(
                glow,
                landmarks_2d[i],
                landmarks_2d[j],
                neon_color,
                6
            )

        joints = []
        for x, y in landmarks_2d:
            if 0 <= x < w and 0 <= y < h:
                joints.append((x, y))
                draw_smooth_circle(glow, (x, y), 8, neon_color)

        trail_points.append(joints.copy())
        if len(trail_points) > max_trail_length:
            trail_points.pop(0)

        for i, trail in enumerate(trail_points[:-1]):
            alpha = (i / len(trail_points)) * 0.4
            color = tuple(int(c * alpha) for c in neon_color)
            for x, y in trail:
                cv2.circle(glow, (x, y), 3, color, -1)

        frame = cv2.addWeighted(frame, 0.7, glow, 0.6, 0)

        if frame_count % 6 == 0:
            ax.clear()
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_facecolor("black")
            ax.set_title("3D Clone", color="cyan")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)

            lc = tuple(c / 255 for c in neon_color)

            for i, j in POSE_CONNECTIONS:
                x, y, z = zip(landmarks_3d[i], landmarks_3d[j])
                ax.plot(x, y, z, color=lc, linewidth=3, alpha=0.8)

            for x, y, z in landmarks_3d:
                ax.scatter(x, y, z, color=lc, s=50, alpha=0.8)

            plt.draw()
            plt.pause(0.001)

    else:
        h, w, _ = frame.shape
        text = "SCANNING..."
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        pos = ((w - size[0]) // 2, (h + size[1]) // 2)
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    get_gradient_color(wave_time * 2), 2)

    if frame_count % 30 == 0:
        fps = int(1.0 / (time.time() - now + 0.001))
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.namedWindow("CLONE TRACKER", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CLONE TRACKER", 800, 600)

    cv2.imshow("CLONE TRACKER", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
