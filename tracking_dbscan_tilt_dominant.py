# =============================
# Adjustable Parameters
# =============================
DBSCAN_EPS = 20           # DBSCAN clustering distance threshold
DBSCAN_MIN_SAMPLES = 30   # Minimum points per cluster
POSITION_WEIGHT = 0.2     # Weight for Kalman velocity in prediction
TILT_WEIGHT = 60          # Pixel influence of tilt direction
RECOVERY_FRAMES = 15      # Frames to keep prediction when cluster disappears
TILT_ALPHA = 0.3          # Smoothing factor for tilt vector
WINDOW_MS = 2             # Event window length in milliseconds
TARGET_FPS = 30           # Display frame rate

import cv2
import numpy as np
import time
from filterpy.kalman import KalmanFilter
from sklearn.cluster import DBSCAN
from evio.source.dat_file import DatFileSource

# -----------------------------
# Event Decoding
# -----------------------------
def decode_window(event_words, time_order, start, stop):
    event_indexes = time_order[start:stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities = ((words >> 28) & 0xF) > 0
    return x_coords, y_coords, polarities

# -----------------------------
# Frame Generation
# -----------------------------
def events_to_frame(window, width=1280, height=720):
    x_coords, y_coords, polarities = window
    frame = np.zeros((height, width), dtype=np.uint8)
    frame[y_coords[polarities], x_coords[polarities]] = 255
    frame[y_coords[~polarities], x_coords[~polarities]] = 128
    return frame

# -----------------------------
# Kalman Filter Initialization
# -----------------------------
def init_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])
    kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.P *= 1000
    kf.R *= 2
    kf.Q *= 0.05
    return kf

# -----------------------------
# Tilt Estimation with Smoothing
# -----------------------------
smoothed_tilt = np.array([0., 0.])

def estimate_tilt_direction(cluster_points):
    global smoothed_tilt
    if cluster_points is None or len(cluster_points) < 3:
        return smoothed_tilt
    pts = cluster_points - np.mean(cluster_points, axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]
    tilt_vector = major_axis / np.linalg.norm(major_axis)
    smoothed_tilt = (1 - TILT_ALPHA) * smoothed_tilt + TILT_ALPHA * tilt_vector
    norm = np.linalg.norm(smoothed_tilt)
    if norm == 0:
        return np.array([0., 0.])
    return smoothed_tilt / norm

def draw_arrow(frame, start, end, color=(0, 0, 255)):
    cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.3)

# -----------------------------
# Background Filtering
# -----------------------------
background_activity = np.zeros((720, 1280), dtype=np.float32)
decay_rate = 0.95
activity_threshold = 50

def filter_background_fast(x_coords, y_coords):
    global background_activity
    np.multiply(background_activity, decay_rate, out=background_activity)
    valid_mask = (x_coords >= 0) & (x_coords < 1280) & (y_coords >= 0) & (y_coords < 720)
    x_valid = x_coords[valid_mask]
    y_valid = y_coords[valid_mask]
    np.add.at(background_activity, (y_valid, x_valid), 1)
    mask = background_activity[y_valid, x_valid] < activity_threshold
    return x_valid[mask], y_valid[mask]

# -----------------------------
# DBSCAN Clustering
# -----------------------------
def cluster_events_dbscan(x_coords, y_coords):
    points = np.column_stack((x_coords, y_coords))
    if len(points) == 0:
        return []
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
    labels = clustering.labels_
    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_points = points[labels == label]
        cx, cy = np.mean(cluster_points, axis=0).astype(int)
        clusters.append((cx, cy, cluster_points))
    return clusters

# -----------------------------
# Tracker with Size Safeguard
# -----------------------------
class Tracker:
    def __init__(self):
        self.previous_info = None
        self.missing_frames = 0

    def update(self, clusters):
        if clusters:
            cluster_infos = []
            for cx, cy, points in clusters:
                size = len(points)
                cluster_infos.append(((cx, cy), size, points))
            if self.previous_info is None:
                best = max(cluster_infos, key=lambda c: c[1])
            else:
                prev_center, prev_size, _ = self.previous_info
                best = min(cluster_infos, key=lambda c: np.hypot(c[0][0]-prev_center[0], c[0][1]-prev_center[1]))
                if abs(best[1] - prev_size) / prev_size > 0.5:
                    return prev_center, None
            self.previous_info = (best[0], best[1], best[2])
            self.missing_frames = 0
            return best[0], best[2]
        else:
            self.missing_frames += 1
            if self.missing_frames <= RECOVERY_FRAMES and self.previous_info:
                return self.previous_info[0], None
            return None, None

# -----------------------------
# Main Loop
# -----------------------------
def track_and_display(dat_file, debug=True):
    width, height = 1280, 720
    src = DatFileSource(dat_file, width=width, height=height, window_length_us=WINDOW_MS * 1000)
    kf = init_kalman()
    path_points = []
    tracker = Tracker()
    prev_raw_x, prev_raw_y = None, None
    alpha = 0.7
    max_jump = 100

    if debug:
        cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video Feed", width, height)
        cv2.namedWindow("Tracking View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking View", width, height)

    frame_interval = 1.0 / TARGET_FPS
    last_time = time.time()

    for batch_range in src.ranges():
        window = decode_window(src.event_words, src.order, batch_range.start, batch_range.stop)
        video_frame = events_to_frame(window, width, height)
        video_frame_bgr = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
        tracking_frame = np.zeros((height, width, 3), dtype=np.uint8)

        x_coords, y_coords = filter_background_fast(window[0], window[1])
        clusters = cluster_events_dbscan(x_coords, y_coords)
        cluster_center, cluster_points = tracker.update(clusters)

        kf.predict()
        smooth_x, smooth_y = int(kf.x[0]), int(kf.x[1])

        if cluster_center:
            raw_cx, raw_cy = cluster_center
            if prev_raw_x is not None:
                raw_cx = int(alpha * raw_cx + (1 - alpha) * prev_raw_x)
                raw_cy = int(alpha * raw_cy + (1 - alpha) * prev_raw_y)
            if prev_raw_x is not None and np.hypot(raw_cx - prev_raw_x, raw_cy - prev_raw_y) > max_jump:
                raw_cx, raw_cy = smooth_x, smooth_y
            else:
                kf.update((raw_cx, raw_cy))
            prev_raw_x, prev_raw_y = raw_cx, raw_cy
            smooth_x, smooth_y = int(kf.x[0]), int(kf.x[1])
            path_points.append((smooth_x, smooth_y))

            tilt_vector = estimate_tilt_direction(cluster_points)

            # Combined prediction: tilt dominates
            kalman_velocity = np.array([kf.x[2], kf.x[3]])
            combined_vector = kalman_velocity * POSITION_WEIGHT + tilt_vector * TILT_WEIGHT
            next_point = (int(cluster_center[0] + combined_vector[0]), int(cluster_center[1] + combined_vector[1]))

        if debug:
            if cluster_points is not None:
                for pt in cluster_points:
                    cv2.circle(tracking_frame, tuple(pt), 2, (255, 255, 0), -1)
            for i in range(1, len(path_points)):
                cv2.line(tracking_frame, path_points[i-1], path_points[i], (0, 255, 0), 2)
            cv2.circle(tracking_frame, (smooth_x, smooth_y), 8, (255, 0, 0), -1)
            if prev_raw_x and prev_raw_y:
                cv2.circle(tracking_frame, (prev_raw_x, prev_raw_y), 5, (0, 255, 255), -1)

            if cluster_center and cluster_points is not None:
                draw_arrow(tracking_frame, cluster_center, next_point, color=(0, 0, 255))
                cv2.circle(tracking_frame, next_point, 6, (0, 255, 255), -1)
                cv2.putText(tracking_frame, f"Tilt: ({tilt_vector[0]:.2f}, {tilt_vector[1]:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Video Feed", video_frame_bgr)
            cv2.imshow("Tracking View", tracking_frame)
            elapsed = time.time() - last_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_time = time.time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if debug:
        cv2.destroyAllWindows()

# Example usage:
# track_and_display("drone_moving.dat", debug=True)
