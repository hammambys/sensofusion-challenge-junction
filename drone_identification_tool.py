
import cv2
import numpy as np
import time
from collections import deque, namedtuple
from evio.source.dat_file import DatFileSource
from filterpy.kalman import KalmanFilter

Detected = namedtuple("Detected", ["center", "bbox", "density", "num_events", "confidence"])

# ---------------- Improved Drone Detector ----------------
class ImprovedDroneDetector:
    def __init__(self, width=1280, height=720, grid_size=64, buffer_s=0.8,
                 min_events=80, min_density=0.08, min_persistence=4,
                 min_speed=10, min_bbox_area=3000, max_bbox_area=150000):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.buffer_s = buffer_s
        self.min_events = min_events
        self.min_density = min_density
        self.min_persistence = min_persistence
        self.min_speed = min_speed
        self.min_bbox_area = min_bbox_area
        self.max_bbox_area = max_bbox_area

        self.nx = (width + grid_size - 1) // grid_size
        self.ny = (height + grid_size - 1) // grid_size
        self.grid = [[deque() for _ in range(self.ny)] for _ in range(self.nx)]
        self.cluster_history = {}

    def append_events(self, x_coords, y_coords, polarities, batch_time, batch_window_s):
        times = np.linspace(batch_time - batch_window_s, batch_time, len(x_coords))
        for x, y, t in zip(x_coords, y_coords, times):
            gx, gy = int(x) // self.grid_size, int(y) // self.grid_size
            if 0 <= gx < self.nx and 0 <= gy < self.ny:
                self.grid[gx][gy].append((x, y, t))
        self._trim_old_events(batch_time)

    def _trim_old_events(self, now):
        cutoff = now - self.buffer_s
        for gx in range(self.nx):
            for gy in range(self.ny):
                dq = self.grid[gx][gy]
                while dq and dq[0][2] < cutoff:
                    dq.popleft()

    def detect(self, now):
        detections = []
        visited = np.zeros((self.nx, self.ny), dtype=bool)
        clusters = []

        for gx in range(self.nx):
            for gy in range(self.ny):
                if visited[gx, gy] or len(self.grid[gx][gy]) < 10:
                    continue
                cluster_events = []
                queue = [(gx, gy)]
                while queue:
                    cx, cy = queue.pop()
                    if visited[cx, cy]:
                        continue
                    visited[cx, cy] = True
                    cluster_events.extend(self.grid[cx][cy])
                    for nx_ in range(max(0,cx-1), min(self.nx,cx+2)):
                        for ny_ in range(max(0,cy-1), min(self.ny,cy+2)):
                            if not visited[nx_, ny_] and len(self.grid[nx_][ny_]) >= 10:
                                queue.append((nx_, ny_))

                if len(cluster_events) < self.min_events:
                    continue

                pts = np.array(cluster_events)
                min_x, min_y = np.min(pts[:, :2], axis=0).astype(int)
                max_x, max_y = np.max(pts[:, :2], axis=0).astype(int)
                cx, cy = int(np.mean(pts[:,0])), int(np.mean(pts[:,1]))
                bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                density = len(cluster_events) / bbox_area

                # Compute compactness and aspect ratio
                width = max_x - min_x + 1
                height = max_y - min_y + 1
                aspect_ratio = max(width, height) / max(1, min(width, height))
                compactness = len(cluster_events) / bbox_area

                # Reject clusters that are too elongated (likely tree)
                if aspect_ratio > 2.0:  # tune threshold
                    continue

                # Reject clusters with very low compactness
                if compactness < 0.05:  # tune threshold
                    continue

                # Early rejection
                if density < self.min_density or bbox_area < self.min_bbox_area or bbox_area > self.max_bbox_area:
                    continue

                clusters.append({"center": (cx, cy), "bbox": (min_x, min_y, max_x, max_y),
                                 "density": density, "num_events": len(cluster_events)})

        merged_clusters = self._merge_clusters(clusters)

        for cluster in merged_clusters:
            cid = self._cluster_id(cluster["center"])
            prev = self.cluster_history.get(cid)
            speed = 0
            if prev:
                prev_center = prev["center"]
                dt = now - prev["last_seen"]
                if dt > 0:
                    speed = np.linalg.norm(np.array(cluster["center"]) - np.array(prev_center)) / dt

            # Update history
            self.cluster_history[cid] = {"count": prev["count"] + 1 if prev else 1,
                                         "last_seen": now, "center": cluster["center"]}

            persistence = self.cluster_history[cid]["count"]
            if persistence < self.min_persistence or speed < self.min_speed:
                continue  # Ignore static or short-lived clusters

            confidence = min(1.0, 0.3 + compactness * 0.4 + persistence * 0.05)
            detections.append(Detected(center=cluster["center"], bbox=cluster["bbox"],
                                       density=cluster["density"], num_events=cluster["num_events"],
                                       confidence=confidence))

        return detections

    def _merge_clusters(self, clusters, iou_thresh=0.3):
        merged = []
        for c in clusters:
            merged_flag = False
            for m in merged:
                if self._iou(c["bbox"], m["bbox"]) > iou_thresh:
                    min_x = min(c["bbox"][0], m["bbox"][0])
                    min_y = min(c["bbox"][1], m["bbox"][1])
                    max_x = max(c["bbox"][2], m["bbox"][2])
                    max_y = max(c["bbox"][3], m["bbox"][3])
                    m["bbox"] = (min_x, min_y, max_x, max_y)
                    m["center"] = ((m["center"][0] + c["center"][0]) // 2,
                                   (m["center"][1] + c["center"][1]) // 2)
                    m["density"] = (m["density"] + c["density"]) / 2
                    m["num_events"] += c["num_events"]
                    merged_flag = True
                    break
            if not merged_flag:
                merged.append(c)
        return merged

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def _cluster_id(self, center):
        return (center[0] // 20, center[1] // 20)

# ---------------- Kalman Tracker with Occlusion Handling ----------------
class Track:
    def __init__(self, track_id, initial_pos, confidence):
        self.id = track_id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.kf.P *= 10
        self.kf.R *= 5
        self.kf.Q *= 0.01
        self.kf.x = np.zeros((4,1))
        self.kf.x[:2] = np.array(initial_pos).reshape(2,1)
        self.confidence = confidence
        self.last_update = time.time()

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2].flatten()

    def update(self, pos, confidence):
        self.kf.update(np.array(pos).reshape(2,1))
        self.confidence = min(1.0, (self.confidence + confidence) / 2)
        self.last_update = time.time()

    def decay(self):
        self.confidence *= 0.98  # slower decay for occlusion handling

    def future_position(self, steps=10):
        pos = self.kf.x[:2].flatten()
        vel = self.kf.x[2:].flatten()
        return pos + vel * steps

class TrackManager:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update_tracks(self, detections):
        now = time.time()
        matched_ids = set()

        for det in detections:
            cx, cy = det.center  # Ensure this is inside the loop
            best_id, best_score = None, -float('inf')

            for tid, track in self.tracks.items():
                pred = track.predict()
                dist = np.linalg.norm(pred - np.array([cx, cy]))
                if dist < 50:  # within match threshold
                    # Score = inverse distance + confidence boost
                    score = (50 - dist) + det.confidence * 50  # weight confidence heavily
                    if score > best_score:
                        best_id, best_score = tid, score

            if best_id is not None:
                self.tracks[best_id].update([cx, cy], det.confidence)
                matched_ids.add(best_id)
            else:
                self.tracks[self.next_id] = Track(self.next_id, [cx, cy], det.confidence)
                self.next_id += 1

        # Handle unmatched tracks (occlusion)
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            if tid not in matched_ids:
                track.decay()  # slower decay for occlusion
            if now - track.last_update > 2.0:  # keep predicting for 2s
                del self.tracks[tid]

    
def update_tracks(self, detections):
    now = time.time()
    matched_ids = set()

    for det in detections:
        cx, cy = det.center  # Make sure this is inside the loop
        best_id, best_score = None, -float('inf')

        for tid, track in self.tracks.items():
            pred = track.predict()
            dist = np.linalg.norm(pred - np.array([cx, cy]))
            if dist < 50:  # within match threshold
                # Score = inverse distance + confidence boost
                score = (50 - dist) + det.confidence * 50  # weight confidence heavily
                if score > best_score:
                    best_id, best_score = tid, score

        if best_id is not None:
            self.tracks[best_id].update([cx, cy], det.confidence)
            matched_ids.add(best_id)
        else:
            self.tracks[self.next_id] = Track(self.next_id, [cx, cy], det.confidence)
            self.next_id += 1

    # Handle unmatched tracks (occlusion)
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            if tid not in matched_ids:
                track.decay()
            if now - track.last_update > 2.0:  # keep predicting for 2s
                del self.tracks[tid]



        for det in detections:
            cx, cy = det.center
            best_id, best_dist = None, float('inf')
            for tid, track in self.tracks.items():
                pred = track.predict()
                dist = np.linalg.norm(pred - np.array([cx, cy]))
                if dist < 50 and dist < best_dist:
                    best_id, best_dist = tid, dist
            if best_id is not None:
                self.tracks[best_id].update([cx, cy], det.confidence)
                matched_ids.add(best_id)
            else:
                self.tracks[self.next_id] = Track(self.next_id, [cx, cy], det.confidence)
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            if tid not in matched_ids:
                track.decay()
            if now - track.last_update > 2.0:  # keep predicting for 2s
                del self.tracks[tid]

# ---------------- Main Loop ----------------
def decode_window(event_words, time_order, start, stop):
    event_indexes = time_order[start:stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities = ((words >> 28) & 0xF) > 0
    return x_coords, y_coords, polarities

def events_to_frame(x_coords, y_coords, width=1280, height=720):
    frame = np.zeros((height, width), dtype=np.uint8)
    frame[y_coords, x_coords] = 255
    return frame

def track_and_visualize(dat_file, window_ms=2, target_fps=30):
    width, height = 1280, 720
    src = DatFileSource(dat_file, width=width, height=height, window_length_us=window_ms * 1000)
    detector = ImprovedDroneDetector(width, height)
    track_manager = TrackManager()
    last_time = time.time()
    frame_interval = 1.0 / target_fps

    cv2.namedWindow("Drone Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Tracking", width, height)

    for batch_range in src.ranges():
        now = time.time()
        if now - last_time < frame_interval:
            continue
        last_time = now

        x_coords, y_coords, polarities = decode_window(src.event_words, src.order, batch_range.start, batch_range.stop)
        if len(x_coords) == 0:
            continue

        batch_time = now
        batch_window_s = window_ms / 1000.0
        detector.append_events(x_coords, y_coords, polarities, batch_time, batch_window_s)

        detections = detector.detect(now)
        track_manager.update_tracks(detections)

        frame = events_to_frame(x_coords, y_coords, width, height)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for det in detections:
            min_x, min_y, max_x, max_y = det.bbox
            cv2.rectangle(frame_bgr, (min_x, min_y), (max_x, max_y), (0,0,255), 2)

        for tid, track in track_manager.tracks.items():
            pos = track.kf.x[:2].flatten().astype(int)
            future_pos = track.future_position(steps=10).astype(int)
            conf_pct = int(track.confidence * 100)
            color = (0,255,0) if conf_pct > 70 else (0,255,255) if conf_pct > 40 else (0,0,255)
            cv2.circle(frame_bgr, tuple(pos), 6, color, -1)
            cv2.line(frame_bgr, tuple(pos), tuple(future_pos), (255,0,0), 2)
            cv2.putText(frame_bgr, f"ID:{tid} Conf:{conf_pct}%",
                        (pos[0]+10, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Drone Tracking", frame_bgr)
        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("datfile", help="Path to .dat event file")
    parser.add_argument("--window_ms", type=int, default=2, help="Event window in ms")
    args = parser.parse_args()
    track_and_visualize(args.datfile, window_ms=args.window_ms)
