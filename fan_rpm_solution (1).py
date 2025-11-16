import argparse
import time
from collections import deque

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource

# ===================== GLOBAL CONFIG =====================

# Dataset resolution
WIDTH = 1280
HEIGHT = 720

# Static ROI for the FAN (tuned around the fan)
FAN_ROI_X_MIN = 520
FAN_ROI_X_MAX = 760
FAN_ROI_Y_MIN = 200
FAN_ROI_Y_MAX = 520

# How many samples we keep for the FFT (more -> smoother but slower to react)
HISTORY_LEN = 256

# Minimum events in ROI to consider that something is actually moving
MIN_EVENTS_ROI = 30

# =========================================================


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode events for a given index window into x, y, polarity."""
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)

    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = WIDTH,
    height: int = HEIGHT,
    *,
    base_color: tuple[int, int, int] = (127, 127, 127),  # gray
    on_color: tuple[int, int, int] = (255, 255, 255),    # white
    off_color: tuple[int, int, int] = (0, 0, 0),         # black
) -> np.ndarray:
    """Render ON/OFF events into a simple RGB frame."""
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color
    return frame


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Overlay timing info: wall time, recording time, and playback speed."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  drops/ms={pacer.instantaneous_drop_rate:.2f}"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    else:
        first_row_str = (
            f"(target) speed={pacer.speed:.2f}x  force_speed = False, no drops"
        )

    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def find_drone_roi_grid(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Dynamic ROI for the drone, based on a coarse event grid.

    1. Build a heatmap from all events in the frame.
    2. Divide the frame into cells (e.g. 64x64 pixels).
    3. Find the cell with the highest number of events.
    4. Return an ROI around that cell, padded a bit.
    """
    if x_coords.size == 0:
        return 0, 0, WIDTH, HEIGHT

    # Build full-frame heatmap
    heat = np.zeros((HEIGHT, WIDTH), np.uint16)
    # Clip just in case
    x_clipped = np.clip(x_coords, 0, WIDTH - 1)
    y_clipped = np.clip(y_coords, 0, HEIGHT - 1)
    heat[y_clipped, x_clipped] += 1

    # Coarse grid size
    CELL_W = 64
    CELL_H = 64

    n_cells_x = (WIDTH + CELL_W - 1) // CELL_W
    n_cells_y = (HEIGHT + CELL_H - 1) // CELL_H

    # Sum events per cell
    grid = np.zeros((n_cells_y, n_cells_x), dtype=np.int32)
    for iy in range(n_cells_y):
        y0 = iy * CELL_H
        y1 = min(HEIGHT, y0 + CELL_H)
        for ix in range(n_cells_x):
            x0 = ix * CELL_W
            x1 = min(WIDTH, x0 + CELL_W)
            grid[iy, ix] = int(heat[y0:y1, x0:x1].sum())

    # Find cell with max events
    max_idx = np.argmax(grid)
    cy, cx = np.unravel_index(max_idx, grid.shape)

    x0 = cx * CELL_W
    y0 = cy * CELL_H
    x1 = min(WIDTH, x0 + CELL_W)
    y1 = min(HEIGHT, y0 + CELL_H)

    # Pad the ROI a bit for safety
    PAD = 20
    roi_x_min = max(0, x0 - PAD)
    roi_y_min = max(0, y0 - PAD)
    roi_x_max = min(WIDTH, x1 + PAD)
    roi_y_max = min(HEIGHT, y1 + PAD)

    return roi_x_min, roi_y_min, roi_x_max, roi_y_max


class FftRotationEstimator:
    """
    Estimate the dominant periodic frequency (Hz) from a history of
    ROI event counts using an FFT, limited to a given frequency band.
    """

    def __init__(
        self,
        window_dt_s: float,
        history_len: int = HISTORY_LEN,
        f_min: float = 5.0,
        f_max: float = 200.0,
    ):
        """
        window_dt_s: time step between samples in seconds
        history_len: number of samples kept in the history.
        f_min, f_max: frequency band of interest (Hz).
        """
        self.window_dt_s = float(window_dt_s)
        self.history = deque(maxlen=history_len)
        self.f_min = float(f_min)
        self.f_max = float(f_max)

        self.prev_freq_hz: float | None = None
        self.prev_rpm: float | None = None

    def update(self, roi_count: int) -> tuple[float | None, float | None]:
        """
        Add one new ROI event count.

        Returns (freq_hz, rpm) where freq_hz is the dominant
        pattern frequency in Hz and rpm is 60 * freq_hz.
        Both are None if we don't have enough history yet.
        """
        self.history.append(float(roi_count))
        n = len(self.history)
        if n < 32:
            return None, None

        x = np.asarray(self.history, dtype=float)

        # Remove mean (DC)
        x = x - x.mean()

        # Apply Hann window to reduce spectral leakage
        x *= np.hanning(n)

        # Frequency axis for rfft
        freqs = np.fft.rfftfreq(n, d=self.window_dt_s)
        spec = np.abs(np.fft.rfft(x))

        if freqs.size < 3:
            return None, None

        # Ignore DC bin
        spec[0] = 0.0

        # Restrict to plausible frequency range
        mask_band = (freqs >= self.f_min) & (freqs <= self.f_max)
        if not np.any(mask_band):
            return None, None

        freqs_band = freqs[mask_band]
        spec_band = spec[mask_band]

        # Local maxima in the band
        if spec_band.size < 3:
            idx = int(np.argmax(spec_band))
            freq_hz = float(freqs_band[idx])
        else:
            interior = spec_band[1:-1]
            left = spec_band[:-2]
            right = spec_band[2:]
            peak_mask = (interior > left) & (interior >= right)
            peak_indices = np.where(peak_mask)[0] + 1

            if peak_indices.size == 0:
                idx = int(np.argmax(spec_band))
                freq_hz = float(freqs_band[idx])
            else:
                peak_amps = spec_band[peak_indices]
                global_max = peak_amps.max()
                strong = peak_indices[peak_amps >= 0.4 * global_max]

                if strong.size == 0:
                    chosen = int(peak_indices[np.argmax(peak_amps)])
                else:
                    chosen = int(strong[0])  # lowest-frequency strong peak

                freq_hz = float(freqs_band[chosen])

        if freq_hz <= 0.0:
            return None, None

        rpm = 60.0 * freq_hz  # pattern cycles per minute

        # Very simple temporal smoothing / outlier rejection
        if self.prev_rpm is not None:
            ratio = rpm / self.prev_rpm
            if ratio > 1.8 or ratio < 0.55:
                # too big jump -> keep previous
                return self.prev_freq_hz, self.prev_rpm

        self.prev_freq_hz = freq_hz
        self.prev_rpm = rpm
        return freq_hz, rpm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument(
        "--window", type=float, default=5.0, help="Window duration in ms"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed (1 is real time)"
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    parser.add_argument(
        "--scenario",
        choices=["fan", "drone"],
        default="fan",
        help="Object type: fan or drone",
    )
    args = parser.parse_args()

    window_dt_s = args.window / 1000.0

    # Frequency band depending on scenario
    if args.scenario == "fan":
        # fan ~1100 rpm => ~18 Hz, keep margin
        f_min, f_max = 10.0, 30.0
    else:
        # drone 5000–6500 rpm => ~83–108 Hz, keep wider band
        f_min, f_max = 60.0, 140.0

    src = DatFileSource(
        args.dat,
        width=WIDTH,
        height=HEIGHT,
        window_length_us=int(args.window * 1000.0),
    )

    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    freq_estimator = FftRotationEstimator(
        window_dt_s=window_dt_s,
        history_len=HISTORY_LEN,
        f_min=f_min,
        f_max=f_max,
    )

    cv2.namedWindow("Evio Player", cv2.WINDOW_NORMAL)

    # For tracking drone motion (in image plane)
    prev_cx: float | None = None
    prev_cy: float | None = None

    for batch_range in pacer.pace(src.ranges()):
        x_coords, y_coords, pixel_polarity = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )

        # ----- ROI selection -----
        if args.scenario == "fan":
            roi_x_min, roi_x_max = FAN_ROI_X_MIN, FAN_ROI_X_MAX
            roi_y_min, roi_y_max = FAN_ROI_Y_MIN, FAN_ROI_Y_MAX
        else:
            roi_x_min, roi_y_min, roi_x_max, roi_y_max = find_drone_roi_grid(
                x_coords, y_coords
            )

        # ---------- build 1-D signal: event count in ROI ----------
        mask_roi = (
            (x_coords >= roi_x_min)
            & (x_coords < roi_x_max)
            & (y_coords >= roi_y_min)
            & (y_coords < roi_y_max)
        )
        roi_count = int(np.count_nonzero(mask_roi))

        if roi_count < MIN_EVENTS_ROI:
            freq_hz, rpm = None, 0.0
        else:
            freq_hz, rpm = freq_estimator.update(roi_count)
        # ---------------------------------------------------------

        frame = get_frame((x_coords, y_coords, pixel_polarity))
        draw_hud(frame, pacer, batch_range)

        # Draw ROI rectangle
        cv2.rectangle(
            frame,
            (roi_x_min, roi_y_min),
            (roi_x_max - 1, roi_y_max - 1),
            (0, 255, 0),
            1,
        )

        # ----- Drone position + speed (only in drone scenario) -----
        if args.scenario == "drone":
            cx = roi_x_min + (roi_x_max - roi_x_min) / 2.0
            cy = roi_y_min + (roi_y_max - roi_y_min) / 2.0

            # blue dot at drone center
            cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)

            speed_pix = None
            dt_s = max(
                (batch_range.end_ts_us - batch_range.start_ts_us) / 1e6,
                1e-6,
            )
            if prev_cx is not None and prev_cy is not None:
                vx = (cx - prev_cx) / dt_s
                vy = (cy - prev_cy) / dt_s
                speed_pix = float(np.hypot(vx, vy))

            prev_cx, prev_cy = cx, cy

            if speed_pix is not None:
                text_speed = f"speed ≈ {speed_pix:7.1f} px/s"
                cv2.putText(
                    frame,
                    text_speed,
                    (8, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        # ----- Overlay frequency / RPM -----
        if rpm is not None:
            if rpm == 0.0:
                text1 = "no periodic motion"
                text2 = "RPM  ≈ 0"
            else:
                text1 = (
                    f"freq ≈ {freq_hz:6.2f} Hz"
                    if freq_hz is not None
                    else "freq ≈ ?"
                )
                text2 = f"RPM  ≈ {rpm:7.1f}"

            cv2.putText(
                frame,
                text1,
                (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text2,
                (8, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Evio Player", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
