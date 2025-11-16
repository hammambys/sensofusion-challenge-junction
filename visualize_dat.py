import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

from evio.core.recording import open_dat
from evio.source.dat_file import DatFileSource

##############################################
# PARAMETERS
##############################################

FILE = "fan_const_rpm.dat"  # <-- PUT YOUR DATASET FILE HERE
WINDOW_US = 5000  # 5 ms windows
PLAYBACK_SPEED = 1.0  # 1.0 = realtime, >1 faster, <1 slower
PLOT_INTERVAL_S = 0.05  # update plot at most every 50 ms
DECAY = 0.8  # per-window decay for motion trails (0..1)
NUM_BLADES = 3  # <-- ESTIMATE: typically 2-5 blades for a fan
MIN_PEAK_DIST = 3  # windows apart for peak detection
FAN_BOX_SIZE = 240  # size of detection box (square)
MIN_ACTIVITY = 10.0  # minimum total activity to trust center-of-mass


##############################################
# EVENT DECODING HELPERS
##############################################


def decode_events(words_u32, timestamps, order, start, stop):
    """Return arrays x, y, pol, ts for events in [start:stop]."""
    idx = order[start:stop]

    w = words_u32[idx]
    ts = timestamps[idx]

    # DAT CD8 layout: [31:28]=polarity (4 bits), [27:14]=y (14 bits), [13:0]=x (14 bits)
    x = (w & 0x3FFF).astype(np.int32)
    y = ((w >> 14) & 0x3FFF).astype(np.int32)
    pol = (((w >> 28) & 0xF) > 0).astype(np.int8)

    return x, y, pol, ts


##############################################
# RPM ESTIMATION
##############################################


def estimate_rpm(peak_times_us, num_blades=1):
    """peak_times_us: list of timestamps in microseconds"""
    if len(peak_times_us) < 2:
        return 0.0
    diffs_us = np.diff(peak_times_us)
    period_us = np.mean(diffs_us) * num_blades
    period_s = period_us / 1e6
    freq = 1.0 / period_s
    return freq * 60.0


def estimate_rpm_fft(counts, window_us, num_blades=1):
    """Estimate RPM by finding dominant frequency in the counts time-series (robust to speed variation)."""
    if len(counts) < 16:
        return 0.0
    arr = np.asarray(counts, dtype=float)
    arr = arr - arr.mean()
    n = arr.size
    # sampling rate in Hz (windows per second)
    fs = 1.0 / (window_us / 1e6)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(arr))
    # ignore DC and very low frequencies
    spec[0] = 0.0
    if len(spec) > 1:
        spec[1] = 0.0
    idx = np.argmax(spec)
    f_spike = freqs[idx] if idx < len(freqs) else 0.0
    if f_spike <= 0:
        return 0.0
    # convert spike frequency to rotations per minute
    rpm = (f_spike * 60.0) / float(num_blades)
    return rpm


##############################################
# MAIN
##############################################


def main():
    # Load full recording
    rec = open_dat(FILE, width=1280, height=720)

    # Build time windows
    src = DatFileSource(
        FILE, window_length_us=WINDOW_US, width=rec.width, height=rec.height
    )

    words = src.event_words
    order = src.order
    timestamps = rec.timestamps

    # RPM tracking
    counts = deque(maxlen=200)
    peaks = deque(maxlen=200)
    last_peak = -999
    rpm_history = deque(maxlen=10)  # moving average of RPM values
    stable_rpm = 0.0  # last valid RPM reading

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))

    # playback pacing: record real start and recording start timestamp
    try:
        rec_start_ts_us = int(src._ranges[0].start_ts_us)
    except Exception:
        rec_start_ts_us = int(rec.timestamps[0])
    playback_start_real = time.time()
    last_plot_time = 0.0

    print("Starting .dat file visualization with RPM detection...")

    # prepare full-frame RGB buffer with polarity
    h, w = int(rec.height), int(rec.width)
    pos_buf = np.zeros((h, w), dtype=np.float32)
    neg_buf = np.zeros((h, w), dtype=np.float32)
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # fan position smoothing (initialize after w, h are defined)
    fan_cx = w // 2
    fan_cy = h // 2

    im = ax.imshow(rgb, origin="lower")
    ax.set_title("Event Camera - Full Frame")
    ax.axis("off")

    # rectangle patch for fan detection box
    rect_patch = plt.Rectangle(
        (0, 0),
        FAN_BOX_SIZE,
        FAN_BOX_SIZE,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rect_patch)

    # RPM text overlay
    rpm_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, color="cyan", fontsize=14, va="top"
    )

    for idx, batch in enumerate(src.ranges()):

        # decode events in this window
        x, y, pol, ts = decode_events(words, timestamps, order, batch.start, batch.stop)

        # apply decay to create motion trails
        pos_buf *= DECAY
        neg_buf *= DECAY

        if x.size > 0:
            # clamp to image bounds
            x = np.clip(x, 0, w - 1).astype(np.intp)
            y = np.clip(y, 0, h - 1).astype(np.intp)
            pol_mask = pol.astype(np.bool_)

            # ON events (red)
            if pol_mask.any():
                xs_on = x[pol_mask]
                ys_on = y[pol_mask]
                np.add.at(pos_buf, (ys_on, xs_on), 1.0)

            # OFF events (blue)
            if (~pol_mask).any():
                xs_off = x[~pol_mask]
                ys_off = y[~pol_mask]
                np.add.at(neg_buf, (ys_off, xs_off), 1.0)

            # event count for peak detection (from full frame)
            c = x.size
            counts.append(c)

            # detect peak (simple local max)
            if len(counts) > 5:
                if counts[-2] == max(list(counts)[-5:]):
                    if idx - last_peak >= MIN_PEAK_DIST:
                        center_ts = int((batch.start_ts_us + batch.end_ts_us) // 2)
                        peaks.append(center_ts)
                        last_peak = idx
        else:
            counts.append(0)

        # Pace playback so overall runtime matches recording duration
        target_elapsed_s = (
            (batch.end_ts_us - rec_start_ts_us) / 1e6 / float(PLAYBACK_SPEED)
        )
        target_real = playback_start_real + target_elapsed_s
        now = time.time()
        sleep_time = target_real - now
        if sleep_time > 0:
            time.sleep(sleep_time)

        # live plot (throttle updates to avoid slow redraws)
        if time.time() - last_plot_time >= PLOT_INTERVAL_S:
            # map polarity buffers to RGB (red=ON, blue=OFF)
            s = max(1e-6, max(pos_buf.max(), neg_buf.max()) / 10.0)
            r = np.clip(pos_buf / s, 0.0, 1.0)
            b = np.clip(neg_buf / s, 0.0, 1.0)
            g = 0.25 * (r + b)
            rgb[..., 0] = r
            rgb[..., 1] = g
            rgb[..., 2] = b
            im.set_data(rgb)

            # compute RPM and find fan location (center of mass in activity)
            raw_rpm = estimate_rpm_fft(list(counts), WINDOW_US, num_blades=NUM_BLADES)
            # filter RPM to reasonable range (allow ±20% variation from ~1100 RPM)
            if 800 <= raw_rpm <= 1400:
                rpm_history.append(raw_rpm)
                stable_rpm = np.mean(list(rpm_history))
            activity = pos_buf + neg_buf
            total_activity = activity.sum()

            # reset RPM if fan has stopped (no significant activity)
            if total_activity < MIN_ACTIVITY:
                stable_rpm = 0.0
                rpm_history.clear()

            if total_activity > MIN_ACTIVITY:
                # compute center of mass
                y_coords, x_coords = np.nonzero(activity)
                if len(x_coords) > 0:
                    weights = activity[y_coords, x_coords]
                    fan_cx = np.average(x_coords, weights=weights)
                    fan_cy = np.average(y_coords, weights=weights)

            # center box on smoothed fan position
            cx = max(FAN_BOX_SIZE // 2, min(int(fan_cx), w - FAN_BOX_SIZE // 2))
            cy = max(FAN_BOX_SIZE // 2, min(int(fan_cy), h - FAN_BOX_SIZE // 2))
            box_x = cx - FAN_BOX_SIZE // 2
            box_y = cy - FAN_BOX_SIZE // 2
            rect_patch.set_xy((box_x, box_y))

            rpm_text.set_text(f"RPM: {stable_rpm:.1f}")

            fig.canvas.draw_idle()
            plt.pause(0.001)
            last_plot_time = time.time()

        if idx % 100 == 0:
            print(f"Window {idx:04d} | Events={c:6d} | RPM={stable_rpm:6.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
