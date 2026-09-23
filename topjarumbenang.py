"""Pelacak lubang jarum kiri + ujung bebas benang coklat di sisi kanan.

Benang coklat masuk dari tepi kanan. Ujung yang dilacak adalah ujung bebasnya
(ujung yang tidak terpotong batas gambar), dengan posisi subpiksel.

Saat ujung diam, koordinat diratakan supaya tidak bergetar. Saat ujung
berpindah, marker langsung mengikuti pengukuran baru.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ===== Kalibrasi lubang jarum kiri pada frame pertama, resolusi 640x480 =====
# Bukaan putih aktual pada ujung silinder/jarum hitam kiri.
NEEDLE_REFERENCE_ELLIPSE = ((176.5, 169.5), (13.0, 34.0), 0.0)
NEEDLE_TEMPLATE_BOX = (145, 132, 65, 78)
NEEDLE_SEARCH_BOX = (0.00, 0.10, 0.40, 0.80)
NEEDLE_MIN_MATCH = 0.67
NEEDLE_SCALES = tuple(np.arange(0.75, 1.26, 0.05))

# ===== Segmentasi benang coklat di sisi kanan =====
# Hue rendah + saturasi tinggi: benang tembaga/coklat, bukan latar hampir putih.
THREAD_HSV_LOW = (0, 60, 18)
THREAD_HSV_HIGH = (22, 255, 230)
THREAD_MIN_AREA = 250.0
THREAD_MIN_WIDTH = 30
THREAD_MIN_RIGHT_X = 0.50
# Ujung = kolom pertama yang ketebalannya mencapai rasio ini terhadap badan benang.
THREAD_TIP_THICKNESS_RATIO = 0.30
# Di bawah radius ini (px/frame) ujung dianggap diam dan diratakan.
THREAD_STILL_RADIUS = 3.2
THREAD_STILL_ALPHA = 0.55
THREAD_QUIET_RADIUS = 1.6
THREAD_QUIET_ALPHA = 0.32


def transformed_ellipse(ellipse, matrix):
    (cx, cy), (a, b), angle = ellipse
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    r = np.deg2rad(angle)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    points = np.column_stack((a / 2 * np.cos(t), b / 2 * np.sin(t))) @ rot.T
    points += (cx, cy)
    points = points @ matrix[:, :2].T + matrix[:, 2]
    return cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))


class NeedleHoleTracker:
    """Template tracker untuk bukaan putih pada ujung jarum hitam sebelah kiri."""

    def __init__(self, first_frame):
        h, w = first_frame.shape[:2]
        sx, sy = w / 640.0, h / 480.0

        x, y, tw, th = NEEDLE_TEMPLATE_BOX
        self.box = tuple(
            int(round(v * s))
            for v, s in zip((x, y, tw, th), (sx, sy, sx, sy))
        )
        x, y, tw, th = self.box

        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y : y + th, x : x + tw].copy()
        if self.template.size == 0:
            raise ValueError("Template jarum kosong. Cek kalibrasi NEEDLE_TEMPLATE_BOX.")

        self.reference_ellipse = transformed_ellipse(
            NEEDLE_REFERENCE_ELLIPSE,
            np.array([[sx, 0, 0], [0, sy, 0]], np.float32),
        )

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        x0, y0, x1, y1 = [
            int(v * s) for v, s in zip(NEEDLE_SEARCH_BOX, (w, h, w, h))
        ]
        roi = gray[y0:y1, x0:x1]
        tx, ty, tw, th = self.box

        best = None
        for scale in NEEDLE_SCALES:
            ww = max(5, int(round(tw * float(scale))))
            hh = max(5, int(round(th * float(scale))))
            if hh > roi.shape[0] or ww > roi.shape[1]:
                continue

            template = cv2.resize(self.template, (ww, hh), interpolation=cv2.INTER_LINEAR)
            scores = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            if best is None or score > best[0]:
                best = (float(score), location, ww, hh)

        if best is None or best[0] < NEEDLE_MIN_MATCH:
            return None

        score, (dx, dy), ww, hh = best
        px, py = x0 + dx, y0 + dy

        # Map koordinat referensi frame pertama ke hasil template match saat ini.
        affine = np.array(
            [
                [ww / tw, 0, px - tx * ww / tw],
                [0, hh / th, py - ty * hh / th],
            ],
            np.float32,
        )
        ellipse = transformed_ellipse(self.reference_ellipse, affine)
        return {
            "ellipse": ellipse,
            "center": ellipse[0],
            "score": score,
            "template_box": (px, py, ww, hh),
        }


class BrownThreadTipTracker:
    """Ujung bebas benang coklat yang masuk dari sisi kanan.

    Titik ukur adalah pusat penampang di muka ujung, subpiksel, pada mask
    sebelum penutupan morfologi supaya ujung tidak bergeser akibat kernel.
    """

    def __init__(self):
        self.prev_raw = None
        self.smooth = None
        self.smooth_thickness = None
        self.smooth_angle = 0.0
        self.missed = 0

    @staticmethod
    def _masks(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        opened = cv2.inRange(
            hsv,
            np.array(THREAD_HSV_LOW, np.uint8),
            np.array(THREAD_HSV_HIGH, np.uint8),
        )
        opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        )
        return opened, closed

    def _column_profile(self, binary):
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return None
        x0 = int(xs.min())
        x1 = int(xs.max())
        length = x1 - x0 + 1
        if length < 24:
            return None

        med = np.full(length, np.nan, np.float64)
        thick = np.zeros(length, np.float64)
        for x in range(x0, x1 + 1):
            column = ys[xs == x]
            if len(column) == 0:
                continue
            med[x - x0] = float(np.median(column))
            thick[x - x0] = float(len(column))

        good = np.isfinite(med)
        if int(good.sum()) < 12:
            return None
        index = np.arange(length)
        filled = np.interp(index, index[good], med[good])
        smooth = np.empty_like(filled)
        for i in range(length):
            smooth[i] = np.median(filled[max(0, i - 4) : min(length, i + 5)])
        thickness = np.convolve(thick, np.ones(5) / 5.0, mode="same")
        thickness[0] = thick[0]
        thickness[1] = thick[1]
        thickness[-1] = thick[-1]
        thickness[-2] = thick[-2]
        return x0, smooth, thickness, thick

    def _measure_tip(self, opened, component):
        profile = self._column_profile(cv2.bitwise_and(opened, component))
        if profile is None:
            return None
        x0, centerline, thickness, raw_thick = profile
        length = len(centerline)
        body = thickness[int(0.35 * length) : int(0.75 * length)]
        body = body[body > 1.0]
        if len(body) < 4:
            return None
        body_thickness = float(np.median(body))
        if body_thickness < 4.0:
            return None

        threshold = max(3.0, THREAD_TIP_THICKNESS_RATIO * body_thickness)
        hit = None
        for i in range(0, length - 5):
            if (
                thickness[i] >= threshold
                and thickness[i + 1] >= threshold
                and thickness[i + 2] >= threshold
                and thickness[i + 3] >= threshold
            ):
                hit = i
                break
        if hit is None:
            return None

        if hit > 0 and thickness[hit] > thickness[hit - 1]:
            frac = (threshold - thickness[hit - 1]) / max(thickness[hit] - thickness[hit - 1], 1e-3)
            tip_index = (hit - 1) + float(np.clip(frac, 0.0, 1.0))
        else:
            tip_index = float(hit)

        near = float(np.interp(tip_index + 5.0, np.arange(length), centerline))
        shaft_x = tip_index + np.arange(12.0, 42.0)
        shaft_x = shaft_x[(shaft_x >= 1.0) & (shaft_x <= length - 2.0)]
        if len(shaft_x) < 8:
            return None
        shaft_y = np.interp(shaft_x, np.arange(length), centerline)
        slope = float(np.polyfit(shaft_x, shaft_y, 1)[0])
        if abs(slope) > 1.0:
            slope = 0.0
            tip_y = near
        else:
            tip_y = near - 5.0 * slope
        tip_x = float(x0) + tip_index

        height, width = opened.shape
        if not (2.0 <= tip_x < width - 2.0 and 2.0 <= tip_y < height - 2.0):
            return None

        loc = int(np.clip(round(tip_index + 12.0), 0, length - 1))
        local = float(np.median(raw_thick[max(0, loc - 2) : loc + 3]))
        local = float(np.clip(local if local > 1.0 else body_thickness, 8.0, 70.0))
        angle = float(np.degrees(np.arctan(slope)))
        along = float(np.clip(local * 0.34, 8.0, 22.0))
        ellipse = ((tip_x, tip_y), (along, local), angle)
        return {
            "tip": (tip_x, tip_y),
            "tip_ellipse": ellipse,
            "local_thickness": local,
            "angle": angle,
        }

    def _candidates(self, opened, closed):
        height, width = closed.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
        found = []
        for label in range(1, count):
            x, y, bw, bh, area = [int(v) for v in stats[label]]
            if area < THREAD_MIN_AREA or bw < THREAD_MIN_WIDTH:
                continue
            if x + bw < int(THREAD_MIN_RIGHT_X * width):
                continue
            if y > int(0.92 * height):
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            measured = self._measure_tip(opened, component)
            if measured is None:
                continue
            measured["area"] = float(area)
            measured["bbox"] = (x, y, bw, bh)
            measured["score"] = float(area) + 2.0 * bw
            found.append(measured)
        return found

    def _choose(self, found):
        if not found:
            return None
        best = max(found, key=lambda item: item["score"])
        if self.prev_raw is None or self.missed > 6:
            return best
        px, py = self.prev_raw
        nearby = []
        for item in found:
            if item["area"] < 0.35 * best["area"]:
                continue
            x, y, bw, bh = item["bbox"]
            if (x - 40) <= px <= (x + bw + 40) and (y - 50) <= py <= (y + bh + 50):
                tip_x, tip_y = item["tip"]
                item = dict(item)
                item["score"] -= 6.0 * float(np.hypot(tip_x - px, tip_y - py))
                nearby.append(item)
        if nearby:
            return max(nearby, key=lambda item: item["score"])
        return best

    def _stabilize(self, raw_tip, thickness, angle):
        raw = np.array(raw_tip, np.float64)
        if self.smooth is None or self.missed > 4:
            self.smooth = raw.copy()
            self.smooth_thickness = float(thickness)
            self.smooth_angle = float(angle)
        else:
            step = float(np.hypot(*(raw - self.prev_raw))) if self.prev_raw is not None else 99.0
            if step <= THREAD_QUIET_RADIUS:
                alpha = THREAD_QUIET_ALPHA
            elif step <= THREAD_STILL_RADIUS:
                alpha = THREAD_STILL_ALPHA
            else:
                alpha = 1.0
            self.smooth = self.smooth + alpha * (raw - self.smooth)
            self.smooth_thickness = self.smooth_thickness + alpha * (float(thickness) - self.smooth_thickness)
            delta_angle = (float(angle) - self.smooth_angle + 180.0) % 360.0 - 180.0
            self.smooth_angle += alpha * delta_angle
        self.prev_raw = raw
        self.missed = 0
        return (
            float(self.smooth[0]),
            float(self.smooth[1]),
            float(self.smooth_thickness),
            float(self.smooth_angle),
        )

    def detect(self, frame):
        opened, closed = self._masks(frame)
        chosen = self._choose(self._candidates(opened, closed))
        if chosen is None:
            self.missed += 1
            if self.missed > 6:
                self.prev_raw = None
                self.smooth = None
                self.smooth_thickness = None
            return None

        tip_x, tip_y, local, angle = self._stabilize(
            chosen["tip"], chosen["local_thickness"], chosen["angle"]
        )
        along = float(np.clip(local * 0.34, 8.0, 22.0))
        ellipse = ((tip_x, tip_y), (along, local), angle)
        return {
            "score": chosen["score"],
            "area": chosen["area"],
            "bbox": chosen["bbox"],
            "tip": (tip_x, tip_y),
            "tip_ellipse": ellipse,
            "local_thickness": local,
            "angle": angle,
        }


def draw(frame, needle, thread):
    out = frame.copy()

    if needle is not None:
        cv2.ellipse(out, needle["ellipse"], (0, 255, 0), 2, cv2.LINE_AA)
        center = tuple(int(round(v)) for v in needle["center"])
        cv2.circle(out, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
        txt = f"JARUM {center[0]},{center[1]}"
    else:
        txt = "JARUM: tidak terdeteksi"
    cv2.putText(out, txt, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    if thread is not None:
        tip_x, tip_y = thread["tip"]
        tip = (int(round(tip_x)), int(round(tip_y)))
        cv2.ellipse(out, thread["tip_ellipse"], (0, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(out, tip, (0, 0, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
        cv2.circle(out, tip, 2, (0, 0, 255), -1, cv2.LINE_AA)
        txt = f"UJUNG BENANG {tip_x:.1f},{tip_y:.1f}"
    else:
        txt = "UJUNG BENANG: tidak terdeteksi"
    cv2.putText(out, txt, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    return out


def main(inp, outp):
    if Path(inp).resolve() == Path(outp).resolve():
        raise ValueError("Gunakan nama output berbeda dari input.")

    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "frame=best_effort_timestamp_time,pkt_duration_time",
                "-of", "json", inp,
            ]
        )
    )
    timestamp_frames = [f for f in info.get("frames", []) if "best_effort_timestamp_time" in f]
    timestamps = [float(f["best_effort_timestamp_time"]) for f in timestamp_frames]

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise RuntimeError(f"Video tidak dapat dibuka: {inp}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Frame pertama tidak dapat dibaca.")

    needle_tracker = NeedleHoleTracker(first_frame)
    thread_tracker = BrownThreadTipTracker()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    records = []
    try:
        with tempfile.TemporaryDirectory(prefix="needle_thread_v3_") as directory:
            folder = Path(directory)
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                needle = needle_tracker.detect(frame)
                thread = thread_tracker.detect(frame)
                rendered = draw(frame, needle, thread)

                filename = folder / f"{i:05d}.png"
                if not cv2.imwrite(str(filename), rendered):
                    raise RuntimeError("Gagal menulis frame.")

                records.append(
                    {
                        "frame": i,
                        "needle_detected": needle is not None,
                        "needle_center": list(map(float, needle["center"])) if needle else None,
                        "needle_score": needle["score"] if needle else None,
                        "needle_ellipse": (
                            [list(map(float, needle["ellipse"][0])),
                             list(map(float, needle["ellipse"][1])),
                             float(needle["ellipse"][2])]
                            if needle else None
                        ),
                        "thread_tip_detected": thread is not None,
                        "thread_tip": list(map(float, thread["tip"])) if thread else None,
                        "thread_tip_ellipse": (
                            [list(map(float, thread["tip_ellipse"][0])),
                             list(map(float, thread["tip_ellipse"][1])),
                             float(thread["tip_ellipse"][2])]
                            if thread else None
                        ),
                        "thread_local_thickness": thread["local_thickness"] if thread else None,
                        "thread_component_bbox": list(map(int, thread["bbox"])) if thread else None,
                    }
                )
                i += 1

            if not records:
                raise RuntimeError("Tidak ada frame yang terbaca.")
            if len(records) != len(timestamps):
                raise RuntimeError(
                    f"Frame terbaca {len(records)} tetapi timestamp {len(timestamps)}."
                )

            durations = np.diff(timestamps).tolist()
            fallback = float(np.median(durations)) if durations else 1 / 30
            last = float(timestamp_frames[-1].get("pkt_duration_time", fallback) or fallback)

            lines = []
            for j, duration in enumerate(durations + [last]):
                lines.extend([
                    f"file '{j:05d}.png'",
                    "option framerate 1000",
                    f"duration {max(float(duration), 1e-6):.9f}",
                ])
            lines.extend([f"file '{len(records)-1:05d}.png'", "option framerate 1000"])
            manifest = folder / "frames.txt"
            manifest.write_text("\n".join(lines) + "\n")

            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(manifest), "-i", inp,
                    "-map", "0:v:0", "-map", "1:a?",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-fps_mode", "vfr", "-c:a", "aac", "-movflags", "+faststart", outp,
                ],
                check=True,
            )
    finally:
        cap.release()

    json_path = Path(outp).with_suffix(".json")
    json_path.write_text(json.dumps(records, indent=2))
    print(json.dumps({
        "frames": len(records),
        "needle_detected": sum(r["needle_detected"] for r in records),
        "thread_tip_detected": sum(r["thread_tip_detected"] for r in records),
        "output": outp,
        "json": str(json_path),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    main(args.input, args.output)
