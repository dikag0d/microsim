"""Pelacak lubang jarum kiri + ujung benang coklat.

Kalibrasi ini khusus untuk video 2026-09-15-133849.webm.
Perubahan utama dari versi sebelumnya:
1) referensi lubang jarum dikalibrasi ulang ke bukaan putih pada ujung jarum hitam di kiri;
2) marker ujung benang berupa oval yang skalanya mengikuti ketebalan lokal benang.
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

# ===== Segmentasi benang coklat =====
THREAD_ROI = (0.16, 0.10, 0.74, 0.68)
THREAD_HSV_LOW = (2, 70, 18)
THREAD_HSV_HIGH = (28, 255, 210)
THREAD_MIN_AREA = 35.0
THREAD_MIN_RIGHT_X = 0.50
THREAD_TIP_PERCENTILE = 1.0
THREAD_TIP_BAND_PX = 8.0


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
    """Deteksi ujung kiri benang coklat dan ukuran lokal penampang ujungnya."""

    def __init__(self):
        self.prev_tip = None
        self.missed = 0

    @staticmethod
    def _tip_geometry(component_mask, x0, y0):
        ys, xs = np.where(component_mask > 0)
        if len(xs) == 0:
            return None

        xs = xs.astype(np.float32) + float(x0)
        ys = ys.astype(np.float32) + float(y0)

        lead_x = float(np.percentile(xs, THREAD_TIP_PERCENTILE))
        band = xs <= (lead_x + THREAD_TIP_BAND_PX)
        if int(np.count_nonzero(band)) < 5:
            band = xs <= (float(xs.min()) + THREAD_TIP_BAND_PX + 4.0)
        if int(np.count_nonzero(band)) == 0:
            return None

        bx = xs[band]
        by = ys[band]
        tip_x = float(np.median(bx))
        tip_y = float(np.median(by))

        # Ketebalan lokal di muka ujung benang. Percentile menahan outlier/streak.
        y_lo = float(np.percentile(by, 5.0))
        y_hi = float(np.percentile(by, 95.0))
        local_thickness = max(16.0, min(65.0, (y_hi - y_lo) + 4.0))

        # Oval tetap sempit pada arah gerak, tetapi tingginya mengikuti lebar/ketebalan benang.
        oval_w = max(8.0, min(30.0, local_thickness * 0.38))
        oval_h = local_thickness
        tip_ellipse = ((tip_x, tip_y), (oval_w, oval_h), 0.0)

        return (tip_x, tip_y), tip_ellipse, local_thickness

    def detect(self, frame):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = [
            int(v * s) for v, s in zip(THREAD_ROI, (w, h, w, h))
        ]
        roi = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            np.array(THREAD_HSV_LOW, np.uint8),
            np.array(THREAD_HSV_HIGH, np.uint8),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < THREAD_MIN_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            gx, gy = x0 + x, y0 + y
            right_x = gx + bw
            if right_x < int(THREAD_MIN_RIGHT_X * w):
                continue
            if bw < 3 or bh < 3:
                continue

            component = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(component, [contour], -1, 255, -1)
            geom = self._tip_geometry(component, x0, y0)
            if geom is None:
                continue

            tip, tip_ellipse, local_thickness = geom
            tip_x, tip_y = tip
            score = area + 2.5 * bw + 0.5 * bh

            if self.prev_tip is not None and self.missed <= 5:
                d = float(np.hypot(tip_x - self.prev_tip[0], tip_y - self.prev_tip[1]))
                score -= d

            candidates.append(
                {
                    "score": score,
                    "area": area,
                    "bbox": (gx, gy, bw, bh),
                    "tip": tip,
                    "tip_ellipse": tip_ellipse,
                    "local_thickness": float(local_thickness),
                }
            )

        if not candidates:
            self.missed += 1
            if self.missed > 5:
                self.prev_tip = None
            return None

        best = max(candidates, key=lambda item: item["score"])
        self.prev_tip = best["tip"]
        self.missed = 0
        return best


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
        tip = tuple(int(round(v)) for v in thread["tip"])
        cv2.ellipse(out, thread["tip_ellipse"], (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, tip, 2, (0, 0, 255), -1, cv2.LINE_AA)
        txt = f"BENANG TIP {tip[0]},{tip[1]}"
    else:
        txt = "BENANG TIP: tidak terdeteksi"
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
