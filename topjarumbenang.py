"""Pelacak lubang jarum kiri + ujung benang coklat.

Kalibrasi jarum mengikuti video 2026-09-15-133849.webm.
Ujung benang dicari di ujung kiri sumbu tengah (centerline), bukan di
median pita mask, supaya oval menempel pada ujung fisik meski bagian
ujung lebih gelap atau mask HSV bolong.
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
# ROI sampai tepi kanan: benang masuk dari kanan frame.
THREAD_ROI = (0.08, 0.04, 0.995, 0.80)
THREAD_HSV_LOW = (0, 40, 12)
THREAD_HSV_HIGH = (26, 255, 250)
THREAD_HSV_WEAK_LOW = (0, 18, 7)
THREAD_HSV_WEAK_HIGH = (26, 255, 245)
THREAD_HUE_WRAP = 168
THREAD_MIN_RB = 8
THREAD_MIN_RB_WEAK = 11
THREAD_MIN_AREA = 140.0
THREAD_MIN_RIGHT_X = 0.68
THREAD_MIN_ASPECT = 1.12
THREAD_MAX_TIP_JUMP = 110.0
THREAD_SMOOTH = 0.58
THREAD_EXTEND_MAX = 40
THREAD_ANGLE_LIMIT = 12.0


def transformed_ellipse(ellipse, matrix):
    (cx, cy), (a, b), angle = ellipse
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    r = np.deg2rad(angle)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    points = np.column_stack((a / 2 * np.cos(t), b / 2 * np.sin(t))) @ rot.T
    points += (cx, cy)
    points = points @ matrix[:, :2].T + matrix[:, 2]
    return cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))


def _theil_sen_angle(xs, ys):
    """Sudut derajat dari slope Theil–Sen (median pasangan), tahan outlier."""
    n = len(xs)
    slopes = []
    for i in range(n - 1):
        dx = xs[i + 1 :] - xs[i]
        dy = ys[i + 1 :] - ys[i]
        ok = np.abs(dx) > 1e-6
        if np.any(ok):
            slopes.append(dy[ok] / dx[ok])
    if not slopes:
        return 0.0
    slope = float(np.median(np.concatenate(slopes)))
    return float(np.degrees(np.arctan(slope)))


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
    """Ujung kiri benang coklat pada centerline, presisi subpiksel.

    Langkah:
    1) segmentasi copper/coklat yang longgar di ujung gelap;
    2) pilih blob horizontal yang menjulur ke kanan;
    3) ketebalan dari radius distance-transform di badan benang;
    4) perpanjang ke kiri sepanjang pita centerline ke material gelap;
    5) X = kolom kiri pertama yang memotong pita; Y = centerline lokal.
    """

    def __init__(self):
        self.prev_tip = None
        self.prev_angle = 0.0
        self.missed = 0

    @staticmethod
    def _hue_ok(h, high=26, wrap=THREAD_HUE_WRAP):
        return (h <= high) | (h >= wrap)

    @classmethod
    def segment(cls, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        b, g, r = cv2.split(roi)
        rb = r.astype(np.int16) - b.astype(np.int16)
        rg = r.astype(np.int16) - g.astype(np.int16)
        hue = cls._hue_ok(h)
        strong = (
            hue
            & (s >= THREAD_HSV_LOW[1])
            & (v >= THREAD_HSV_LOW[2])
            & (v <= THREAD_HSV_HIGH[2])
            & (rb >= THREAD_MIN_RB)
        )
        weak = (
            hue
            & (s >= THREAD_HSV_WEAK_LOW[1])
            & (v >= THREAD_HSV_WEAK_LOW[2])
            & (v <= THREAD_HSV_WEAK_HIGH[2])
            & (rb >= THREAD_MIN_RB_WEAK)
            & (rg >= -4)
        )
        mask = np.where(strong | weak, 255, 0).astype(np.uint8)
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
        )
        return mask, hsv, rb

    @staticmethod
    def _column_profile(component):
        ys, xs = np.where(component > 0)
        if len(xs) < 20:
            return None
        xmin, xmax = int(xs.min()), int(xs.max())
        width = xmax - xmin + 1
        counts = np.bincount(xs - xmin, minlength=width).astype(np.float64)
        y_min = np.full(width, np.inf)
        y_max = np.full(width, -np.inf)
        np.minimum.at(y_min, xs - xmin, ys)
        np.maximum.at(y_max, xs - xmin, ys)
        return xmin, xmax, counts, y_min, y_max

    @classmethod
    def _extend_left(cls, hsv, rb, lead_x, center_y, thickness):
        """Tumbuhkan ujung ke kiri di pita centerline untuk benang yang gelap."""
        h, s, v = cv2.split(hsv)
        height = h.shape[0]
        y_a = int(max(0, round(center_y - thickness * 0.52)))
        y_b = int(min(height, round(center_y + thickness * 0.52)))
        if y_b - y_a < 3:
            return lead_x
        ext_x = lead_x
        x = lead_x - 1
        while x >= 0 and (lead_x - x) <= THREAD_EXTEND_MAX:
            hue = h[y_a:y_b, x]
            sat = s[y_a:y_b, x]
            val = v[y_a:y_b, x]
            chroma = rb[y_a:y_b, x]
            copper = (
                cls._hue_ok(hue, high=30)
                & (chroma >= 5)
                & (sat >= 10)
                & (val >= 5)
                & (val <= 210)
            )
            frac = float(np.mean(copper))
            if frac < 0.22:
                break
            # Logam abu-abu di alur: saturasi dan R-B rendah.
            if float(np.median(sat)) < 16 and float(np.median(chroma)) < 10:
                break
            ys_ok = np.where(copper)[0]
            if len(ys_ok) < 3:
                break
            if abs(float(np.median(ys_ok) + y_a) - center_y) > thickness * 0.35:
                break
            ext_x = x
            x -= 1
        return ext_x

    @classmethod
    def _tip_geometry(cls, component, hsv, rb, x0, y0):
        prof = cls._column_profile(component)
        if prof is None:
            return None
        xmin, xmax, counts, y_min, y_max = prof
        width = xmax - xmin + 1
        if width < 10:
            return None

        min_count = 4
        valid = counts >= min_count
        if not np.any(valid):
            return None
        lead_i = int(np.argmax(valid))
        lead_x = xmin + lead_i

        body_start = min(width - 1, lead_i + 8)
        body_end = min(width, max(body_start + 8, lead_i + 26))
        body = slice(body_start, body_end)
        body_valid = counts[body] >= min_count
        if int(np.count_nonzero(body_valid)) < 4:
            body = slice(lead_i, min(width, lead_i + 40))
            body_start, body_end = lead_i, min(width, lead_i + 40)
            body_valid = counts[body] >= min_count
        if int(np.count_nonzero(body_valid)) < 3:
            return None

        dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        body_dist = dist[:, xmin + body_start : xmin + body_end]
        if body_dist.size == 0:
            return None
        col_radius = body_dist.max(axis=0)
        finite = col_radius > 0.8
        if not np.any(finite):
            return None
        radius = float(np.median(col_radius[finite]))
        thickness = float(np.clip(2.0 * radius + 1.0, 8.0, 48.0))

        # Centerline: baris radius maksimum di tiap kolom badan.
        col_center = np.argmax(body_dist, axis=0).astype(np.float64)
        xs_b = (xmin + np.arange(body_start, body_end)).astype(np.float64)
        xs_b = xs_b[finite]
        ys_b = col_center[finite]
        # Median piksel badan lebih stabil daripada satu baris argmax.
        body_pixels = component[:, xmin + body_start : xmin + body_end] > 0
        if np.any(body_pixels):
            ys_pix, _ = np.where(body_pixels)
            center_y = float(np.median(ys_pix.astype(np.float64)))
        else:
            center_y = float(np.median(ys_b))
        if len(xs_b) >= 5:
            angle = float(np.clip(_theil_sen_angle(xs_b, ys_b), -THREAD_ANGLE_LIMIT, THREAD_ANGLE_LIMIT))
            slope = float(np.tan(np.radians(angle)))
            body_x_mean = float(xs_b.mean())
        else:
            slope, angle, body_x_mean = 0.0, 0.0, float(lead_x + 12)

        ext_x = cls._extend_left(hsv, rb, lead_x, center_y, thickness)

        band = max(3.0, thickness * 0.42)
        if ext_x < lead_x:
            # Material gelap di kiri mask HSV: pakai hasil pertumbuhan.
            tip_x = float(ext_x)
        else:
            tip_x = float(lead_x)
            for i in range(lead_i, width):
                if counts[i] < 3:
                    continue
                if y_min[i] <= center_y + band and y_max[i] >= center_y - band:
                    tip_x = float(xmin + i)
                    break

        i0 = int(round(tip_x - xmin))
        if 0 <= i0 < width - 1:
            expected = max(thickness * 0.35, 3.0)
            frac = float(np.clip(counts[i0] / expected, 0.15, 1.0))
            tip_x = tip_x - 0.5 * frac

        # Jangan ekstrapolasi Y dengan sudut curam: alur benang hampir horizontal.
        if abs(angle) <= 8.0:
            tip_y = center_y + slope * (tip_x - body_x_mean)
        else:
            tip_y = center_y
            angle = 0.0
        oval_w = max(6.0, min(22.0, thickness * 0.38))
        oval_h = thickness
        tip = (float(tip_x + x0), float(tip_y + y0))
        ellipse = (tip, (oval_w, oval_h), angle)
        return tip, ellipse, thickness, angle

    def detect(self, frame):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = [int(v * s) for v, s in zip(THREAD_ROI, (w, h, w, h))]
        roi = frame[y0:y1, x0:x1]
        mask, hsv, rb = self.segment(roi)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < THREAD_MIN_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            gx, gy = x0 + x, y0 + y
            if gx + bw < int(THREAD_MIN_RIGHT_X * w):
                continue
            if bw < max(12, int(bh * THREAD_MIN_ASPECT)):
                continue

            component = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(component, [contour], -1, 255, -1)
            component = cv2.morphologyEx(
                component, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8)
            )
            geom = self._tip_geometry(component, hsv, rb, x0, y0)
            if geom is None:
                continue

            tip, tip_ellipse, local_thickness, angle = geom
            tip_x, tip_y = tip
            score = area + 3.0 * bw

            if self.prev_tip is not None and self.missed <= 5:
                d = float(np.hypot(tip_x - self.prev_tip[0], tip_y - self.prev_tip[1]))
                score -= 1.8 * d
                if d > THREAD_MAX_TIP_JUMP and self.missed <= 2:
                    score -= 5000.0

            candidates.append(
                {
                    "score": score,
                    "area": area,
                    "bbox": (gx, gy, bw, bh),
                    "tip": tip,
                    "tip_ellipse": tip_ellipse,
                    "local_thickness": float(local_thickness),
                    "angle": float(angle),
                }
            )

        if not candidates:
            self.missed += 1
            if self.missed > 6:
                self.prev_tip = None
            return None

        best = max(candidates, key=lambda item: item["score"])
        tip = best["tip"]
        if self.prev_tip is not None and self.missed <= 4:
            jump = float(np.hypot(tip[0] - self.prev_tip[0], tip[1] - self.prev_tip[1]))
            if jump < 40.0:
                a = THREAD_SMOOTH
                tip = (
                    a * tip[0] + (1.0 - a) * self.prev_tip[0],
                    a * tip[1] + (1.0 - a) * self.prev_tip[1],
                )
                best["tip"] = tip
                (cx, cy), axes, ang = best["tip_ellipse"]
                best["tip_ellipse"] = (tip, axes, ang)

        self.prev_tip = best["tip"]
        self.prev_angle = best["angle"]
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
        txt = f"BENANG TIP {thread['tip'][0]:.1f},{thread['tip'][1]:.1f}"
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
                        "thread_tip_angle": thread.get("angle") if thread else None,
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
