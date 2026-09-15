"""
GUI dua sumber kamera/video untuk pelacakan pinhole.

Jalankan:
    python pinhole_gui.py --top top.webm

TOP:
    Template matching multi-skala + ECC affine.
SIDE:
    Preview dan pengaturan citra; algoritme deteksi belum ditambahkan.

Kalibrasi disimpan dalam koordinat referensi 640 x 480.
Diameter oval dan koordinat hasil merupakan piksel, bukan ukuran metrologi.
"""

import argparse
import base64
import glob
import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)


# key, label, minimum, maksimum, default, langkah
IMAGE_FIELDS = [
    ("brightness", "Kecerahan", -100, 100, 0, 1),
    ("contrast", "Kontras", 0.10, 3.0, 1.0, 0.05),
    ("gamma", "Gamma (>1 lebih terang)", 0.20, 3.0, 1.0, 0.05),
    ("clahe", "CLAHE (0 = mati)", 0.0, 8.0, 0.0, 0.20),
    ("blur", "Radius Gaussian blur", 0, 5, 0, 1),
]

HSV_FIELDS = [
    ("hmin", "H minimum", 0, 179, 0, 1),
    ("hmax", "H maksimum", 0, 179, 179, 1),
    ("smin", "S minimum", 0, 255, 0, 1),
    ("smax", "S maksimum", 0, 255, 255, 1),
    ("vmin", "V minimum", 0, 255, 0, 1),
    ("vmax", "V maksimum", 0, 255, 255, 1),
    ("opening", "Radius opening (0 = mati)", 0, 5, 0, 1),
    ("closing", "Radius closing (0 = mati)", 0, 5, 0, 1),
]

TRACK_FIELDS = [
    ("min_match", "Minimum match", 0.0, 1.0, 0.68, 0.01),
    ("min_ecc", "Minimum ECC", 0.0, 1.0, 0.72, 0.01),
    ("scale_min", "Skala minimum", 0.50, 2.0, 0.90, 0.05),
    ("scale_max", "Skala maksimum", 0.50, 2.0, 1.10, 0.05),
    ("scale_count", "Jumlah skala", 1, 21, 5, 1),
    ("ecc_iterations", "Iterasi ECC", 10, 200, 50, 10),
    ("ecc_shift", "Batas translasi ECC (px)", 1, 100, 10, 1),
]

GEOMETRY_FIELDS = [
    ("cx", "Pusat oval X", 0.0, 640.0, 175.5, 0.5),
    ("cy", "Pusat oval Y", 0.0, 480.0, 222.5, 0.5),
    ("axis_a", "Diameter sumbu A", 1.0, 640.0, 25.0, 0.5),
    ("axis_b", "Diameter sumbu B", 1.0, 640.0, 67.0, 0.5),
    ("angle", "Sudut oval (derajat)", -180.0, 180.0, -6.0, 0.5),
    ("tx", "Template X", 0.0, 639.0, 150.0, 1.0),
    ("ty", "Template Y", 0.0, 479.0, 178.0, 1.0),
    ("tw", "Lebar template", 3.0, 640.0, 49.0, 1.0),
    ("th", "Tinggi template", 3.0, 480.0, 91.0, 1.0),
    ("rx0", "Search X awal (0–1)", 0.0, 1.0, 0.10, 0.01),
    ("ry0", "Search Y awal (0–1)", 0.0, 1.0, 0.18, 0.01),
    ("rx1", "Search X akhir (0–1)", 0.0, 1.0, 0.46, 0.01),
    ("ry1", "Search Y akhir (0–1)", 0.0, 1.0, 0.64, 0.01),
]

ALL_FIELDS = IMAGE_FIELDS + HSV_FIELDS + TRACK_FIELDS + GEOMETRY_FIELDS
DEFAULTS = {key: value for key, _, _, _, value, _ in ALL_FIELDS}
DEFAULTS.update(
    hsv_on=False,
    invert=False,
    ecc_on=True,
    loop=True,
    guides=False,
)

BACKENDS = [
    ("Auto", cv2.CAP_ANY),
    ("V4L2 / Linux", cv2.CAP_V4L2),
    ("DirectShow / Windows", cv2.CAP_DSHOW),
    ("MSMF / Windows", cv2.CAP_MSMF),
]


def detect_cameras():
    """Deteksi kamera yang terhubung dari /sys/class/video4linux.

    Mengembalikan list of (label, index) untuk setiap perangkat capture
    (hanya index=0 di sysfs, yaitu node capture utama).
    Kamera internal (integrated) dan eksternal (USB) diberi label berbeda.
    Jika tidak ada kamera terdeteksi, kembalikan fallback indeks 0–9.
    """
    cameras = []
    if sys.platform.startswith("linux"):
        sysfs_dirs = sorted(
            glob.glob("/sys/class/video4linux/video*"),
            key=lambda p: int(os.path.basename(p).replace("video", "")),
        )
        for dev_path in sysfs_dirs:
            dev_name = os.path.basename(dev_path)
            try:
                num = int(dev_name.replace("video", ""))
            except ValueError:
                continue

            # Hanya ambil node capture utama (index 0 di sysfs).
            index_file = os.path.join(dev_path, "index")
            if os.path.exists(index_file):
                try:
                    with open(index_file) as fh:
                        if fh.read().strip() != "0":
                            continue
                except OSError:
                    pass

            # Baca nama perangkat dari sysfs.
            name_file = os.path.join(dev_path, "name")
            hw_name = ""
            if os.path.exists(name_file):
                try:
                    with open(name_file) as fh:
                        hw_name = fh.read().strip()
                except OSError:
                    pass

            # Tentukan label internal / eksternal.
            lower = hw_name.lower()
            if any(kw in lower for kw in ("integrated", "internal", "built-in")):
                tag = "Internal"
            else:
                tag = "Eksternal USB"

            if hw_name:
                label = f"Kamera {num}: {hw_name} ({tag})"
            else:
                label = f"Kamera {num}"

            cameras.append((label, num))

    # Fallback: jika sysfs tidak tersedia, tampilkan 0–9 generik.
    if not cameras:
        cameras = [(f"Kamera indeks {i}", i) for i in range(10)]

    return cameras


def validate_parameters(p, top=True):
    for key, _, low, high, _, _ in ALL_FIELDS:
        value = p[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"Parameter tidak valid: {key}")

    for key in ("hsv_on", "invert", "ecc_on", "loop", "guides"):
        if not isinstance(p[key], bool):
            raise ValueError(f"Parameter harus boolean: {key}")

    if p["smin"] > p["smax"] or p["vmin"] > p["vmax"]:
        raise ValueError("S/V minimum harus <= maksimum.")

    if not top:
        return

    if p["scale_min"] > p["scale_max"]:
        raise ValueError("Skala minimum harus <= maksimum.")

    if not (p["rx0"] < p["rx1"] and p["ry0"] < p["ry1"]):
        raise ValueError("Koordinat awal search harus lebih kecil dari akhir.")

    if p["tx"] + p["tw"] > 640 or p["ty"] + p["th"] > 480:
        raise ValueError("Template melewati batas referensi 640 x 480.")


def prepare_image(frame, p):
    """Proses yang sama digunakan untuk frame dan gambar referensi."""
    adjusted = np.clip(
        frame.astype(np.float32) * p["contrast"] + p["brightness"],
        0, 255,
    ).astype(np.uint8)

    if abs(p["gamma"] - 1.0) > 1e-6:
        lut = np.clip(
            (np.arange(256, dtype=np.float32) / 255.0)
            ** (1.0 / p["gamma"]) * 255.0,
            0, 255,
        ).astype(np.uint8)
        adjusted = cv2.LUT(adjusted, lut)

    if p["clahe"] > 0:
        lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=float(p["clahe"]), tileGridSize=(8, 8)
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    radius = int(p["blur"])
    if radius:
        kernel = 2 * radius + 1
        adjusted = cv2.GaussianBlur(adjusted, (kernel, kernel), 0)

    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    h0, h1 = int(p["hmin"]), int(p["hmax"])
    s0, s1 = int(p["smin"]), int(p["smax"])
    v0, v1 = int(p["vmin"]), int(p["vmax"])

    # H minimum > maksimum berarti rentang hue melewati 179 -> 0.
    if h0 <= h1:
        mask = cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    else:
        a = cv2.inRange(hsv, (h0, s0, v0), (179, s1, v1))
        b = cv2.inRange(hsv, (0, s0, v0), (h1, s1, v1))
        mask = cv2.bitwise_or(a, b)

    if p["invert"]:
        mask = cv2.bitwise_not(mask)

    for key, operation in (
        ("opening", cv2.MORPH_OPEN),
        ("closing", cv2.MORPH_CLOSE),
    ):
        radius = int(p[key])
        if radius:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            mask = cv2.morphologyEx(mask, operation, kernel)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    if p["hsv_on"]:
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    return adjusted, mask, gray


def ellipse_points(ellipse):
    (cx, cy), (a, b), angle = ellipse
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    r = np.deg2rad(angle)
    rotation = np.array([
        [np.cos(r), -np.sin(r)],
        [np.sin(r), np.cos(r)],
    ])
    points = np.column_stack(
        (a / 2 * np.cos(t), b / 2 * np.sin(t))
    )
    return points @ rotation.T + (cx, cy)


def transformed_ellipse(ellipse, matrix):
    points = ellipse_points(ellipse)
    points = points @ matrix[:, :2].T + matrix[:, 2]
    return cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))


def geometry(p, width, height):
    sx, sy = width / 640.0, height / 480.0

    reference = (
        (p["cx"], p["cy"]),
        (p["axis_a"], p["axis_b"]),
        p["angle"],
    )
    ellipse = transformed_ellipse(
        reference,
        np.array([[sx, 0, 0], [0, sy, 0]], np.float32),
    )

    x = round(p["tx"] * sx)
    y = round(p["ty"] * sy)
    x_end = round((p["tx"] + p["tw"]) * sx)
    y_end = round((p["ty"] + p["th"]) * sy)

    search = (
        int(p["rx0"] * width), int(p["ry0"] * height),
        int(p["rx1"] * width), int(p["ry1"] * height),
    )
    return ellipse, (x, y, x_end - x, y_end - y), search


class PinholeTracker:
    def __init__(self, reference_frame):
        self.reference = reference_frame.copy()
        self.cache_key = None
        self.templates = []

    def prepare_reference(self, shape, p):
        height, width = shape
        key = (height, width, json.dumps(p, sort_keys=True))
        if key == self.cache_key:
            return

        reference = self.reference
        if reference.shape[:2] != shape:
            reference = cv2.resize(reference, (width, height))

        _, _, gray = prepare_image(reference, p)
        self.ellipse, self.box, self.search = geometry(p, width, height)
        tx, ty, tw, th = self.box

        if (
            tw < 3 or th < 3 or tx < 0 or ty < 0
            or tx + tw > width or ty + th > height
        ):
            raise ValueError("Template terlalu kecil atau di luar gambar.")

        points = ellipse_points(self.ellipse)
        if (
            points[:, 0].min() < tx
            or points[:, 0].max() >= tx + tw
            or points[:, 1].min() < ty
            or points[:, 1].max() >= ty + th
        ):
            raise ValueError("Perbesar template agar seluruh oval berada di dalamnya.")

        template = gray[ty:ty + th, tx:tx + tw].copy()
        if template.std() < 1.0:
            raise ValueError(
                "Template terlalu seragam. Periksa HSV, cahaya, atau kalibrasi."
            )

        self.templates = []
        for scale in np.linspace(
            p["scale_min"], p["scale_max"], int(p["scale_count"])
        ):
            size = (max(3, round(tw * scale)), max(3, round(th * scale)))
            candidate = cv2.resize(template, size)
            if candidate.std() >= 1.0:
                self.templates.append(candidate)

        self.cache_key = key

    def detect(self, gray, p):
        self.prepare_reference(gray.shape, p)

        x0, y0, x1, y1 = self.search
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0 or roi.std() < 1.0:
            return None, "Area pencarian kosong atau terlalu seragam."

        best = None
        for template in self.templates:
            hh, ww = template.shape
            if hh > roi.shape[0] or ww > roi.shape[1]:
                continue

            scores = cv2.matchTemplate(
                roi, template, cv2.TM_CCOEFF_NORMED
            )
            scores = np.nan_to_num(
                scores, nan=-1.0, posinf=-1.0, neginf=-1.0
            )
            _, score, _, location = cv2.minMaxLoc(scores)
            if best is None or score > best[0]:
                best = (float(score), location, template)

        if best is None:
            return None, "Search terlalu kecil untuk ukuran template."

        if best[0] < p["min_match"]:
            return None, f"Tidak terdeteksi | match terbaik {best[0]:.3f}"

        score, (dx, dy), template = best
        hh, ww = template.shape
        px, py = x0 + dx, y0 + dy
        patch = gray[py:py + hh, px:px + ww]

        if patch.std() < 1.0:
            return None, "Patch kandidat terlalu seragam."

        warp = np.eye(2, 3, dtype=np.float32)
        ecc_score = None
        refined = False

        if p["ecc_on"]:
            try:
                cc, candidate = cv2.findTransformECC(
                    template,
                    patch,
                    warp.copy(),
                    cv2.MOTION_AFFINE,
                    (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        int(p["ecc_iterations"]),
                        1e-4,
                    ),
                    None,
                    3,
                )

                if np.isfinite(cc) and np.isfinite(candidate).all():
                    ecc_score = float(cc)
                    singular = np.linalg.svd(
                        candidate[:, :2], compute_uv=False
                    )
                    valid = (
                        cc >= p["min_ecc"]
                        and np.all((singular > 0.85) & (singular < 1.18))
                        and np.linalg.det(candidate[:, :2]) > 0
                        and np.linalg.norm(candidate[:, 2]) < p["ecc_shift"]
                    )
                    if valid:
                        warp = candidate
                        refined = True
            except cv2.error:
                pass

        tx, ty, tw, th = self.box
        sx, sy = ww / tw, hh / th

        # Koordinat full frame referensi -> template hasil resize.
        # Koreksi setengah piksel mengikuti pemetaan pusat piksel resize.
        origin = np.array([
            [sx, 0, -tx * sx + (sx - 1) / 2],
            [0, sy, -ty * sy + (sy - 1) / 2],
            [0, 0, 1],
        ], dtype=np.float32)

        # ECC memetakan koordinat template -> patch saat ini.
        affine = np.vstack((warp, [0, 0, 1])) @ origin
        affine[0, 2] += px
        affine[1, 2] += py

        ellipse = transformed_ellipse(self.ellipse, affine[:2])
        cx, cy = ellipse[0]
        height, width = gray.shape

        if not (
            np.isfinite(ellipse_points(ellipse)).all()
            and 0 <= cx < width
            and 0 <= cy < height
        ):
            return None, "Transformasi di luar frame."

        result = {
            "ellipse": ellipse,
            "center": (float(cx), float(cy)),
            "score": score,
            "ecc": ecc_score,
            "refined": refined,
        }
        return result, "TM + ECC" if refined else "TM; tanpa penyempurnaan ECC"


def draw_guides(frame, p):
    out = frame.copy()
    height, width = out.shape[:2]
    ellipse, (x, y, tw, th), (x0, y0, x1, y1) = geometry(
        p, width, height
    )
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 180, 0), 1)
    cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 220, 255), 1)
    cv2.ellipse(out, ellipse, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def draw_detection(frame, result):
    out = frame.copy()
    if result:
        cv2.ellipse(out, result["ellipse"], (0, 255, 0), 1, cv2.LINE_AA)
        center = tuple(round(v) for v in result["center"])
        cv2.circle(out, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
        label = f"PINHOLE {center[0]},{center[1]}"
    else:
        label = "PINHOLE: tidak terdeteksi"

    cv2.putText(
        out, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 1, cv2.LINE_AA,
    )
    return out


class CaptureWorker(threading.Thread):
    """Capture dan tracking berjalan di luar thread GUI."""

    def __init__(self, source, backend, size, fps, p, reference, top):
        super().__init__(daemon=True)
        self.source = source
        self.backend = backend
        self.size = size
        self.requested_fps = fps
        self.top = top
        self.initial_reference = (
            reference.copy() if reference is not None else None
        )
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()
        self.parameters = p.copy()
        self.pending_reference = None

        self.latest = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.commands = queue.Queue()

    def configure(self, p, reference=None):
        with self.lock:
            self.parameters = p.copy()
            if reference is not None:
                self.pending_reference = reference.copy()

    def publish(self, packet):
        try:
            self.latest.get_nowait()
        except queue.Empty:
            pass
        self.latest.put_nowait(packet)

    def apply_hardware(self, cap, settings):
        if not isinstance(self.source, int):
            self.events.put(("message", "Kontrol hardware hanya untuk kamera."))
            return

        backend = cap.getBackendName().upper()
        auto_mode, exposure, gain = settings
        messages = []

        if auto_mode is not None:
            if "V4L" in backend:
                value = 0.75 if auto_mode else 0.25
            elif "DSHOW" in backend:
                value = 1.0 if auto_mode else 0.0
            else:
                value = None
                messages.append(
                    f"Auto exposure {backend}: atur melalui aplikasi driver."
                )

            if value is not None:
                ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
                messages.append(
                    "Perintah auto exposure diterima."
                    if ok else "Auto exposure ditolak driver."
                )

        if auto_mode is False:
            ok = cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            readback = cap.get(cv2.CAP_PROP_EXPOSURE)
            messages.append(
                f"Exposure: {'diterima' if ok else 'ditolak'}, "
                f"readback={readback:g}."
            )

        ok = cap.set(cv2.CAP_PROP_GAIN, float(gain))
        readback = cap.get(cv2.CAP_PROP_GAIN)
        messages.append(
            f"Gain: {'diterima' if ok else 'ditolak'}, readback={readback:g}."
        )
        self.events.put(("message", " ".join(messages)))

    def run(self):
        cap = None
        try:
            camera = isinstance(self.source, int)
            cap = cv2.VideoCapture(
                self.source, self.backend if camera else cv2.CAP_ANY
            )
            if not cap.isOpened():
                raise RuntimeError(
                    "Sumber tidak dapat dibuka. Periksa indeks, path, "
                    "backend, izin kamera, atau aplikasi lain."
                )

            if camera:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or not 0.1 <= fps <= 240:
                fps = self.requested_fps

            self.events.put((
                "message",
                f"Terbuka: {cap.getBackendName()} | FPS dilaporkan {fps:.2f}",
            ))

            tracker = (
                PinholeTracker(self.initial_reference)
                if self.top and self.initial_reference is not None
                else None
            )
            raw = None
            index = -1
            failures = 0

            while not self.stop_event.is_set():
                tick = time.monotonic()

                with self.lock:
                    p = self.parameters.copy()
                    reference = self.pending_reference
                    self.pending_reference = None

                if reference is not None and self.top:
                    tracker = PinholeTracker(reference)

                while True:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self.apply_hardware(cap, command)
                    except cv2.error as exc:
                        self.events.put(("message", f"Kontrol kamera: {exc}"))

                if camera or not self.paused.is_set() or raw is None:
                    ok, next_frame = cap.read()

                    if not ok and not camera and p["loop"]:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, next_frame = cap.read()
                        index = -1

                    if not ok:
                        if not camera:
                            self.events.put(("message", "Video selesai."))
                            break
                        failures += 1
                        if failures >= 10:
                            raise RuntimeError("Kamera berhenti mengirim frame.")
                        self.stop_event.wait(0.1)
                        continue

                    failures = 0
                    raw = next_frame
                    index += 1

                if self.top and tracker is None:
                    tracker = PinholeTracker(raw)
                    self.events.put(("reference", raw.copy()))

                result = None
                adjusted = raw
                mask = np.zeros(raw.shape[:2], np.uint8)
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

                try:
                    validate_parameters(p, self.top)
                    adjusted, mask, gray = prepare_image(raw, p)
                    if self.top:
                        result, note = tracker.detect(gray, p)
                    else:
                        note = "SIDE: preview; detektor belum ditambahkan."
                except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                    note = str(exc)

                elapsed_ms = (time.monotonic() - tick) * 1000
                self.publish({
                    "raw": raw,
                    "adjusted": adjusted,
                    "mask": mask,
                    "gray": gray,
                    "result": result,
                    "note": note,
                    "index": index,
                    "ms": elapsed_ms,
                })

                if not camera:
                    period = 1.0 / (30 if self.paused.is_set() else fps)
                    self.stop_event.wait(
                        max(0, period - (time.monotonic() - tick))
                    )

        except Exception as exc:
            self.events.put(("message", f"ERROR: {exc}"))
        finally:
            if cap is not None:
                cap.release()


class VideoView(QWidget):
    selected = Signal(float, float, float, float)

    def __init__(self):
        super().__init__()
        self.image = None
        self.image_rect = QRectF()
        self.selecting = False
        self.drag_start = None
        self.drag_end = None
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, frame):
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        self.image = QImage(
            frame.data, width, height, frame.strides[0],
            QImage.Format.Format_BGR888,
        ).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#08111c"))

        if self.image is None:
            painter.setPen(QColor("#93a6bc"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Pilih sumber lalu klik Mulai",
            )
            return

        width, height = self.image.width(), self.image.height()
        scale = min(self.width() / width, self.height() / height)
        self.image_rect = QRectF(
            (self.width() - width * scale) / 2,
            (self.height() - height * scale) / 2,
            width * scale,
            height * scale,
        )
        painter.drawImage(self.image_rect, self.image)

        if self.drag_start is not None and self.drag_end is not None:
            painter.setPen(QPen(QColor("#ffdc63"), 2))
            painter.drawRect(
                QRectF(self.drag_start, self.drag_end).normalized()
            )

    def mousePressEvent(self, event):
        if (
            self.selecting
            and self.image is not None
            and event.button() == Qt.MouseButton.LeftButton
            and self.image_rect.contains(event.position())
        ):
            self.drag_start = event.position()
            self.drag_end = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            self.drag_end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_start is None or self.image is None:
            return

        rect = self.image_rect
        if rect.width() <= 0 or rect.height() <= 0:
            return

        def convert(point):
            x = (point.x() - rect.left()) / rect.width()
            y = (point.y() - rect.top()) / rect.height()
            return (
                float(np.clip(x, 0, 1) * self.image.width()),
                float(np.clip(y, 0, 1) * self.image.height()),
            )

        x0, y0 = convert(self.drag_start)
        x1, y1 = convert(event.position())
        self.drag_start = None
        self.drag_end = None
        self.update()

        if abs(x1 - x0) >= 3 and abs(y1 - y0) >= 3:
            self.selected.emit(
                min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
            )


class CameraPane(QGroupBox):
    def __init__(self, title, top, owner, initial_file=""):
        super().__init__(title)
        self.top = top
        self.owner = owner
        self.worker = None
        self.packet = None
        self.reference = None
        self.previous_source = None
        self.active_source = None
        self.frozen = None
        self.shown = None
        self.old_parameters = None
        self.was_paused = False
        self.loading = False
        self.recorder = None
        self.recording = False
        self.record_path = None
        self.inputs = {}
        self.checks = {}

        layout = QVBoxLayout(self)

        self.source = QComboBox()
        self._detected_cameras = detect_cameras()
        for label, cam_index in self._detected_cameras:
            self.source.addItem(label, cam_index)
        self.source.addItem("File video", "file")
        file_idx = len(self._detected_cameras)
        if initial_file:
            self.source.setCurrentIndex(file_idx)
        elif len(self._detected_cameras) >= 2 and not top:
            self.source.setCurrentIndex(1)
        else:
            self.source.setCurrentIndex(0)

        self.backend = QComboBox()
        for label, value in BACKENDS:
            self.backend.addItem(label, value)

        self.resolution = QComboBox()
        for size in ("640x480", "800x600", "1280x720", "1920x1080"):
            self.resolution.addItem(size)

        self.fps = QDoubleSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setDecimals(0)
        self.fps.setValue(30)
        self.fps.setSuffix(" FPS")

        source_row = QHBoxLayout()
        for widget in (self.source, self.backend, self.resolution, self.fps):
            source_row.addWidget(widget)
        layout.addLayout(source_row)

        self.file_path = QLineEdit(initial_file)
        self.file_path.setPlaceholderText("Path video, misalnya top.webm")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse)

        file_row = QHBoxLayout()
        file_row.addWidget(self.file_path, 1)
        file_row.addWidget(self.browse_button)
        layout.addLayout(file_row)

        self.start_button = QPushButton("Mulai")
        self.stop_button = QPushButton("Stop")
        self.pause_button = QPushButton("Pause video")
        self.pause_button.setCheckable(True)
        self.snapshot_button = QPushButton("Snapshot")
        self.record_button = QPushButton("⏺ Rekam")
        self.record_button.setCheckable(True)

        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.pause_button.toggled.connect(self.pause)
        self.snapshot_button.clicked.connect(self.snapshot)
        self.record_button.toggled.connect(self.toggle_recording)

        button_row = QHBoxLayout()
        for button in (
            self.start_button, self.stop_button,
            self.pause_button, self.snapshot_button,
            self.record_button,
        ):
            button_row.addWidget(button)
        layout.addLayout(button_row)

        self.view_mode = QComboBox()
        self.view_mode.addItems([
            "Overlay deteksi" if top else "Preview side",
            "Gambar asli",
            "Gambar terkoreksi",
            "Mask HSV",
            "Grayscale detektor",
        ])
        self.view_mode.currentIndexChanged.connect(self.render)
        layout.addWidget(self.view_mode)

        self.view = VideoView()
        self.view.selected.connect(self.select_region)
        layout.addWidget(self.view, 1)

        self.stats = QLabel("Belum ada frame.")
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)

        self.message = QLabel(
            "Preset awal untuk top.webm. Kalibrasi ulang untuk kamera/video lain."
            if top else "Algoritme side dapat ditambahkan kemudian."
        )
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

        tabs = QTabWidget()
        tabs.setMinimumHeight(230)
        layout.addWidget(tabs)

        image_form = self.add_tab(tabs, "Citra")
        self.add_fields(image_form, IMAGE_FIELDS, sliders=True)
        self.add_check(image_form, "loop", "Ulangi file video")

        hsv_form = self.add_tab(tabs, "HSV")
        self.add_check(hsv_form, "hsv_on", "Gunakan mask HSV pada detektor")
        self.add_check(hsv_form, "invert", "Balik mask")
        self.add_fields(hsv_form, HSV_FIELDS, sliders=True)
        note = QLabel("H: 0–179. H minimum > maksimum mendukung rentang melingkar.")
        note.setWordWrap(True)
        hsv_form.addRow(note)

        if top:
            tracking_form = self.add_tab(tabs, "Tracking")
            self.add_check(tracking_form, "ecc_on", "Aktifkan ECC")
            self.add_check(tracking_form, "guides", "Tampilkan panduan referensi")
            self.add_fields(tracking_form, TRACK_FIELDS)

            geometry_form = self.add_tab(tabs, "Kalibrasi")
            self.calibration_button = QPushButton("Bekukan untuk kalibrasi")
            self.cancel_button = QPushButton("Batal kalibrasi")
            self.cancel_button.setEnabled(False)
            self.calibration_button.clicked.connect(self.calibrate)
            self.cancel_button.clicked.connect(self.cancel_calibration)

            self.selection_mode = QComboBox()
            self.selection_mode.addItems(["Oval", "Template", "Area pencarian"])
            geometry_form.addRow(self.calibration_button)
            geometry_form.addRow(self.cancel_button)
            geometry_form.addRow("Objek yang di-drag", self.selection_mode)

            note = QLabel(
                "Bekukan, drag pada gambar, lalu terapkan. "
                "Kuning: template; biru: search; hijau: oval. "
                "Angka geometri memakai referensi 640 x 480."
            )
            note.setWordWrap(True)
            geometry_form.addRow(note)
            self.add_fields(geometry_form, GEOMETRY_FIELDS)

        hardware_form = self.add_tab(tabs, "Kamera")
        self.auto_exposure = QComboBox()
        self.auto_exposure.addItem("Biarkan pengaturan driver", None)
        self.auto_exposure.addItem("Manual", False)
        self.auto_exposure.addItem("Otomatis", True)

        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(-20, 10000)
        self.exposure.setValue(-6)

        self.gain = QDoubleSpinBox()
        self.gain.setRange(0, 1000)
        self.gain.setValue(0)

        self.hardware_button = QPushButton("Terapkan ke kamera")
        self.hardware_button.clicked.connect(self.apply_hardware)
        hardware_form.addRow("Auto exposure", self.auto_exposure)
        hardware_form.addRow("Exposure", self.exposure)
        hardware_form.addRow("Gain", self.gain)
        hardware_form.addRow(self.hardware_button)

        note = QLabel(
            "Satuan exposure/gain bergantung driver. "
            "Exposure manual dikirim saat mode Manual dipilih. "
            "Hasil perintah dan nilai readback muncul di atas."
        )
        note.setWordWrap(True)
        hardware_form.addRow(note)

        self.source.currentIndexChanged.connect(self.update_controls)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(30)
        self.update_controls()

    @staticmethod
    def add_tab(tabs, name):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        scroll.setWidget(content)
        tabs.addTab(scroll, name)
        return form

    def add_fields(self, form, definitions, sliders=False):
        for key, label, low, high, default, step in definitions:
            spin = QDoubleSpinBox()
            integer = all(
                isinstance(value, int) for value in (low, high, default, step)
            )
            spin.setDecimals(0 if integer else 3)
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setKeyboardTracking(False)
            self.inputs[key] = spin

            if sliders:
                container = QWidget()
                row = QHBoxLayout(container)
                row.setContentsMargins(0, 0, 0, 0)
                slider = QSlider(Qt.Orientation.Horizontal)
                factor = 1 if integer else 1000
                slider.setRange(round(low * factor), round(high * factor))
                slider.setSingleStep(max(1, round(step * factor)))
                slider.setValue(round(default * factor))
                slider.valueChanged.connect(
                    lambda value, target=spin, f=factor:
                    target.setValue(value / f)
                )
                spin.valueChanged.connect(
                    lambda value, target=slider, f=factor:
                    target.setValue(round(value * f))
                )
                row.addWidget(slider, 1)
                row.addWidget(spin)
                form.addRow(label, container)
            else:
                form.addRow(label, spin)

            spin.valueChanged.connect(self.changed)

    def add_check(self, form, key, label):
        checkbox = QCheckBox(label)
        checkbox.setChecked(DEFAULTS[key])
        checkbox.toggled.connect(self.changed)
        self.checks[key] = checkbox
        form.addRow(checkbox)

    def parameters(self):
        p = DEFAULTS.copy()
        p.update({key: widget.value() for key, widget in self.inputs.items()})
        p.update({key: widget.isChecked() for key, widget in self.checks.items()})
        return p

    def fill_parameters(self, p):
        self.loading = True
        try:
            for key, widget in self.inputs.items():
                widget.setValue(p[key])
            for key, widget in self.checks.items():
                widget.setChecked(p[key])
        finally:
            self.loading = False
        self.changed()

    def changed(self, *_):
        if self.loading:
            return
        if self.worker is not None:
            self.worker.configure(self.parameters())
        if self.frozen is not None:
            self.render()

    def source_value(self):
        selected = self.source.currentData()
        if selected == "file":
            return str(Path(self.file_path.text()).expanduser().resolve())
        return int(selected)

    def source_signature(self):
        return (
            self.source_value(),
            self.backend.currentData(),
            self.resolution.currentText(),
            self.fps.value(),
        )

    def update_controls(self, *_):
        busy = self.worker is not None
        file_source = self.source.currentData() == "file"
        for widget in (self.source, self.resolution, self.fps):
            widget.setEnabled(not busy)
        self.backend.setEnabled(not busy and not file_source)
        self.file_path.setEnabled(not busy and file_source)
        self.browse_button.setEnabled(not busy and file_source)
        self.start_button.setEnabled(not busy and self.frozen is None)
        self.stop_button.setEnabled(busy)
        self.pause_button.setEnabled(
            busy and file_source and self.frozen is None
        )
        self.record_button.setEnabled(busy and self.frozen is None)
        self.hardware_button.setEnabled(busy and not file_source)

        # Gaya visual tombol rekam.
        if self.recording:
            self.record_button.setStyleSheet(
                "QPushButton { background-color: #c0392b; color: white; "
                "font-weight: bold; }"
            )
            self.record_button.setText("⏹ Berhenti Rekam")
        else:
            self.record_button.setStyleSheet("")
            self.record_button.setText("⏺ Rekam")

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih video", self.file_path.text(),
            "Video (*.webm *.mp4 *.avi *.mkv *.mov);;Semua file (*)",
        )
        if path:
            self.file_path.setText(path)

    def start(self):
        if self.worker is not None:
            return
        try:
            p = self.parameters()
            validate_parameters(p, self.top)
            source = self.source_value()
            if isinstance(source, str) and not Path(source).is_file():
                raise ValueError(f"File tidak ditemukan: {source}")

            for other in self.owner.panes:
                if (
                    other is not self
                    and other.worker is not None
                    and isinstance(source, int)
                    and other.active_source == source
                ):
                    raise ValueError(
                        "Indeks kamera sedang digunakan panel lain. "
                        "Pilih kamera berbeda."
                    )

            signature = self.source_signature()
            if signature != self.previous_source:
                self.reference = None
                self.previous_source = signature

            size = tuple(map(int, self.resolution.currentText().split("x")))
            self.packet = None
            self.shown = None
            self.view.image = None
            self.view.update()
            self.pause_button.setChecked(False)
            self.active_source = source
            self.worker = CaptureWorker(
                source, self.backend.currentData(), size,
                self.fps.value(), p, self.reference, self.top,
            )
            self.worker.start()
            self.message.setText("Membuka sumber...")
            self.update_controls()

        except Exception as exc:
            QMessageBox.warning(self, "Tidak dapat memulai", str(exc))

    def stop(self):
        if self.recording:
            self.stop_recording()
            self.record_button.setChecked(False)
        if self.frozen is not None:
            self.cancel_calibration()
        if self.worker is not None:
            self.worker.stop_event.set()
            self.message.setText("Menghentikan sumber...")

    def pause(self, checked):
        if self.worker is not None:
            if checked:
                self.worker.paused.set()
            else:
                self.worker.paused.clear()

    def poll(self):
        worker = self.worker
        if worker is None:
            return

        while True:
            try:
                kind, payload = worker.events.get_nowait()
            except queue.Empty:
                break
            if kind == "reference":
                self.reference = payload
            else:
                self.message.setText(payload)

        try:
            self.packet = worker.latest.get_nowait()
        except queue.Empty:
            pass
        else:
            self.render()

        if not worker.is_alive():
            if self.recording:
                self.stop_recording()
                self.record_button.setChecked(False)
            self.worker = None
            self.active_source = None
            self.pause_button.setChecked(False)
            self.update_controls()

    def render(self, *_):
        if self.frozen is not None:
            try:
                p = self.parameters()
                adjusted, _, _ = prepare_image(self.frozen, p)
                self.shown = draw_guides(adjusted, p)
                self.view.set_frame(self.shown)
            except (ValueError, cv2.error) as exc:
                self.message.setText(str(exc))
            return

        if self.packet is None:
            return

        packet = self.packet
        mode = self.view_mode.currentIndex()

        if mode == 1:
            out = packet["raw"]
        elif mode == 3:
            out = cv2.cvtColor(packet["mask"], cv2.COLOR_GRAY2BGR)
        elif mode == 4:
            out = cv2.cvtColor(packet["gray"], cv2.COLOR_GRAY2BGR)
        else:
            out = packet["adjusted"]

        if mode == 0 and self.top:
            if self.parameters()["guides"]:
                out = draw_guides(out, self.parameters())
            out = draw_detection(out, packet["result"])

        self.shown = out
        self.view.set_frame(out)

        # Tulis frame ke rekaman jika sedang merekam.
        if self.recording and self.recorder is not None:
            try:
                self.recorder.write(out)
            except Exception:
                pass

        height, width = packet["raw"].shape[:2]
        text = (
            f"{width}x{height} | frame {packet['index']} | "
            f"{packet['ms']:.1f} ms | {packet['note']}"
        )
        result = packet["result"]
        if result:
            cx, cy = result["center"]
            text += f" | X={cx:.2f}, Y={cy:.2f} px | match={result['score']:.3f}"
            if result["ecc"] is not None:
                text += f" | ECC={result['ecc']:.3f}"
        self.stats.setText(text)

    def calibrate(self):
        if self.frozen is None:
            if self.packet is None:
                QMessageBox.information(
                    self, "Kalibrasi", "Mulai sumber sampai gambar muncul."
                )
                return
            self.old_parameters = self.parameters()
            self.frozen = self.packet["raw"].copy()
            self.was_paused = self.pause_button.isChecked()

            if isinstance(self.active_source, str):
                self.pause_button.setChecked(True)

            self.view.selecting = True
            self.calibration_button.setText("Terapkan referensi")
            self.cancel_button.setEnabled(True)
            self.message.setText(
                "Drag oval, template, dan search pada frame beku. "
                "Sesuaikan sudut oval bila diperlukan."
            )
            self.update_controls()
            self.render()
            return

        try:
            p = self.parameters()
            validate_parameters(p, True)

            # Validasi bahwa template dan oval dapat dipakai sebelum diterapkan.
            trial = PinholeTracker(self.frozen)
            trial.prepare_reference(self.frozen.shape[:2], p)

            self.reference = self.frozen.copy()
            if self.worker is not None:
                self.worker.configure(p, self.reference)

            self.finish_calibration()
            self.message.setText("Referensi baru diterapkan.")

        except Exception as exc:
            QMessageBox.warning(self, "Kalibrasi belum valid", str(exc))

    def finish_calibration(self):
        self.frozen = None
        self.view.selecting = False
        self.view.drag_start = None
        self.view.drag_end = None
        self.calibration_button.setText("Bekukan untuk kalibrasi")
        self.cancel_button.setEnabled(False)
        self.pause_button.setChecked(self.was_paused)
        self.update_controls()
        self.render()

    def cancel_calibration(self):
        if self.frozen is not None:
            self.fill_parameters(self.old_parameters)
            self.finish_calibration()
            self.message.setText("Kalibrasi dibatalkan.")

    def select_region(self, x0, y0, x1, y1):
        if not self.top or self.frozen is None:
            return

        height, width = self.frozen.shape[:2]
        p = self.parameters()
        mode = self.selection_mode.currentText()

        if mode == "Area pencarian":
            p.update(
                rx0=x0 / width, ry0=y0 / height,
                rx1=x1 / width, ry1=y1 / height,
            )
        else:
            x0, x1 = x0 * 640 / width, x1 * 640 / width
            y0, y1 = y0 * 480 / height, y1 * 480 / height
            if mode == "Template":
                p.update(tx=x0, ty=y0, tw=x1 - x0, th=y1 - y0)
            else:
                p.update(
                    cx=(x0 + x1) / 2,
                    cy=(y0 + y1) / 2,
                    axis_a=x1 - x0,
                    axis_b=y1 - y0,
                    angle=0.0,
                )

        self.fill_parameters(p)

    def apply_hardware(self):
        if self.worker is not None:
            self.worker.commands.put((
                self.auto_exposure.currentData(),
                self.exposure.value(),
                self.gain.value(),
            ))

    def snapshot(self):
        if self.shown is None:
            return
        prefix = "top" if self.top else "side"
        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan snapshot",
            f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.png",
            "PNG (*.png)",
        )
        if path:
            try:
                if not path.lower().endswith(".png"):
                    path += ".png"
                if not cv2.imwrite(path, self.shown):
                    raise RuntimeError("Gagal menulis gambar.")
                self.message.setText(f"Snapshot tersimpan: {path}")
            except Exception as exc:
                QMessageBox.warning(self, "Snapshot gagal", str(exc))

    def toggle_recording(self, checked):
        if checked:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if self.recording:
            return
        if self.shown is None:
            self.record_button.setChecked(False)
            QMessageBox.information(
                self, "Rekam",
                "Mulai sumber sampai gambar muncul sebelum merekam.",
            )
            return

        prefix = "top" if self.top else "side"
        default_name = f"rekam_{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan rekaman video", default_name,
            "Video MP4 (*.mp4);;Video AVI (*.avi)",
        )
        if not path:
            self.record_button.setChecked(False)
            return

        if not (path.lower().endswith(".mp4") or path.lower().endswith(".avi")):
            path += ".mp4"

        try:
            height, width = self.shown.shape[:2]
            fps = self.fps.value()
            if path.lower().endswith(".avi"):
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.recorder = cv2.VideoWriter(path, fourcc, fps, (width, height))
            if not self.recorder.isOpened():
                raise RuntimeError("Gagal membuka VideoWriter.")

            self.recording = True
            self.record_path = path
            self.message.setText(f"⏺ Merekam ke: {path}")
            self.update_controls()

        except Exception as exc:
            self.recorder = None
            self.recording = False
            self.record_button.setChecked(False)
            QMessageBox.warning(self, "Rekam gagal", str(exc))

    def stop_recording(self):
        if not self.recording:
            return
        try:
            if self.recorder is not None:
                self.recorder.release()
        except Exception:
            pass
        self.recorder = None
        self.recording = False
        self.message.setText(f"Rekaman tersimpan: {self.record_path}")
        self.record_path = None
        self.update_controls()

    def profile(self):
        p = self.parameters()
        validate_parameters(p, self.top)
        reference = None
        if self.top and self.reference is not None:
            ok, encoded = cv2.imencode(".png", self.reference)
            if not ok:
                raise RuntimeError("Gagal mengodekan gambar referensi.")
            reference = base64.b64encode(encoded.tobytes()).decode("ascii")

        return {
            "source": self.source.currentData(),
            "file": self.file_path.text(),
            "backend": self.backend.currentData(),
            "resolution": self.resolution.currentText(),
            "fps": self.fps.value(),
            "parameters": p,
            "reference_png": reference,
        }

    def restore_profile(self, data, decoded_reference):
        self.source.setCurrentIndex(self.source.findData(data["source"]))
        self.file_path.setText(data["file"])
        self.backend.setCurrentIndex(self.backend.findData(data["backend"]))
        self.resolution.setCurrentText(data["resolution"])
        self.fps.setValue(data["fps"])
        self.fill_parameters(data["parameters"])
        self.reference = decoded_reference if self.top else None
        self.previous_source = self.source_signature()
        self.packet = None
        self.shown = None
        self.view.image = None
        self.view.update()
        self.message.setText("Profil dimuat. Klik Mulai.")
        self.update_controls()


def atomic_json_write(path, data):
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8",
            dir=str(destination.parent),
            prefix="pinhole_", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, allow_nan=False)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class MainWindow(QMainWindow):
    def __init__(self, top_file, side_file):
        super().__init__()
        self.setWindowTitle("Pinhole Monitor | TOP + SIDE")
        self.setMinimumSize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        title = QLabel("PINHOLE MONITOR — TOP / SIDE")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        toolbar.addWidget(title)
        toolbar.addStretch()

        save_button = QPushButton("Simpan profil")
        load_button = QPushButton("Muat profil")
        stop_button = QPushButton("Stop semua")
        save_button.clicked.connect(self.save_profile)
        load_button.clicked.connect(self.load_profile)
        stop_button.clicked.connect(self.stop_all)

        for button in (save_button, load_button, stop_button):
            toolbar.addWidget(button)
        layout.addLayout(toolbar)

        self.panes = []
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.panes = [
            CameraPane("TOP — pelacak oval", True, self, top_file),
            CameraPane("SIDE — preview", False, self, side_file),
        ]
        for pane in self.panes:
            splitter.addWidget(pane)
        splitter.setSizes([700, 700])
        layout.addWidget(splitter)

        self.statusBar().showMessage(
            "Koordinat dalam piksel. Kalibrasi ulang jika kamera, posisi, "
            "fokus, atau geometri pengambilan berubah."
        )

        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1500, screen.width()), min(980, screen.height()))

    def stop_all(self):
        for pane in self.panes:
            pane.stop()

    def save_profile(self):
        if any(pane.frozen is not None for pane in self.panes):
            QMessageBox.information(
                self, "Profil",
                "Terapkan atau batalkan kalibrasi sebelum menyimpan profil.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan profil", "pinhole_profile.json", "JSON (*.json)"
        )
        if not path:
            return

        try:
            if not path.lower().endswith(".json"):
                path += ".json"
            data = {
                "version": 1,
                "panes": [pane.profile() for pane in self.panes],
            }
            atomic_json_write(path, data)
            self.statusBar().showMessage(f"Profil tersimpan: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Simpan profil gagal", str(exc))

    def load_profile(self):
        if any(
            pane.worker is not None or pane.frozen is not None
            for pane in self.panes
        ):
            QMessageBox.information(
                self, "Profil",
                "Stop kedua sumber dan selesaikan kalibrasi sebelum memuat profil.",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Muat profil", "", "JSON (*.json)"
        )
        if not path:
            return

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("version") != 1 or len(data.get("panes", [])) != 2:
                raise ValueError("Format profil tidak didukung.")

            prepared = []
            for index, pane_data in enumerate(data["panes"]):
                p = DEFAULTS.copy()
                p.update({
                    key: value
                    for key, value in pane_data["parameters"].items()
                    if key in DEFAULTS
                })
                validate_parameters(p, top=(index == 0))
                pane_data["parameters"] = p

                source = pane_data["source"]
                if source != "file" and (
                    type(source) is not int or source not in range(10)
                ):
                    raise ValueError("Indeks sumber tidak valid.")

                if not isinstance(pane_data["file"], str):
                    raise ValueError("Path video tidak valid.")

                if pane_data["backend"] not in [value for _, value in BACKENDS]:
                    raise ValueError("Backend tidak didukung.")

                if pane_data["resolution"] not in (
                    "640x480", "800x600", "1280x720", "1920x1080"
                ):
                    raise ValueError("Resolusi tidak didukung.")

                fps = float(pane_data["fps"])
                if not math.isfinite(fps) or not 1 <= fps <= 120:
                    raise ValueError("FPS tidak valid.")

                reference = None
                encoded = pane_data.get("reference_png")
                if encoded:
                    raw = base64.b64decode(encoded, validate=True)
                    reference = cv2.imdecode(
                        np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR
                    )
                    if reference is None:
                        raise ValueError("Gambar referensi rusak.")

                prepared.append((pane_data, reference))

            # Semua data sudah divalidasi sebelum mengubah GUI.
            for pane, (pane_data, reference) in zip(self.panes, prepared):
                pane.restore_profile(pane_data, reference)

            self.statusBar().showMessage(f"Profil dimuat: {path}")

        except Exception as exc:
            QMessageBox.warning(self, "Muat profil gagal", str(exc))

    def closeEvent(self, event):
        workers = [
            pane.worker for pane in self.panes if pane.worker is not None
        ]
        for worker in workers:
            worker.stop_event.set()
        for worker in workers:
            worker.join(timeout=0.5)
        event.accept()


STYLE = """
QWidget {
    background: #142132;
    color: #e6edf5;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #364960;
    border-radius: 7px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
}
QPushButton {
    background: #263e56;
    border: 1px solid #405b75;
    border-radius: 5px;
    padding: 7px 10px;
}
QPushButton:hover { background: #31536f; }
QPushButton:checked { background: #146b70; }
QPushButton:disabled { color: #64758a; background: #1a293b; }
QLineEdit, QComboBox, QDoubleSpinBox {
    background: #0c1725;
    border: 1px solid #40536a;
    border-radius: 4px;
    padding: 4px;
}
QTabBar::tab {
    background: #203449;
    padding: 7px 10px;
}
QTabBar::tab:selected {
    background: #17636c;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #354b61;
}
QSlider::handle:horizontal {
    background: #50d0cb;
    width: 12px;
    margin: -4px 0;
    border-radius: 5px;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", default="top.webm", help="Video awal panel TOP")
    parser.add_argument("--side", default="", help="Video awal panel SIDE")
    args = parser.parse_args()

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    window = MainWindow(args.top, args.side)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
