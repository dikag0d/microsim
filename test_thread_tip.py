"""Uji presisi pelacak ujung benang pada citra sintetis."""

import unittest

import cv2
import numpy as np

from topjarumbenang import BrownThreadTipTracker, draw


def metal_background(width=640, height=480):
    frame = np.full((height, width, 3), 210, np.uint8)
    # Alur logam horizontal (desaturasi, mirip video atas).
    frame[150:270, 180:] = (188, 186, 184)
    frame[270:340, 180:] = (245, 245, 245)
    frame[:, :170] = (230, 228, 226)
    return frame


def paint_thread(frame, x0, x1, y, thickness, bgr=(40, 70, 160)):
    """Gambar batang benang coklat/copper horizontal."""
    y0 = int(round(y - thickness / 2.0))
    y1 = int(round(y + thickness / 2.0))
    cv2.rectangle(frame, (int(x0), y0), (int(x1), y1), bgr, -1)
    # Ujung kiri membulat agar mirip cap silinder.
    r = max(2, int(round(thickness / 2.0)))
    cv2.circle(frame, (int(x0), int(round(y))), r, bgr, -1)
    return frame


class ThreadTipPrecisionTests(unittest.TestCase):
    def test_tip_at_left_end_centerline(self):
        frame = metal_background()
        x_end, y, thick = 360, 210, 22
        paint_thread(frame, x_end, 639, y, thick)

        tracker = BrownThreadTipTracker()
        result = tracker.detect(frame)
        self.assertIsNotNone(result)
        tx, ty = result["tip"]
        expected_x = x_end - thick / 2.0  # tepi kiri cap membulat
        self.assertLess(abs(tx - expected_x), 3.5, f"X ujung {tx:.2f} jauh dari {expected_x}")
        self.assertLess(abs(ty - y), 4.0, f"Y ujung {ty:.2f} jauh dari centerline {y}")
        self.assertGreater(result["local_thickness"], 12.0)
        self.assertLess(result["local_thickness"], 36.0)

    def test_dark_tip_is_not_clipped(self):
        """Ujung lebih gelap dari badan tetap diukur di kiri, bukan di tepi mask terang."""
        frame = metal_background()
        x_end, y, thick = 300, 208, 24
        paint_thread(frame, x_end + 28, 639, y, thick, bgr=(36, 78, 175))
        paint_thread(frame, x_end, x_end + 30, y, thick, bgr=(18, 32, 55))

        tracker = BrownThreadTipTracker()
        result = tracker.detect(frame)
        self.assertIsNotNone(result)
        tx, ty = result["tip"]
        expected_x = x_end - thick / 2.0
        self.assertLess(abs(tx - expected_x), 4.0, f"X {tx:.2f} seharusnya di cap gelap {expected_x:.1f}")
        self.assertLess(tx, x_end + 8.0, "Ujung tidak boleh mundur ke badan terang")
        self.assertLess(abs(ty - y), 5.0)

    def test_ignores_left_side_specks(self):
        frame = metal_background()
        paint_thread(frame, 400, 639, 215, 20)
        # Noda coklat di kiri, tidak menjulur ke kanan.
        cv2.circle(frame, (90, 200), 8, (30, 60, 140), -1)

        result = BrownThreadTipTracker().detect(frame)
        self.assertIsNotNone(result)
        self.assertGreater(result["tip"][0], 380)

    def test_no_false_positive_without_thread(self):
        frame = metal_background()
        result = BrownThreadTipTracker().detect(frame)
        self.assertIsNone(result)

    def test_tracks_leftward_motion(self):
        tracker = BrownThreadTipTracker()
        xs = [480, 430, 380, 330]
        tips = []
        for x in xs:
            frame = metal_background()
            paint_thread(frame, x, 639, 212, 20)
            result = tracker.detect(frame)
            self.assertIsNotNone(result)
            tips.append(result["tip"][0])
        for earlier, later in zip(tips, tips[1:]):
            self.assertLess(later, earlier - 20, "Ujung harus mengikuti gerak ke kiri")

    def test_draw_overlay_keeps_size(self):
        frame = metal_background()
        paint_thread(frame, 350, 639, 210, 22)
        result = BrownThreadTipTracker().detect(frame)
        out = draw(frame, None, result)
        self.assertEqual(out.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
