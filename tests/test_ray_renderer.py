"""Focused tests for the standard-library ray renderer and TUI compositor."""

from __future__ import annotations

import math
import re
import time
import unittest

from raychat.ui.controller import compose_frame, quality_for_size
from raychat.ui.renderer import Benchmark, RayTracer, Sphere, Surface, benchmark
from raychat.ui.state import TuiState, display_width
from raychat.ui.terminal import LineEditor


def pixel_color(surface: Surface, x: int, pixel_y: int) -> tuple[int, int, int]:
    """Return one of the two ray-traced pixels represented by a terminal cell."""
    index = (pixel_y // 2) * surface.width + x
    return surface.foreground[index] if pixel_y % 2 == 0 else surface.background[index]


class GeometryTests(unittest.TestCase):
    def test_sphere_hit_and_miss_are_analytic(self) -> None:
        sphere = Sphere(0.0, 0.0, 5.0, 1.0, (200, 100, 50), 0.25)

        hit = RayTracer._sphere_hit(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, sphere)
        miss = RayTracer._sphere_hit(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, sphere)

        self.assertAlmostEqual(hit, 4.0)
        self.assertTrue(math.isinf(miss))

    def test_sphere_tangent_and_inside_origin_use_valid_positive_roots(self) -> None:
        tangent = Sphere(1.0, 0.0, 5.0, 1.0, (1, 2, 3), 0.0)
        surrounding = Sphere(0.0, 0.0, 0.0, 1.0, (1, 2, 3), 0.0)

        self.assertAlmostEqual(
            RayTracer._sphere_hit(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, tangent),
            5.0,
        )
        self.assertAlmostEqual(
            RayTracer._sphere_hit(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, surrounding),
            1.0,
        )

    def test_reflection_changes_a_direct_sphere_hit(self) -> None:
        tracer = RayTracer()
        matte = Sphere(0.0, 0.18, 3.0, 1.0, (228, 42, 42), 0.0)
        mirror = Sphere(0.0, 0.18, 3.0, 1.0, (228, 42, 42), 1.0)
        distant = Sphere(100.0, 100.0, 100.0, 1.0, (0, 0, 0), 0.0)

        matte_color = tracer._trace((0.0, 0.0, 1.0), (matte, distant, distant))
        mirror_color = tracer._trace((0.0, 0.0, 1.0), (mirror, distant, distant))

        self.assertNotEqual(matte_color, mirror_color)
        self.assertGreater(matte_color[0], mirror_color[0])

    def test_animated_scene_has_valid_materials(self) -> None:
        first = RayTracer._scene(0.0)
        later = RayTracer._scene(1.0)

        self.assertNotEqual(first, later)
        self.assertEqual(len(first), 3)
        for sphere in first:
            self.assertGreater(sphere.radius, 0.0)
            self.assertGreaterEqual(sphere.reflection, 0.0)
            self.assertLessEqual(sphere.reflection, 1.0)
            self.assertTrue(all(0 <= channel <= 255 for channel in sphere.color))


class RayTracerTests(unittest.TestCase):
    def test_render_is_deterministic_and_uses_half_blocks(self) -> None:
        tracer = RayTracer()

        first = tracer.render(24, 10, 0.5, quality=1)
        second = tracer.render(24, 10, 0.5, quality=1)

        self.assertEqual(first.checksum(), second.checksum())
        self.assertEqual(first.foreground, second.foreground)
        self.assertEqual(first.background, second.background)
        self.assertEqual(set(first.chars), {"▀"})
        self.assertGreater(len(set(first.foreground + first.background)), 12)

    def test_animation_changes_the_rendered_frame(self) -> None:
        tracer = RayTracer()
        still = tracer.render(24, 10, 0.0, quality=1)
        animated = tracer.render(24, 10, 0.75, quality=1)

        self.assertNotEqual(still.checksum(), animated.checksum())

    def test_quality_stride_reuses_each_sample_block(self) -> None:
        width, height, quality = 11, 6, 3
        surface = RayTracer().render(width, height, 1.25, quality=quality)

        for top in range(0, height * 2, quality):
            for left in range(0, width, quality):
                expected = pixel_color(surface, left, top)
                for y in range(top, min(top + quality, height * 2)):
                    for x in range(left, min(left + quality, width)):
                        self.assertEqual(pixel_color(surface, x, y), expected)

    def test_renderer_reuses_itself_across_sizes(self) -> None:
        tracer = RayTracer()
        small = tracer.render(7, 3, quality=2)
        large = tracer.render(13, 5, quality=2)
        small_again = tracer.render(7, 3, quality=2)

        self.assertEqual((small.width, small.height), (7, 3))
        self.assertEqual((large.width, large.height), (13, 5))
        self.assertEqual(small.checksum(), small_again.checksum())

    def test_invalid_render_arguments_are_rejected(self) -> None:
        tracer = RayTracer()
        for width, height, quality in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1)):
            with self.subTest(width=width, height=height, quality=quality):
                with self.assertRaises(ValueError):
                    tracer.render(width, height, quality=quality)
        for moment in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(moment=moment), self.assertRaises(ValueError):
                tracer.render(1, 1, moment)


class SurfaceTests(unittest.TestCase):
    def test_dimensions_must_be_positive(self) -> None:
        for width, height in ((0, 1), (1, 0), (-1, 2), (2, -1)):
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    Surface(width, height)

    def test_set_and_fill_rect_clip_to_bounds(self) -> None:
        surface = Surface(3, 2, (1, 2, 3))
        original = surface.checksum()
        surface.set(-1, 0, "X", background=(9, 9, 9))
        surface.set(3, 1, "X", background=(9, 9, 9))
        self.assertEqual(surface.checksum(), original)

        surface.fill_rect(-1, -1, 2, 2, (7, 8, 9), char="Z")
        self.assertEqual(surface.chars, ["Z", " ", " ", " ", " ", " "])
        self.assertEqual(surface.background[0], (7, 8, 9))
        self.assertEqual(surface.background[1:], [(1, 2, 3)] * 5)

    def test_text_replaces_terminal_controls_and_respects_wide_cells(self) -> None:
        surface = Surface(30, 1)
        hostile = "\x1b[31mRED\x9b2J\n\t\u202e"

        used = surface.text(0, 0, hostile)
        plain = surface.to_plain()
        ansi = surface.to_ansi()

        self.assertEqual(used, len(hostile))
        self.assertNotIn("\x1b", plain)
        self.assertNotIn("\x9b", plain)
        self.assertNotIn("\n", plain)
        self.assertNotIn("\t", plain)
        self.assertNotIn("\u202e", plain)
        self.assertNotIn("\x1b[31m", ansi)
        self.assertNotIn("\x9b2J", ansi)
        self.assertGreaterEqual(plain.count("?"), 5)

        wide = Surface(4, 1)
        self.assertEqual(wide.text(0, 0, "界A", max_width=2), 2)
        self.assertEqual(wide.chars, ["界", "", " ", " "])
        self.assertEqual(display_width(wide.to_plain()), 4)
        payload = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", wide.to_ansi())
        self.assertEqual(display_width(payload), 4)

        # Overwriting either half of a wide glyph cannot leave an orphaned
        # lead or continuation cell, and a glyph at the edge is replaced.
        wide.set(1, 0, "A")
        self.assertEqual(wide.chars, [" ", "A", " ", " "])
        wide.set(3, 0, "界")
        self.assertEqual(wide.chars[-1], "?")

    def test_box_is_clipped_and_has_safe_title(self) -> None:
        surface = Surface(8, 4)
        surface.box(
            -1,
            0,
            8,
            4,
            border=(1, 2, 3),
            background=(4, 5, 6),
            title="\x1bbad",
            ascii_only=True,
        )

        self.assertNotIn("\x1b", surface.to_plain())
        self.assertIn("?", surface.to_plain())
        self.assertIn("|", surface.to_plain())

    def test_copy_is_detached_and_checksum_tracks_all_layers(self) -> None:
        surface = Surface(2, 1, (4, 5, 6))
        clone = surface.copy()
        self.assertEqual(surface.checksum(), clone.checksum())

        clone.set(0, 0, "A", (7, 8, 9), (10, 11, 12), bold=True)
        self.assertNotEqual(surface.checksum(), clone.checksum())
        self.assertEqual(surface.chars[0], " ")
        self.assertEqual(surface.background[0], (4, 5, 6))

    def test_ansi_frame_uses_crlf_and_finishes_reset(self) -> None:
        surface = Surface(2, 2, (3, 9, 15))
        surface.text(0, 0, "A", (20, 30, 40), (3, 9, 15), bold=True)

        truecolor = surface.to_ansi()
        palette = surface.to_ansi(home=False, truecolor=False)

        self.assertTrue(truecolor.startswith("\x1b[H"))
        self.assertTrue(truecolor.endswith("\x1b[0m"))
        self.assertEqual(truecolor.count("\r\n"), 1)
        self.assertNotIn("\r", truecolor.replace("\r\n", ""))
        self.assertNotIn("\n", truecolor.replace("\r\n", ""))
        self.assertIn("38;2;", truecolor)
        self.assertFalse(palette.startswith("\x1b[H"))
        self.assertIn("38;5;", palette)
        self.assertNotIn("38;2;", palette)
        self.assertNotIn(";\x1b[", palette)
        self.assertTrue(palette.endswith("\x1b[0m"))

    def test_ppm_contains_two_rgb_pixels_per_cell(self) -> None:
        surface = Surface(1, 1)
        surface.set(0, 0, "▀", (1, 2, 3), (4, 5, 6))

        self.assertEqual(surface.ppm(), b"P6\n1 2\n255\n\x01\x02\x03\x04\x05\x06")


class IntegrationAndBenchmarkTests(unittest.TestCase):
    def test_full_chat_frame_composites_without_mutating_cached_background(
        self,
    ) -> None:
        tracer = RayTracer()
        state = TuiState()
        editor = LineEditor("hello \x1b[31mworld")
        background = tracer.render(100, 16, 0.25, quality=4)
        original_checksum = background.checksum()

        frame = compose_frame(
            tracer,
            state,
            editor,
            100,
            16,
            0.25,
            model="vendor/model-x",
            workspace="workspace",
            measured_fps=60.0,
            quality=4,
            ascii_only=True,
            background=background,
        )
        plain = frame.to_plain()

        self.assertEqual(background.checksum(), original_checksum)
        self.assertNotEqual(frame.checksum(), original_checksum)
        self.assertIn("RAY CHAT", plain)
        self.assertIn("CHAT", plain)
        self.assertIn("MESSAGE", plain)
        self.assertNotIn("model-x", plain)
        self.assertNotIn("\x1b", plain)

    def test_size_policy_increases_sampling_stride(self) -> None:
        self.assertEqual(quality_for_size(25, 25), 1)
        self.assertEqual(quality_for_size(100, 30), 4)
        self.assertGreaterEqual(quality_for_size(500, 100), 5)
        with self.assertRaises(ValueError):
            quality_for_size(0, 10)

    def test_benchmark_validates_arguments_and_reports_metadata(self) -> None:
        for kwargs in (
            {"width": 0},
            {"height": 0},
            {"seconds": 0.0},
            {"seconds": float("nan")},
            {"seconds": float("inf")},
            {"quality": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                benchmark(**kwargs)

        result = benchmark(20, 6, 0.01, quality=3, include_ansi=False)
        self.assertIsInstance(result, Benchmark)
        self.assertEqual((result.width, result.height, result.quality), (20, 6, 3))
        self.assertGreaterEqual(result.frames, 1)
        self.assertGreater(result.seconds, 0.0)
        self.assertGreater(result.fps, 0.0)
        self.assertEqual(result.ansi_bytes_per_frame, 0)
        self.assertGreaterEqual(result.checksum, 0)
        self.assertLessEqual(result.checksum, 0xFFFFFFFF)

    def test_reference_size_full_pipeline_performance_smoke(self) -> None:
        started = time.perf_counter()
        result = benchmark(100, 30, 0.04, quality=3, include_ansi=True)
        wall_time = time.perf_counter() - started

        # Deliberately loose: this catches accidental quadratic regressions but
        # does not make shared or emulated CI machines prove the 60 FPS target.
        self.assertGreater(result.fps, 10.0)
        self.assertGreater(result.ansi_bytes_per_frame, 0)
        self.assertLess(wall_time, 3.0)


if __name__ == "__main__":
    unittest.main()
