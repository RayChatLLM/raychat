"""Focused tests for the standard-library ray renderer and TUI compositor."""

from __future__ import annotations

import math
import re
import time
import unittest

from raychat.ui.controller import FrameComposition, compose_frame, quality_for_size
from raychat.ui.renderer import (
    Benchmark,
    CellStyle,
    RayTracer,
    Sphere,
    Surface,
    benchmark,
)
from raychat.ui.state import Rect, TuiState, display_width
from raychat.ui.terminal import LineEditor
from tests.assertions import TypedTestCase

_MAX_COLOR_CHANNEL = 255
_MIN_DISTINCT_RENDER_COLORS = 12
_MIN_REPLACED_CONTROLS = 5
_LARGE_WINDOW_QUALITY = 5
_MAX_CHECKSUM = 0xFFFFFFFF
_MIN_SMOKE_FPS = 10.0
_MAX_SMOKE_SECONDS = 3.0


def pixel_color(surface: Surface, x: int, pixel_y: int) -> tuple[int, int, int]:
    """Read one of the two rendered pixels represented by a terminal cell.

    Returns
    -------
    tuple[int, int, int]
        The exact RGB color stored for this half-block pixel.

    """
    index = (pixel_y // 2) * surface.width + x
    return surface.foreground[index] if pixel_y % 2 == 0 else surface.background[index]


class GeometryTests(TypedTestCase):
    """Check Geometry behavior and failure boundaries."""

    def test_sphere_hit_and_miss_are_analytic(self) -> None:
        """Check sphere hit and miss are analytic."""
        sphere = Sphere(0.0, 0.0, 5.0, 1.0, (200, 100, 50), 0.25)

        hit = RayTracer.sphere_hit((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), sphere)
        miss = RayTracer.sphere_hit((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), sphere)

        self.almost_equal(hit, 4.0)
        self.require(math.isinf(miss))

    def test_sphere_tangent_and_inside_origin_use_valid_positive_roots(self) -> None:
        """Check sphere tangent and inside origin use valid positive roots."""
        tangent = Sphere(1.0, 0.0, 5.0, 1.0, (1, 2, 3), 0.0)
        surrounding = Sphere(0.0, 0.0, 0.0, 1.0, (1, 2, 3), 0.0)

        self.almost_equal(
            RayTracer.sphere_hit((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), tangent),
            5.0,
        )
        self.almost_equal(
            RayTracer.sphere_hit((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), surrounding),
            1.0,
        )

    def test_reflection_changes_a_direct_sphere_hit(self) -> None:
        """Check reflection changes a direct sphere hit."""
        tracer = RayTracer()
        matte = Sphere(0.0, 0.18, 3.0, 1.0, (228, 42, 42), 0.0)
        mirror = Sphere(0.0, 0.18, 3.0, 1.0, (228, 42, 42), 1.0)
        distant = Sphere(100.0, 100.0, 100.0, 1.0, (0, 0, 0), 0.0)

        matte_color = tracer.trace_ray((0.0, 0.0, 1.0), (matte, distant, distant))
        mirror_color = tracer.trace_ray((0.0, 0.0, 1.0), (mirror, distant, distant))

        self.require((matte_color) != (mirror_color))
        self.require((matte_color[0]) > (mirror_color[0]))

    def test_animated_scene_has_valid_materials(self) -> None:
        """Check animated scene has valid materials."""
        first = RayTracer.scene(0.0)
        later = RayTracer.scene(1.0)

        self.require((first) != (later))
        self.equal(len(first), 3)
        for sphere in first:
            self.require((sphere.radius) > (0.0))
            self.require((sphere.reflection) >= (0.0))
            self.require((sphere.reflection) <= (1.0))
            self.require(
                all(0 <= channel <= _MAX_COLOR_CHANNEL for channel in sphere.color),
            )


class RayTracerTests(TypedTestCase):
    """Check RayTracer behavior and failure boundaries."""

    def test_render_is_deterministic_and_uses_half_blocks(self) -> None:
        """Check render is deterministic and uses half blocks."""
        tracer = RayTracer()

        first = tracer.render(24, 10, 0.5, quality=1)
        second = tracer.render(24, 10, 0.5, quality=1)

        self.equal(first.checksum(), second.checksum())
        self.equal(first.foreground, second.foreground)
        self.equal(first.background, second.background)
        self.equal(set(first.chars), {"▀"})
        self.require(
            (len(set(first.foreground + first.background)))
            > _MIN_DISTINCT_RENDER_COLORS,
        )

    def test_animation_changes_the_rendered_frame(self) -> None:
        """Check animation changes the rendered frame."""
        tracer = RayTracer()
        still = tracer.render(24, 10, 0.0, quality=1)
        animated = tracer.render(24, 10, 0.75, quality=1)

        self.require((still.checksum()) != (animated.checksum()))

    def test_quality_stride_reuses_each_sample_block(self) -> None:
        """Check quality stride reuses each sample block."""
        width, height, quality = 11, 6, 3
        surface = RayTracer().render(width, height, 1.25, quality=quality)

        for top in range(0, height * 2, quality):
            for left in range(0, width, quality):
                expected = pixel_color(surface, left, top)
                for y in range(top, min(top + quality, height * 2)):
                    for x in range(left, min(left + quality, width)):
                        self.equal(pixel_color(surface, x, y), expected)

    def test_renderer_reuses_itself_across_sizes(self) -> None:
        """Check renderer reuses itself across sizes."""
        tracer = RayTracer()
        small = tracer.render(7, 3, quality=2)
        large = tracer.render(13, 5, quality=2)
        small_again = tracer.render(7, 3, quality=2)

        self.equal((small.width, small.height), (7, 3))
        self.equal((large.width, large.height), (13, 5))
        self.equal(small.checksum(), small_again.checksum())

    def test_invalid_render_arguments_are_rejected(self) -> None:
        """Check invalid render arguments are rejected."""
        tracer = RayTracer()
        for width, height, quality in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1)):
            with (
                self.subTest(width=width, height=height, quality=quality),
                self.rejected(ValueError),
            ):
                tracer.render(width, height, quality=quality)
        for moment in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(moment=moment), self.rejected(ValueError):
                tracer.render(1, 1, moment)


class SurfaceTests(TypedTestCase):
    """Check Surface behavior and failure boundaries."""

    def test_dimensions_must_be_positive(self) -> None:
        """Check dimensions must be positive."""
        for width, height in ((0, 1), (1, 0), (-1, 2), (2, -1)):
            with self.subTest(width=width, height=height), self.rejected(ValueError):
                Surface(width, height)

    def test_set_and_fill_rect_clip_to_bounds(self) -> None:
        """Check set and fill rect clip to bounds."""
        surface = Surface(3, 2, (1, 2, 3))
        original = surface.checksum()
        surface.set(-1, 0, "X", style=CellStyle(background=(9, 9, 9)))
        surface.set(3, 1, "X", style=CellStyle(background=(9, 9, 9)))
        self.equal(surface.checksum(), original)

        surface.fill_rect(Rect(-1, -1, 2, 2), (7, 8, 9), char="Z")
        self.equal(surface.chars, ["Z", " ", " ", " ", " ", " "])
        self.equal(surface.background[0], (7, 8, 9))
        self.equal(surface.background[1:], [(1, 2, 3)] * 5)

    def test_text_replaces_terminal_controls_and_respects_wide_cells(self) -> None:
        """Check text replaces terminal controls and respects wide cells."""
        surface = Surface(30, 1)
        hostile = "\x1b[31mRED\x9b2J\n\t\u202e"

        used = surface.text(0, 0, hostile)
        plain = surface.to_plain()
        ansi = surface.to_ansi()

        self.equal(used, len(hostile))
        self.require(("\x1b") not in (plain))
        self.require(("\x9b") not in (plain))
        self.require(("\n") not in (plain))
        self.require(("\t") not in (plain))
        self.require(("\u202e") not in (plain))
        self.require(("\x1b[31m") not in (ansi))
        self.require(("\x9b2J") not in (ansi))
        self.require((plain.count("?")) >= _MIN_REPLACED_CONTROLS)

        wide = Surface(4, 1)
        self.equal(wide.text(0, 0, "界A", max_width=2), 2)
        self.equal(wide.chars, ["界", "", " ", " "])
        self.equal(display_width(wide.to_plain()), 4)
        payload = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", wide.to_ansi())
        self.equal(display_width(payload), 4)

        # Overwriting either half of a wide glyph cannot leave an orphaned
        # lead or continuation cell, and a glyph at the edge is replaced.
        wide.set(1, 0, "A")
        self.equal(wide.chars, [" ", "A", " ", " "])
        wide.set(3, 0, "界")
        self.equal(wide.chars[-1], "?")

    def test_box_is_clipped_and_has_safe_title(self) -> None:
        """Check box is clipped and has safe title."""
        surface = Surface(8, 4)
        surface.box(
            Rect(-1, 0, 8, 4),
            border=(1, 2, 3),
            background=(4, 5, 6),
            title="\x1bbad",
            ascii_only=True,
        )

        self.require(("\x1b") not in (surface.to_plain()))
        self.require(("?") in (surface.to_plain()))
        self.require(("|") in (surface.to_plain()))

    def test_copy_is_detached_and_checksum_tracks_all_layers(self) -> None:
        """Check copy is detached and checksum tracks all layers."""
        surface = Surface(2, 1, (4, 5, 6))
        clone = surface.copy()
        self.equal(surface.checksum(), clone.checksum())

        clone.set(
            0,
            0,
            "A",
            style=CellStyle(foreground=(7, 8, 9), background=(10, 11, 12), bold=True),
        )
        self.require((surface.checksum()) != (clone.checksum()))
        self.equal(surface.chars[0], " ")
        self.equal(surface.background[0], (4, 5, 6))

    def test_ansi_frame_uses_crlf_and_finishes_reset(self) -> None:
        """Check ansi frame uses crlf and finishes reset."""
        surface = Surface(2, 2, (3, 9, 15))
        surface.text(
            0,
            0,
            "A",
            style=CellStyle(foreground=(20, 30, 40), background=(3, 9, 15), bold=True),
        )

        truecolor = surface.to_ansi()
        palette = surface.to_ansi(home=False, truecolor=False)

        self.require(truecolor.startswith("\x1b[H"))
        self.require(truecolor.endswith("\x1b[0m"))
        self.equal(truecolor.count("\r\n"), 1)
        self.require(("\r") not in (truecolor.replace("\r\n", "")))
        self.require(("\n") not in (truecolor.replace("\r\n", "")))
        self.require(("38;2;") in (truecolor))
        self.require(not (palette.startswith("\x1b[H")))
        self.require(("38;5;") in (palette))
        self.require(("38;2;") not in (palette))
        self.require((";\x1b[") not in (palette))
        self.require(palette.endswith("\x1b[0m"))

    def test_ppm_contains_two_rgb_pixels_per_cell(self) -> None:
        """Check ppm contains two rgb pixels per cell."""
        surface = Surface(1, 1)
        surface.set(
            0,
            0,
            "▀",
            style=CellStyle(foreground=(1, 2, 3), background=(4, 5, 6)),
        )
        self.equal(surface.ppm(), b"P6\n1 2\n255\n\x01\x02\x03\x04\x05\x06")


class IntegrationAndBenchmarkTests(TypedTestCase):
    """Check IntegrationAndBenchmark behavior and failure boundaries."""

    def test_full_chat_frame_composites_without_mutating_cached_background(
        self,
    ) -> None:
        """Check full chat frame composites without mutating cached background."""
        tracer = RayTracer()
        state = TuiState()
        editor = LineEditor("hello \x1b[31mworld")
        background = tracer.render(100, 16, 0.25, quality=4)
        original_checksum = background.checksum()

        frame = compose_frame(
            tracer,
            state,
            editor,
            FrameComposition(
                width=100,
                height=16,
                moment=0.25,
                model="vendor/model-x",
                workspace="workspace",
                measured_fps=60.0,
                quality=4,
                ascii_only=True,
                background=background,
            ),
        )
        plain = frame.to_plain()

        self.equal(background.checksum(), original_checksum)
        self.require((frame.checksum()) != (original_checksum))
        self.require(("RAY CHAT") in (plain))
        self.require(("CHAT") in (plain))
        self.require(("MESSAGE") in (plain))
        self.require(("model-x") not in (plain))
        self.require(("\x1b") not in (plain))

    def test_size_policy_increases_sampling_stride(self) -> None:
        """Check size policy increases sampling stride."""
        self.equal(quality_for_size(25, 25), 1)
        self.equal(quality_for_size(100, 30), 4)
        self.require((quality_for_size(500, 100)) >= _LARGE_WINDOW_QUALITY)
        with self.rejected(ValueError):
            quality_for_size(0, 10)

    def test_benchmark_validates_arguments_and_reports_metadata(self) -> None:
        """Check benchmark validates arguments and reports metadata."""
        for kwargs in (
            {"width": 0},
            {"height": 0},
            {"seconds": 0.0},
            {"seconds": float("nan")},
            {"seconds": float("inf")},
            {"quality": 0},
        ):
            with self.subTest(kwargs=kwargs), self.rejected(ValueError):
                benchmark(**kwargs)

        result = benchmark(20, 6, 0.01, quality=3, include_ansi=False)
        self.require(isinstance(result, Benchmark))
        self.equal((result.width, result.height, result.quality), (20, 6, 3))
        self.require((result.frames) >= (1))
        self.require((result.seconds) > (0.0))
        self.require((result.fps) > (0.0))
        self.equal(result.ansi_bytes_per_frame, 0)
        self.require((result.checksum) >= (0))
        self.require((result.checksum) <= _MAX_CHECKSUM)

    def test_reference_size_full_pipeline_performance_smoke(self) -> None:
        """Check reference size full pipeline performance smoke."""
        started = time.perf_counter()
        result = benchmark(100, 30, 0.04, quality=3, include_ansi=True)
        wall_time = time.perf_counter() - started

        # Deliberately loose: this catches accidental quadratic regressions but
        # does not make shared or emulated CI machines prove the 60 FPS target.
        self.require((result.fps) > _MIN_SMOKE_FPS)
        self.require((result.ansi_bytes_per_frame) > (0))
        self.require((wall_time) < _MAX_SMOKE_SECONDS)


if __name__ == "__main__":
    unittest.main()
