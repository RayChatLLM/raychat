#!/usr/bin/env python3
"""Fast, deterministic ray tracing and ANSI cell compositing.

The renderer is deliberately small and uses only the Python standard library.
It traces two scene samples per terminal cell and represents them with the
Unicode upper-half block. UI code can then paint opaque cells over the scene.
"""

from __future__ import annotations

import math
import time
import unicodedata
import zlib
from collections.abc import Iterator
from dataclasses import dataclass

from raychat.configuration import SETTINGS
from raychat.ui.state import display_clusters, display_width

RGB = tuple[int, int, int]

BLACK: RGB = SETTINGS.renderer.black
WHITE: RGB = SETTINGS.renderer.white
RGB_QUANTIZATION_STEP = SETTINGS.renderer.rgb_quantization_step
SAMPLES_PER_CELL = SETTINGS.renderer.samples_per_cell
ANIMATION_HERTZ = SETTINGS.renderer.animation_hertz
DEFAULT_QUALITY = SETTINGS.renderer.default_quality
_SCENE = SETTINGS.renderer.scene
_BENCHMARK = SETTINGS.renderer.benchmark


def _rgb(value: tuple[float, float, float]) -> RGB:
    """Clamp and lightly quantize a linear RGB triple for smaller ANSI frames."""
    result = []
    for channel in value:
        integer = max(0, min(255, int(channel)))
        result.append((integer // RGB_QUANTIZATION_STEP) * RGB_QUANTIZATION_STEP)
    return result[0], result[1], result[2]


def _safe_glyph(value: str) -> str:
    """Return one printable terminal glyph; never pass control bytes through."""
    if not value:
        return " "
    value = display_clusters(value)[0]
    category = unicodedata.category(value[0])
    if category in {"Cc", "Cf", "Cs"}:
        return "?"
    if category.startswith("M"):
        # Keep an isolated mark inside this cell instead of decorating an
        # unrelated cell painted before it. Stored transcript data is unchanged.
        return "◌" + value
    return value


class Surface:
    """A fixed terminal-cell buffer with safe text and ANSI serialization."""

    __slots__ = ("background", "bold", "chars", "foreground", "height", "width")

    def __init__(self, width: int, height: int, background: RGB = BLACK) -> None:
        if type(width) is not int or type(height) is not int or width < 1 or height < 1:
            error_message = "Surface dimensions must be positive."
            raise ValueError(error_message)
        self.width = width
        self.height = height
        size = width * height
        self.chars = [" "] * size
        self.foreground = [WHITE] * size
        self.background = [background] * size
        self.bold = bytearray(size)

    def copy(self) -> Surface:
        result = Surface(self.width, self.height)
        result.chars = self.chars.copy()
        result.foreground = self.foreground.copy()
        result.background = self.background.copy()
        result.bold = self.bold.copy()
        return result

    def set(
        self,
        x: int,
        y: int,
        char: str = " ",
        foreground: RGB = WHITE,
        background: RGB = BLACK,
        *,
        bold: bool = False,
    ) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            index = y * self.width + x
            row_start = y * self.width
            row_end = row_start + self.width

            def unlink(cell: int) -> None:
                if self.chars[cell] == "" and cell > row_start:
                    self.chars[cell - 1] = " "
                elif (
                    cell + 1 < row_end
                    and self.chars[cell + 1] == ""
                    and display_width(self.chars[cell]) == 2
                ):
                    self.chars[cell + 1] = " "

            unlink(index)
            glyph = _safe_glyph(char)
            cell_width = display_width(glyph)
            if cell_width == 2 and index + 1 >= row_end:
                glyph = "?"
                cell_width = 1
            if cell_width == 2:
                unlink(index + 1)
            self.chars[index] = glyph
            self.foreground[index] = foreground
            self.background[index] = background
            self.bold[index] = bool(bold)
            if cell_width == 2:
                # Empty string is a continuation sentinel. Serializers skip it
                # because the preceding glyph already occupies this cell.
                self.chars[index + 1] = ""
                self.foreground[index + 1] = foreground
                self.background[index + 1] = background
                self.bold[index + 1] = bool(bold)

    def fill_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        background: RGB,
        *,
        char: str = " ",
        foreground: RGB = WHITE,
        bold: bool = False,
    ) -> None:
        left = max(0, x)
        top = max(0, y)
        right = min(self.width, x + max(0, width))
        bottom = min(self.height, y + max(0, height))
        glyph = _safe_glyph(char)
        if display_width(glyph) == 2:
            # A two-cell glyph cannot tile arbitrary one-cell rectangle edges.
            glyph = "?"
        style = int(bool(bold))
        for row in range(top, bottom):
            start = row * self.width + left
            end = row * self.width + right
            row_start = row * self.width
            row_end = row_start + self.width
            if start > row_start and self.chars[start] == "":
                self.chars[start - 1] = " "
            if end < row_end and self.chars[end] == "":
                self.chars[end] = " "
            count = end - start
            self.chars[start:end] = [glyph] * count
            self.foreground[start:end] = [foreground] * count
            self.background[start:end] = [background] * count
            self.bold[start:end] = bytes([style]) * count

    def text(
        self,
        x: int,
        y: int,
        value: str,
        foreground: RGB = WHITE,
        background: RGB = BLACK,
        *,
        bold: bool = False,
        max_width: int | None = None,
    ) -> int:
        """Paint printable text, approximately respecting wide Unicode cells."""
        if not 0 <= y < self.height:
            return 0
        available = self.width - max(0, x)
        if max_width is not None:
            available = min(available, max(0, max_width))
        cursor = x
        used = 0
        for raw in display_clusters(value):
            glyph = _safe_glyph(raw)
            cell_width = display_width(glyph)
            if used + cell_width > available:
                break
            if cursor >= 0:
                self.set(cursor, y, glyph, foreground, background, bold=bold)
            cursor += cell_width
            used += cell_width
        return used

    def box(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        border: RGB,
        background: RGB,
        title: str = "",
        ascii_only: bool = False,
    ) -> None:
        if width < 2 or height < 2:
            return
        self.fill_rect(x, y, width, height, background)
        tl, tr, bl, br, horizontal, vertical = (
            ("+", "+", "+", "+", "-", "|")
            if ascii_only
            else ("╭", "╮", "╰", "╯", "─", "│")
        )
        for column in range(x + 1, x + width - 1):
            self.set(column, y, horizontal, border, background)
            self.set(column, y + height - 1, horizontal, border, background)
        for row in range(y + 1, y + height - 1):
            self.set(x, row, vertical, border, background)
            self.set(x + width - 1, row, vertical, border, background)
        self.set(x, y, tl, border, background)
        self.set(x + width - 1, y, tr, border, background)
        self.set(x, y + height - 1, bl, border, background)
        self.set(x + width - 1, y + height - 1, br, border, background)
        if title and width > 6:
            label = " " + title + " "
            self.text(
                x + 2,
                y,
                label,
                border,
                background,
                bold=True,
                max_width=width - 4,
            )

    def to_plain(self) -> str:
        return "\n".join(
            "".join(self.chars[row * self.width : (row + 1) * self.width])
            for row in range(self.height)
        )

    def _changed_spans(self, previous: Surface | None) -> Iterator[tuple[int, int]]:
        """Yield row-contained cell runs, including both halves of wide glyphs."""
        for row in range(self.height):
            start = row * self.width
            end = start + self.width
            if previous is None:
                yield start, end
                continue
            if (
                self.chars[start:end] == previous.chars[start:end]
                and self.foreground[start:end] == previous.foreground[start:end]
                and self.background[start:end] == previous.background[start:end]
                and self.bold[start:end] == previous.bold[start:end]
            ):
                continue
            dirty = bytearray(self.width)
            for index in range(start, end):
                if (
                    self.chars[index] == previous.chars[index]
                    and self.foreground[index] == previous.foreground[index]
                    and self.background[index] == previous.background[index]
                    and self.bold[index] == previous.bold[index]
                ):
                    continue
                column = index - start
                dirty[column] = 1
                if column and (self.chars[index] == "" or previous.chars[index] == ""):
                    dirty[column - 1] = 1
                if index + 1 < end and (
                    self.chars[index + 1] == "" or previous.chars[index + 1] == ""
                ):
                    dirty[column + 1] = 1
            column = 0
            while column < self.width:
                if not dirty[column]:
                    column += 1
                    continue
                first = column
                while column < self.width and dirty[column]:
                    column += 1
                yield start + first, start + column

    def to_ansi(
        self,
        *,
        home: bool = True,
        truecolor: bool = True,
        previous: Surface | None = None,
    ) -> str:
        """Serialize a full frame or only cells changed since the displayed frame.

        A missing or differently sized previous surface forces a full repaint.
        An unchanged frame emits nothing. Neither surface is mutated.
        """
        if previous is not None and (
            previous.width != self.width or previous.height != self.height
        ):
            previous = None
        output: list[str] = []
        current: tuple[RGB, RGB, int] | None = None
        for start, end in self._changed_spans(previous):
            if previous is not None:
                row, column = divmod(start, self.width)
                output.append(f"\x1b[{row + 1};{column + 1}H")
            for index in range(start, end):
                if self.chars[index] == "":
                    continue
                style = (
                    self.foreground[index],
                    self.background[index],
                    self.bold[index],
                )
                if style != current:
                    fg, bg, strong = style
                    if truecolor:
                        output.append(
                            f"\x1b[{1 if strong else 22};38;2;{fg[0]};{fg[1]};{fg[2]};"
                            f"48;2;{bg[0]};{bg[1]};{bg[2]}m",
                        )
                    else:
                        # Conservative 256-color mapping for older terminals.
                        def palette(color: RGB) -> int:
                            return (
                                16
                                + 36 * round(color[0] / 51)
                                + 6 * round(color[1] / 51)
                                + round(color[2] / 51)
                            )

                        output.append(
                            f"\x1b[{1 if strong else 22};38;5;{palette(fg)};"
                            f"48;5;{palette(bg)}m",
                        )
                    current = style
                output.append(self.chars[index])
            if previous is None and end < self.width * self.height:
                # Raw terminal modes commonly disable the output post-processing
                # that turns LF into CRLF, so move to column zero explicitly.
                output.append("\x1b[0m\r\n")
                current = None
        if not output:
            return ""
        if home:
            output.insert(0, "\x1b[H")
        output.append("\x1b[0m")
        return "".join(output)

    def checksum(self) -> int:
        payload = bytearray()
        for char, foreground, background, bold in zip(
            self.chars,
            self.foreground,
            self.background,
            self.bold,
            strict=True,
        ):
            payload.extend(char.encode("utf-8", errors="replace"))
            payload.append(0)
            payload.extend(foreground)
            payload.extend(background)
            payload.append(bold)
        return zlib.crc32(payload)

    def ppm(self) -> bytes:
        """Return a two-pixels-per-cell PPM preview (useful for visual tests)."""
        payload = bytearray(
            f"P6\n{self.width} {self.height * 2}\n255\n".encode("ascii"),
        )
        for y in range(self.height):
            offset = y * self.width
            for x in range(self.width):
                payload.extend(self.foreground[offset + x])
            for x in range(self.width):
                payload.extend(self.background[offset + x])
        return bytes(payload)


@dataclass(frozen=True)
class Sphere:
    x: float
    y: float
    z: float
    radius: float
    color: RGB
    reflection: float


class RayTracer:
    """Small analytic ray tracer optimized for terminal-sized frames."""

    def __init__(self) -> None:
        self._ray_cache: dict[
            tuple[int, int, int],
            list[tuple[float, float, float]],
        ] = {}

    def _rays(
        self,
        width: int,
        pixel_height: int,
        quality: int,
    ) -> list[tuple[float, float, float]]:
        key = (width, pixel_height, quality)
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached
        aspect = width / max(1.0, pixel_height * _SCENE.pixel_aspect)
        field_scale = _SCENE.field_scale
        rays: list[tuple[float, float, float]] = []
        for y in range(0, pixel_height, quality):
            vertical = (1.0 - 2.0 * ((y + 0.5) / pixel_height)) * field_scale
            for x in range(0, width, quality):
                horizontal = (2.0 * ((x + 0.5) / width) - 1.0) * aspect * field_scale
                inverse = 1.0 / math.sqrt(
                    horizontal * horizontal + vertical * vertical + 1.0,
                )
                rays.append((horizontal * inverse, vertical * inverse, inverse))
        if len(self._ray_cache) > _SCENE.ray_cache_entries:
            self._ray_cache.clear()
        self._ray_cache[key] = rays
        return rays

    @staticmethod
    def _scene(moment: float) -> tuple[Sphere, ...]:
        functions = {"sin": math.sin, "cos": math.cos}

        def coordinate(base: float, motion: tuple[str, float, float]) -> float:
            name, frequency, amplitude = motion
            return float(base) + functions[str(name)](
                moment * float(frequency),
            ) * float(amplitude)

        return tuple(
            Sphere(
                coordinate(item.position[0], item.x_motion),
                coordinate(item.position[1], item.y_motion),
                item.position[2],
                item.radius,
                item.color,
                item.reflection,
            )
            for item in _SCENE.spheres
        )

    @staticmethod
    def _sphere_hit(
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        sphere: Sphere,
    ) -> float:
        rx, ry, rz = ox - sphere.x, oy - sphere.y, oz - sphere.z
        projection = rx * dx + ry * dy + rz * dz
        discriminant = projection * projection - (
            rx * rx + ry * ry + rz * rz - sphere.radius * sphere.radius
        )
        if discriminant < 0.0:
            return float("inf")
        root = math.sqrt(discriminant)
        near = -projection - root
        if near > _SCENE.hit_epsilon:
            return near
        far = -projection + root
        return far if far > _SCENE.hit_epsilon else float("inf")

    @staticmethod
    def _sky(dx: float, dy: float, dz: float) -> tuple[float, float, float]:
        sky = _SCENE.sky
        fade = max(
            0.0,
            min(1.0, (dy + sky.fade_y_offset) * sky.fade_scale),
        )
        glow_vector = sky.glow_vector
        glow = (
            max(
                0.0,
                dx * glow_vector[0] + dy * glow_vector[1] + dz * glow_vector[2],
            )
            ** sky.glow_exponent
        )
        stars = 0.0
        if dy > sky.star_y_threshold:
            noise_scale = sky.star_noise
            noise = math.sin(dx * noise_scale[0] + dz * noise_scale[1]) * math.sin(
                dy * noise_scale[2],
            )
            stars = sky.star_intensity if noise > sky.star_threshold else 0.0
        return tuple(
            sky.base_color[index]
            + fade * sky.fade_color[index]
            + glow * sky.glow_color[index]
            + stars
            for index in range(3)
        )

    def _trace(
        self,
        direction: tuple[float, float, float],
        spheres: tuple[Sphere, ...],
    ) -> RGB:
        ox, oy, oz = _SCENE.camera
        plane = _SCENE.plane
        lighting = _SCENE.lighting
        dx, dy, dz = direction
        nearest = float("inf")
        hit_sphere: Sphere | None = None
        for sphere in spheres:
            distance = self._sphere_hit(ox, oy, oz, dx, dy, dz, sphere)
            if distance < nearest:
                nearest, hit_sphere = distance, sphere

        plane_distance = float("inf")
        if dy < -_SCENE.plane_direction_epsilon:
            candidate = (plane.y - oy) / dy
            if candidate > _SCENE.hit_epsilon:
                plane_distance = candidate

        if hit_sphere is None and plane_distance == float("inf"):
            return _rgb(self._sky(dx, dy, dz))

        if plane_distance < nearest:
            nearest = plane_distance
            px, py, pz = ox + dx * nearest, plane.y, oz + dz * nearest
            nx, ny, nz = 0.0, 1.0, 0.0
            checker_scale = plane.checker_scale
            checker = (
                math.floor(px * checker_scale) + math.floor(pz * checker_scale)
            ) & 1
            base = plane.odd_color if checker else plane.even_color
            reflection = plane.reflection
        else:
            assert hit_sphere is not None
            px, py, pz = (
                ox + dx * nearest,
                oy + dy * nearest,
                oz + dz * nearest,
            )
            inverse_radius = 1.0 / hit_sphere.radius
            nx = (px - hit_sphere.x) * inverse_radius
            ny = (py - hit_sphere.y) * inverse_radius
            nz = (pz - hit_sphere.z) * inverse_radius
            base = hit_sphere.color
            reflection = hit_sphere.reflection

        light = _SCENE.light
        lx, ly, lz = light[0] - px, light[1] - py, light[2] - pz
        light_distance = math.sqrt(lx * lx + ly * ly + lz * lz)
        lx, ly, lz = lx / light_distance, ly / light_distance, lz / light_distance
        diffuse = max(0.0, nx * lx + ny * ly + nz * lz)

        shadow = 1.0
        bias = _SCENE.surface_bias
        sx, sy, sz = px + nx * bias, py + ny * bias, pz + nz * bias
        for sphere in spheres:
            blocker = self._sphere_hit(sx, sy, sz, lx, ly, lz, sphere)
            if blocker < light_distance:
                shadow = lighting.shadow
                break
        diffuse *= shadow

        hx, hy, hz = lx - dx, ly - dy, lz - dz
        inverse_half = 1.0 / max(
            _SCENE.normal_floor,
            math.sqrt(hx * hx + hy * hy + hz * hz),
        )
        specular = (
            max(0.0, (nx * hx + ny * hy + nz * hz) * inverse_half)
            ** lighting.specular_exponent
        )
        illumination = lighting.ambient + diffuse * lighting.diffuse
        direct = tuple(
            base[index] * illumination + specular * lighting.specular_color[index]
            for index in range(3)
        )

        dot = dx * nx + dy * ny + dz * nz
        rdx, rdy, rdz = dx - 2.0 * dot * nx, dy - 2.0 * dot * ny, dz - 2.0 * dot * nz
        reflected = self._sky(rdx, rdy, rdz)
        # A one-bounce reflection sees the other analytic spheres.
        secondary_nearest = float("inf")
        secondary: Sphere | None = None
        for sphere in spheres:
            distance = self._sphere_hit(sx, sy, sz, rdx, rdy, rdz, sphere)
            if distance < secondary_nearest:
                secondary_nearest, secondary = distance, sphere
        if secondary is not None:
            reflected = (
                secondary.color[0] * lighting.secondary_reflection_scale,
                secondary.color[1] * lighting.secondary_reflection_scale,
                secondary.color[2] * lighting.secondary_reflection_scale,
            )

        fog = max(
            0.0,
            min(
                lighting.fog_max,
                (nearest - lighting.fog_start) * lighting.fog_density,
            ),
        )
        sky = self._sky(dx, dy, dz)
        mixed = (
            direct[0] * (1.0 - reflection) + reflected[0] * reflection,
            direct[1] * (1.0 - reflection) + reflected[1] * reflection,
            direct[2] * (1.0 - reflection) + reflected[2] * reflection,
        )
        return _rgb(
            (
                mixed[0] * (1.0 - fog) + sky[0] * fog,
                mixed[1] * (1.0 - fog) + sky[1] * fog,
                mixed[2] * (1.0 - fog) + sky[2] * fog,
            ),
        )

    def render(
        self,
        width: int,
        height: int,
        moment: float = 0.0,
        *,
        quality: int = DEFAULT_QUALITY,
    ) -> Surface:
        """Trace an animated scene; quality is a positive sampling stride."""
        if (
            type(width) is not int
            or type(height) is not int
            or type(quality) is not int
            or width < 1
            or height < 1
            or quality < 1
        ):
            error_message = "Dimensions and quality must be positive."
            raise ValueError(error_message)
        if (
            not isinstance(moment, (int, float))
            or isinstance(moment, bool)
            or not math.isfinite(moment)
        ):
            error_message = "The animation moment must be finite."
            raise ValueError(error_message)
        pixel_height = height * SAMPLES_PER_CELL
        rays = self._rays(width, pixel_height, quality)
        spheres = self._scene(moment)
        surface = Surface(width, height)
        colors = [self._trace(ray, spheres) for ray in rays]
        sample_columns = (width + quality - 1) // quality

        def sample(x: int, y: int) -> RGB:
            return colors[(y // quality) * sample_columns + x // quality]

        for y in range(height):
            row = y * width
            upper_y = y * SAMPLES_PER_CELL
            lower_y = upper_y + 1
            for x in range(width):
                index = row + x
                surface.chars[index] = "▀"
                surface.foreground[index] = sample(x, upper_y)
                surface.background[index] = sample(x, lower_y)
        return surface


@dataclass(frozen=True)
class Benchmark:
    width: int
    height: int
    quality: int
    frames: int
    seconds: float
    fps: float
    milliseconds_per_frame: float
    checksum: int
    ansi_bytes_per_frame: int


def benchmark(
    width: int = _BENCHMARK.width,
    height: int = _BENCHMARK.height,
    seconds: float = _BENCHMARK.seconds,
    *,
    quality: int = _BENCHMARK.quality,
    include_ansi: bool = _BENCHMARK.include_ansi,
) -> Benchmark:
    """Measure tracing plus optional ANSI encoding without terminal I/O."""
    if (
        type(width) is not int
        or type(height) is not int
        or type(quality) is not int
        or not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or width < 1
        or height < 1
        or quality < 1
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        error_message = "Benchmark dimensions, duration, and quality must be positive."
        raise ValueError(
            error_message,
        )
    tracer = RayTracer()
    warm = tracer.render(width, height, 0.0, quality=quality)
    if include_ansi:
        warm.to_ansi()
    frames = 0
    checksum = 0
    ansi_bytes = 0
    start = time.perf_counter()
    now = start
    while now - start < seconds or frames == 0:
        surface = tracer.render(
            width,
            height,
            frames / ANIMATION_HERTZ,
            quality=quality,
        )
        if include_ansi:
            ansi = surface.to_ansi()
            ansi_bytes += len(ansi.encode("utf-8"))
        checksum ^= surface.checksum()
        frames += 1
        now = time.perf_counter()
    elapsed = max(now - start, 1e-12)
    fps = frames / elapsed
    return Benchmark(
        width=width,
        height=height,
        quality=quality,
        frames=frames,
        seconds=elapsed,
        fps=fps,
        milliseconds_per_frame=elapsed * 1000.0 / frames,
        checksum=checksum,
        ansi_bytes_per_frame=(ansi_bytes // frames if include_ansi else 0),
    )


__all__ = [
    "BLACK",
    "RGB",
    "WHITE",
    "Benchmark",
    "RayTracer",
    "Sphere",
    "Surface",
    "benchmark",
]
