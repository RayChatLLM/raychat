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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

from raychat.configuration import SETTINGS
from raychat.ui.caching import cache_function
from raychat.ui.state import Rect, display_clusters, display_width

RGB = tuple[int, int, int]
Vector = tuple[float, float, float]

BLACK: RGB = SETTINGS.renderer.black
WHITE: RGB = SETTINGS.renderer.white
RGB_QUANTIZATION_STEP = SETTINGS.renderer.rgb_quantization_step
SAMPLES_PER_CELL = SETTINGS.renderer.samples_per_cell
ANIMATION_HERTZ = SETTINGS.renderer.animation_hertz
DEFAULT_QUALITY = SETTINGS.renderer.default_quality
_SCENE = SETTINGS.renderer.scene
_BENCHMARK = SETTINGS.renderer.benchmark


def _positive_dimensions(*values: object) -> bool:
    return all(type(value) is int and value > 0 for value in values)


def _finite(value: object) -> bool:
    return (type(value) is int or type(value) is float) and math.isfinite(value)


@runtime_checkable
class _Sky(Protocol):
    def __call__(self, dx: float, dy: float, dz: float, /) -> Vector: ...


def _uncached_sky(dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    sky = _SCENE.sky
    fade = max(
        0.0,
        min(1.0, (dy + sky.fade_y_offset) * sky.fade_scale),
    )
    glow_vector = sky.glow_vector
    glow = math.pow(
        max(0.0, dx * glow_vector[0] + dy * glow_vector[1] + dz * glow_vector[2]),
        sky.glow_exponent,
    )
    stars = 0.0
    if dy > sky.star_y_threshold:
        noise_scale = sky.star_noise
        noise = math.sin(dx * noise_scale[0] + dz * noise_scale[1]) * math.sin(
            dy * noise_scale[2],
        )
        stars = sky.star_intensity if noise > sky.star_threshold else 0.0
    return (
        sky.base_color[0] + fade * sky.fade_color[0] + glow * sky.glow_color[0] + stars,
        sky.base_color[1] + fade * sky.fade_color[1] + glow * sky.glow_color[1] + stars,
        sky.base_color[2] + fade * sky.fade_color[2] + glow * sky.glow_color[2] + stars,
    )


def _bind_sky() -> _Sky:
    cached = cache_function(_uncached_sky, 8192)
    if not isinstance(cached, _Sky):
        message = "The sky cache must retain its typed callable contract."
        raise TypeError(message)
    return cached


_sky = _bind_sky()


def _rgb(value: tuple[float, float, float]) -> RGB:
    """Clamp and quantize a linear RGB triple for smaller ANSI frames.

    Returns
    -------
    RGB
        A bounded integer color with stable quantization.

    """
    result = []
    for channel in value:
        integer = max(0, min(255, int(channel)))
        result.append((integer // RGB_QUANTIZATION_STEP) * RGB_QUANTIZATION_STEP)
    return result[0], result[1], result[2]


def _safe_glyph(value: str) -> str:
    """Return one printable terminal glyph without terminal control bytes.

    Returns
    -------
    str
        A printable cluster, replacement character or space.

    """
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


@dataclass(frozen=True, slots=True)
class CellStyle:
    """Share foreground, background and emphasis across a paint operation."""

    foreground: RGB = WHITE
    background: RGB = BLACK
    bold: bool = False


DEFAULT_STYLE = CellStyle()
_WIDE_CELLS = 2
_MIN_BOX_SIZE = 2
_MIN_TITLE_WIDTH = 6


_Style = tuple[RGB, RGB, int]


@runtime_checkable
class _AnsiStyle(Protocol):
    def __call__(self, style: _Style, *, truecolor: bool) -> str: ...


def _palette(color: RGB) -> int:
    return (
        16 + 36 * round(color[0] / 51) + 6 * round(color[1] / 51) + round(color[2] / 51)
    )


def _uncached_ansi_style(style: _Style, *, truecolor: bool) -> str:
    fg, bg, strong = style
    if truecolor:
        return (
            f"\x1b[{1 if strong else 22};38;2;{fg[0]};{fg[1]};{fg[2]};"
            f"48;2;{bg[0]};{bg[1]};{bg[2]}m"
        )
    return f"\x1b[{1 if strong else 22};38;5;{_palette(fg)};48;5;{_palette(bg)}m"


def _bind_ansi_style() -> _AnsiStyle:
    cached = cache_function(_uncached_ansi_style, 4096)
    if not isinstance(cached, _AnsiStyle):
        message = "The ANSI cache must retain its typed callable contract."
        raise TypeError(message)
    return cached


_ansi_style = _bind_ansi_style()


def _dirty_runs(dirty: bytearray) -> Iterator[tuple[int, int]]:
    column = 0
    while column < len(dirty):
        if not dirty[column]:
            column += 1
            continue
        first = column
        while column < len(dirty) and dirty[column]:
            column += 1
        yield first, column


class Surface:
    """A fixed terminal-cell buffer with safe text and ANSI serialization."""

    __slots__ = ("background", "bold", "chars", "foreground", "height", "width")

    def __init__(self, width: int, height: int, background: RGB = BLACK) -> None:
        """Allocate a cell buffer with the requested dimensions.

        Raises
        ------
        ValueError
            When dimensions are not positive integers.

        """
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
        """Detach all cell and style arrays.

        Returns
        -------
        Surface
            A copy that can be painted without modifying this frame.

        """
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
        *,
        style: CellStyle = DEFAULT_STYLE,
    ) -> None:
        """Paint a clipped cell and preserve complete wide-character pairs."""
        foreground, background, bold = style.foreground, style.background, style.bold
        if 0 <= x < self.width and 0 <= y < self.height:
            index = y * self.width + x
            row_start = y * self.width
            row_end = row_start + self.width

            def unlink(cell: int) -> None:
                if not self.chars[cell] and cell > row_start:
                    self.chars[cell - 1] = " "
                elif (
                    cell + 1 < row_end
                    and not self.chars[cell + 1]
                    and display_width(self.chars[cell]) == _WIDE_CELLS
                ):
                    self.chars[cell + 1] = " "

            unlink(index)
            glyph = _safe_glyph(char)
            cell_width = display_width(glyph)
            if cell_width == _WIDE_CELLS and index + 1 >= row_end:
                glyph = "?"
                cell_width = 1
            if cell_width == _WIDE_CELLS:
                unlink(index + 1)
            self.chars[index] = glyph
            self.foreground[index] = foreground
            self.background[index] = background
            self.bold[index] = bool(bold)
            if cell_width == _WIDE_CELLS:
                # Empty string is a continuation sentinel. Serializers skip it
                # because the preceding glyph already occupies this cell.
                self.chars[index + 1] = ""
                self.foreground[index + 1] = foreground
                self.background[index + 1] = background
                self.bold[index + 1] = bool(bold)

    def fill_rect(
        self,
        rect: Rect,
        background: RGB,
        *,
        char: str = " ",
        foreground: RGB = WHITE,
        bold: bool = False,
    ) -> None:
        """Fill the clipped rectangle and unlink intersecting wide glyphs."""
        x, y, width, height = rect.x, rect.y, rect.width, rect.height
        left = max(0, x)
        top = max(0, y)
        right = min(self.width, x + max(0, width))
        bottom = min(self.height, y + max(0, height))
        glyph = _safe_glyph(char)
        if display_width(glyph) == _WIDE_CELLS:
            # A two-cell glyph cannot tile arbitrary one-cell rectangle edges.
            glyph = "?"
        style = int(bool(bold))
        for row in range(top, bottom):
            start = row * self.width + left
            end = row * self.width + right
            row_start = row * self.width
            row_end = row_start + self.width
            if start > row_start and not self.chars[start]:
                self.chars[start - 1] = " "
            if end < row_end and not self.chars[end]:
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
        *,
        style: CellStyle = DEFAULT_STYLE,
        max_width: int | None = None,
    ) -> int:
        """Paint printable text while respecting wide Unicode cells.

        Returns
        -------
        int
            The number of terminal columns consumed within the available width.

        """
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
                self.set(cursor, y, glyph, style=style)
            cursor += cell_width
            used += cell_width
        return used

    def box(
        self,
        rect: Rect,
        *,
        border: RGB,
        background: RGB,
        title: str = "",
        ascii_only: bool = False,
    ) -> None:
        """Paint a clipped border, opaque interior and optional safe title."""
        x, y, width, height = rect.x, rect.y, rect.width, rect.height
        if width < _MIN_BOX_SIZE or height < _MIN_BOX_SIZE:
            return
        self.fill_rect(rect, background)
        style = CellStyle(border, background)
        tl, tr, bl, br, horizontal, vertical = (
            ("+", "+", "+", "+", "-", "|")
            if ascii_only
            else ("╭", "╮", "╰", "╯", "─", "│")
        )
        for column in range(x + 1, x + width - 1):
            self.set(column, y, horizontal, style=style)
            self.set(column, y + height - 1, horizontal, style=style)
        for row in range(y + 1, y + height - 1):
            self.set(x, row, vertical, style=style)
            self.set(x + width - 1, row, vertical, style=style)
        self.set(x, y, tl, style=style)
        self.set(x + width - 1, y, tr, style=style)
        self.set(x, y + height - 1, bl, style=style)
        self.set(x + width - 1, y + height - 1, br, style=style)
        if title and width > _MIN_TITLE_WIDTH:
            label = " " + title + " "
            self.text(
                x + 2,
                y,
                label,
                style=CellStyle(border, background, bold=True),
                max_width=width - 4,
            )

    def to_plain(self) -> str:
        """Join printable cell rows without ANSI control sequences.

        Returns
        -------
        str
            The complete frame text, with newlines between rows.

        """
        return "\n".join(
            "".join(self.chars[row * self.width : (row + 1) * self.width])
            for row in range(self.height)
        )

    def _changed_spans(self, previous: Surface | None) -> Iterator[tuple[int, int]]:
        """Find row-contained cell runs, including both halves of wide glyphs.

        Yields
        ------
        tuple[int, int]
            Start and exclusive end indices of a changed cell run.

        """
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
                if column and (not self.chars[index] or not previous.chars[index]):
                    dirty[column - 1] = 1
                if index + 1 < end and (
                    not self.chars[index + 1] or not previous.chars[index + 1]
                ):
                    dirty[column + 1] = 1
            for first, column in _dirty_runs(dirty):
                yield start + first, start + column

    def to_ansi(
        self,
        *,
        home: bool = True,
        truecolor: bool = True,
        previous: Surface | None = None,
    ) -> str:
        """Serialize a full frame or the cells changed since the displayed frame.

        A missing or differently sized previous surface forces a full repaint.
        An unchanged frame emits nothing. Neither surface is mutated.

        Returns
        -------
        str
            Exact cursor, color and glyph sequences for the selected repaint.

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
                if not self.chars[index]:
                    continue
                style = (
                    self.foreground[index],
                    self.background[index],
                    self.bold[index],
                )
                if style != current:
                    output.append(_ansi_style(style, truecolor=truecolor))
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
        """Checksum glyphs and every cell style field in display order.

        Returns
        -------
        int
            A deterministic CRC32 of the complete surface.

        """
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
        """Encode a portable image with two vertical pixels per cell.

        Returns
        -------
        bytes
            The PPM header followed by foreground and background pixels.

        """
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
    """Describe an analytic sphere and its reflective material."""

    x: float
    y: float
    z: float
    radius: float
    color: RGB
    reflection: float


@dataclass(frozen=True, slots=True)
class _Hit:
    point: Vector
    normal: Vector
    base: RGB
    reflection: float
    distance: float


def _dot(left: Vector, right: Vector) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _point(origin: Vector, direction: Vector, distance: float) -> Vector:
    return (
        origin[0] + direction[0] * distance,
        origin[1] + direction[1] * distance,
        origin[2] + direction[2] * distance,
    )


class RayTracer:
    """Small analytic ray tracer optimized for terminal-sized frames."""

    def __init__(self) -> None:
        """Initialize the bounded cache of camera rays."""
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
    def scene(moment: float) -> tuple[Sphere, ...]:
        """Position configured spheres at the requested animation time.

        Returns
        -------
        tuple[Sphere, ...]
            The scene geometry in its configured order.

        """
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
    def sphere_hit(origin: Vector, direction: Vector, sphere: Sphere) -> float:
        """Find the first positive intersection of a ray and an analytic sphere.

        Returns
        -------
        float
            Ray distance, or infinity when the sphere is not in front of the ray.

        """
        rx, ry, rz = origin[0] - sphere.x, origin[1] - sphere.y, origin[2] - sphere.z
        projection = rx * direction[0] + ry * direction[1] + rz * direction[2]
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

    def _nearest(
        self,
        origin: Vector,
        direction: Vector,
        spheres: tuple[Sphere, ...],
    ) -> tuple[float, Sphere | None]:
        nearest = float("inf")
        hit: Sphere | None = None
        for sphere in spheres:
            distance = self.sphere_hit(origin, direction, sphere)
            if distance < nearest:
                nearest, hit = distance, sphere
        return nearest, hit

    def _intersection(
        self,
        direction: Vector,
        spheres: tuple[Sphere, ...],
    ) -> _Hit | None:
        nearest, sphere = self._nearest(_SCENE.camera, direction, spheres)
        plane = _SCENE.plane
        plane_distance = float("inf")
        if direction[1] < -_SCENE.plane_direction_epsilon:
            candidate = (plane.y - _SCENE.camera[1]) / direction[1]
            if candidate > _SCENE.hit_epsilon:
                plane_distance = candidate
        if plane_distance < nearest:
            point = (
                _SCENE.camera[0] + direction[0] * plane_distance,
                plane.y,
                _SCENE.camera[2] + direction[2] * plane_distance,
            )
            checker = (
                math.floor(point[0] * plane.checker_scale)
                + math.floor(point[2] * plane.checker_scale)
            ) & 1
            return _Hit(
                point,
                (0.0, 1.0, 0.0),
                plane.odd_color if checker else plane.even_color,
                plane.reflection,
                plane_distance,
            )
        if sphere is None:
            return None
        point = _point(_SCENE.camera, direction, nearest)
        inverse_radius = 1.0 / sphere.radius
        normal = (
            (point[0] - sphere.x) * inverse_radius,
            (point[1] - sphere.y) * inverse_radius,
            (point[2] - sphere.z) * inverse_radius,
        )
        return _Hit(point, normal, sphere.color, sphere.reflection, nearest)

    def _illuminate(
        self,
        hit: _Hit,
        direction: Vector,
        spheres: tuple[Sphere, ...],
    ) -> Vector:
        light = (
            _SCENE.light[0] - hit.point[0],
            _SCENE.light[1] - hit.point[1],
            _SCENE.light[2] - hit.point[2],
        )
        light_distance = math.sqrt(_dot(light, light))
        light = (
            light[0] / light_distance,
            light[1] / light_distance,
            light[2] / light_distance,
        )
        diffuse = max(0.0, _dot(hit.normal, light))
        shadow = 1.0
        origin = _point(hit.point, hit.normal, _SCENE.surface_bias)
        for sphere in spheres:
            if self.sphere_hit(origin, light, sphere) < light_distance:
                shadow = _SCENE.lighting.shadow
                break
        diffuse *= shadow
        half = light[0] - direction[0], light[1] - direction[1], light[2] - direction[2]
        inverse_half = 1.0 / max(_SCENE.normal_floor, math.sqrt(_dot(half, half)))
        specular = math.pow(
            max(0.0, _dot(hit.normal, half) * inverse_half),
            _SCENE.lighting.specular_exponent,
        )
        illumination = _SCENE.lighting.ambient + diffuse * _SCENE.lighting.diffuse
        return (
            hit.base[0] * illumination + specular * _SCENE.lighting.specular_color[0],
            hit.base[1] * illumination + specular * _SCENE.lighting.specular_color[1],
            hit.base[2] * illumination + specular * _SCENE.lighting.specular_color[2],
        )

    def _reflection(
        self,
        hit: _Hit,
        direction: Vector,
        spheres: tuple[Sphere, ...],
    ) -> Vector:
        dot = _dot(direction, hit.normal)
        reflected = (
            direction[0] - 2.0 * dot * hit.normal[0],
            direction[1] - 2.0 * dot * hit.normal[1],
            direction[2] - 2.0 * dot * hit.normal[2],
        )
        sky = _sky(*reflected)
        origin = _point(hit.point, hit.normal, _SCENE.surface_bias)
        _, secondary = self._nearest(origin, reflected, spheres)
        if secondary is None:
            return sky
        return (
            secondary.color[0] * _SCENE.lighting.secondary_reflection_scale,
            secondary.color[1] * _SCENE.lighting.secondary_reflection_scale,
            secondary.color[2] * _SCENE.lighting.secondary_reflection_scale,
        )

    def trace_ray(self, direction: Vector, spheres: tuple[Sphere, ...]) -> RGB:
        """Shade the first surface, including shadows, reflection and distance fog.

        Returns
        -------
        RGB
            The quantized terminal color for this ray.

        """
        hit = self._intersection(direction, spheres)
        if hit is None:
            return _rgb(_sky(*direction))
        direct = self._illuminate(hit, direction, spheres)
        reflected = self._reflection(hit, direction, spheres)
        lighting = _SCENE.lighting
        fog = max(
            0.0,
            min(
                lighting.fog_max,
                (hit.distance - lighting.fog_start) * lighting.fog_density,
            ),
        )
        sky = _sky(*direction)
        reflection = hit.reflection
        mixed = (
            direct[0] * (1.0 - reflection) + reflected[0] * reflection,
            direct[1] * (1.0 - reflection) + reflected[1] * reflection,
            direct[2] * (1.0 - reflection) + reflected[2] * reflection,
        )
        return _rgb((
            mixed[0] * (1.0 - fog) + sky[0] * fog,
            mixed[1] * (1.0 - fog) + sky[1] * fog,
            mixed[2] * (1.0 - fog) + sky[2] * fog,
        ))

    def render(
        self,
        width: int,
        height: int,
        moment: float = 0.0,
        *,
        quality: int = DEFAULT_QUALITY,
    ) -> Surface:
        """Trace an animated scene using a positive sampling stride.

        Returns
        -------
        Surface
            The complete sampled scene as terminal cells.

        Raises
        ------
        ValueError
            When dimensions, sampling stride or animation time are invalid.

        """
        if not _positive_dimensions(width, height, quality):
            message = "Dimensions and quality must be positive."
            raise ValueError(message)
        if not _finite(moment):
            message = "The animation moment must be finite."
            raise ValueError(message)
        pixel_height = height * SAMPLES_PER_CELL
        rays = self._rays(width, pixel_height, quality)
        spheres = self.scene(moment)
        surface = Surface(width, height)
        colors = [self.trace_ray(ray, spheres) for ray in rays]
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
    """Record renderer throughput, encoded size and checksum evidence."""

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
    """Measure tracing plus optional ANSI encoding without terminal I/O.

    Returns
    -------
    Benchmark
        Measured cadence, byte volume and deterministic frame checksums.

    Raises
    ------
    ValueError
        When dimensions, duration or sampling stride are invalid.

    """
    if (
        not _positive_dimensions(width, height, quality)
        or not _finite(seconds)
        or seconds <= 0
    ):
        message = "Benchmark dimensions, duration, and quality must be positive."
        raise ValueError(message)
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
    "DEFAULT_STYLE",
    "RGB",
    "WHITE",
    "Benchmark",
    "CellStyle",
    "RayTracer",
    "Sphere",
    "Surface",
    "benchmark",
]
