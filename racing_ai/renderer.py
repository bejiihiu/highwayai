from __future__ import annotations

import math

import pyglet
from pyglet import shapes
from pyglet.math import Mat4, Vec3
from pyglet.window import key, mouse

from racing_ai.math2d import angle_to_vector
from racing_ai.world import RacingWorld


WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 760
CAMERA_EDGE_SCROLL_MARGIN = 28
CAMERA_SCROLL_SPEED = 920.0
CAMERA_FOLLOW_SPEED = 6.5
CAMERA_BOUNDS_PADDING = 240.0
CAMERA_ZOOM_STEP = 1.12
CAMERA_MIN_ZOOM = 0.45
CAMERA_MAX_ZOOM = 2.65


class RacingWindow(pyglet.window.Window):
    def __init__(self, world: RacingWorld, visible: bool = True) -> None:
        super().__init__(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            "Pyglet Racing AI Sandbox",
            resizable=False,
            visible=visible,
        )
        self.world = world
        self.background_color = (18, 20, 22)
        self.fps_display = pyglet.window.FPSDisplay(self)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.mouse_position = (WINDOW_WIDTH * 0.5, WINDOW_HEIGHT * 0.5)
        self.camera_offset = (0.0, 0.0)
        self.camera_position = self.world.car.position
        self.camera_zoom = 1.0
        self.static_batch = pyglet.graphics.Batch()
        self.static_shapes: list[object] = []
        self._build_static_track()
        pyglet.clock.schedule_interval(self.update, 1.0 / 60.0)

    def update(self, dt: float) -> None:
        self.world.update(dt)
        self._update_camera(dt)

    def on_draw(self) -> None:
        pyglet.gl.glClearColor(
            self.background_color[0] / 255.0,
            self.background_color[1] / 255.0,
            self.background_color[2] / 255.0,
            1.0,
        )
        self.clear()
        self.view = self._camera_view()
        self.static_batch.draw()
        self._draw_markers()
        self._draw_rays()
        self._draw_car()
        self.view = Mat4()
        self._draw_overlay()
        self.fps_display.draw()

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self.mouse_position = (float(x), float(y))

    def on_mouse_drag(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        buttons: int,
        modifiers: int,
    ) -> None:
        self.mouse_position = (float(x), float(y))
        if buttons & mouse.MIDDLE:
            self.camera_offset = (
                self.camera_offset[0] - dx / self.camera_zoom,
                self.camera_offset[1] - dy / self.camera_zoom,
            )

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        self.mouse_position = (float(x), float(y))
        if abs(scroll_y) <= 1e-9:
            return

        old_zoom = self.camera_zoom
        zoom_direction = 1.0 if scroll_y > 0.0 else -1.0
        next_zoom = old_zoom * (CAMERA_ZOOM_STEP**zoom_direction)
        self.camera_zoom = self._clamp_axis(next_zoom, CAMERA_MIN_ZOOM, CAMERA_MAX_ZOOM)

        cursor_world = self._screen_to_world(float(x), float(y), old_zoom)
        cursor_offset = (
            (float(x) - self.width * 0.5) / self.camera_zoom,
            (float(y) - self.height * 0.5) / self.camera_zoom,
        )
        next_center = (
            cursor_world[0] - cursor_offset[0],
            cursor_world[1] - cursor_offset[1],
        )
        self.camera_position = self._clamp_camera_center(next_center)
        car = self.world.car
        self.camera_offset = (self.camera_position[0] - car.x, self.camera_position[1] - car.y)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == key.SPACE:
            self.camera_offset = (0.0, 0.0)

    def capture_frame_image(self) -> pyglet.image.ImageData:
        """Return the current color buffer for agents that want pixel observations."""
        return pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()

    def _build_static_track(self) -> None:
        shoulder_color = (12, 44, 67)
        road_color = (62, 175, 245)
        border_color = (246, 250, 255)
        curb_red = (174, 48, 45)
        curb_white = (236, 238, 238)
        centerline_color = (225, 201, 93)

        for start, end in self.world.track.center_segments:
            self.static_shapes.append(
                shapes.Line(
                    *start,
                    *end,
                    thickness=self.world.track.width + 42,
                    color=shoulder_color,
                    batch=self.static_batch,
                )
            )

        for start, end in self.world.track.center_segments:
            self.static_shapes.append(
                shapes.Line(*start, *end, thickness=self.world.track.width, color=road_color, batch=self.static_batch)
            )

        for segments in (self.world.track.inner_segments, self.world.track.outer_segments):
            for index, (start, end) in enumerate(segments):
                curb_color = curb_red if (index // 3) % 2 == 0 else curb_white
                self.static_shapes.append(
                    shapes.Line(*start, *end, thickness=11, color=curb_color, batch=self.static_batch)
                )
                self.static_shapes.append(
                    shapes.Line(*start, *end, thickness=4, color=border_color, batch=self.static_batch)
                )

        for index, (start, end) in enumerate(self.world.track.center_segments):
            if index % 8 in (0, 1, 2):
                self.static_shapes.append(
                    shapes.Line(*start, *end, thickness=3, color=centerline_color, batch=self.static_batch)
                )

        start_a, start_b = self.world.track.start_line()
        self.static_shapes.append(
            shapes.Line(*start_a, *start_b, thickness=8, color=(245, 245, 245), batch=self.static_batch)
        )
        self.static_shapes.append(
            shapes.Line(
                start_a[0],
                start_a[1],
                start_b[0],
                start_b[1],
                thickness=3,
                color=(35, 35, 35),
                batch=self.static_batch,
            )
        )

    def _update_camera(self, dt: float) -> None:
        self._update_camera_controls(dt)
        car = self.world.car
        target = (car.x + self.camera_offset[0], car.y + self.camera_offset[1])
        target = self._clamp_camera_center(target)
        blend = min(1.0, 1.0 - math.exp(-CAMERA_FOLLOW_SPEED * dt))
        self.camera_position = (
            self.camera_position[0] + (target[0] - self.camera_position[0]) * blend,
            self.camera_position[1] + (target[1] - self.camera_position[1]) * blend,
        )

    def _update_camera_controls(self, dt: float) -> None:
        dx = 0.0
        dy = 0.0
        mouse_x, mouse_y = self.mouse_position

        if self.keys[key.A] or self.keys[key.LEFT] or mouse_x <= CAMERA_EDGE_SCROLL_MARGIN:
            dx -= 1.0
        if self.keys[key.D] or self.keys[key.RIGHT] or mouse_x >= self.width - CAMERA_EDGE_SCROLL_MARGIN:
            dx += 1.0
        if self.keys[key.S] or self.keys[key.DOWN] or mouse_y <= CAMERA_EDGE_SCROLL_MARGIN:
            dy -= 1.0
        if self.keys[key.W] or self.keys[key.UP] or mouse_y >= self.height - CAMERA_EDGE_SCROLL_MARGIN:
            dy += 1.0

        size = math.hypot(dx, dy)
        if size <= 1e-9:
            return

        step = CAMERA_SCROLL_SPEED * dt / self.camera_zoom
        self.camera_offset = (
            self.camera_offset[0] + dx / size * step,
            self.camera_offset[1] + dy / size * step,
        )

    def _camera_view(self) -> Mat4:
        return (
            Mat4.from_translation(Vec3(self.width * 0.5, self.height * 0.5, 0.0))
            @ Mat4.from_scale(Vec3(self.camera_zoom, self.camera_zoom, 1.0))
            @ Mat4.from_translation(Vec3(-self.camera_position[0], -self.camera_position[1], 0.0))
        )

    def _clamp_camera_center(self, target: tuple[float, float]) -> tuple[float, float]:
        min_x, min_y, max_x, max_y = self.world.track.bounds
        left = min_x - CAMERA_BOUNDS_PADDING
        right = max_x + CAMERA_BOUNDS_PADDING
        bottom = min_y - CAMERA_BOUNDS_PADDING
        top = max_y + CAMERA_BOUNDS_PADDING
        half_visible_width = self.width * 0.5 / self.camera_zoom
        half_visible_height = self.height * 0.5 / self.camera_zoom
        min_center_x = left + half_visible_width
        max_center_x = right - half_visible_width
        min_center_y = bottom + half_visible_height
        max_center_y = top - half_visible_height
        return (
            self._clamp_axis(target[0], min_center_x, max_center_x),
            self._clamp_axis(target[1], min_center_y, max_center_y),
        )

    def _screen_to_world(self, x: float, y: float, zoom: float) -> tuple[float, float]:
        return (
            self.camera_position[0] + (x - self.width * 0.5) / zoom,
            self.camera_position[1] + (y - self.height * 0.5) / zoom,
        )

    @staticmethod
    def _clamp_axis(value: float, low: float, high: float) -> float:
        if low > high:
            return (low + high) * 0.5
        return max(low, min(high, value))

    def _draw_markers(self) -> None:
        color_by_kind = {
            "apex": (83, 178, 121),
            "speed": (68, 150, 220),
            "drift": (222, 156, 65),
        }
        for marker in self.world.markers:
            if marker.collected:
                continue
            color = color_by_kind.get(marker.kind, (210, 210, 210))
            halo = shapes.Circle(*marker.position, marker.radius + 8.0, color=(38, 42, 44))
            body = shapes.Circle(*marker.position, marker.radius, color=color)
            ring = shapes.Arc(*marker.position, marker.radius + 4.0, color=(240, 243, 246), segments=32)
            halo.opacity = 155
            body.opacity = 230
            halo.draw()
            body.draw()
            ring.draw()

    def _draw_rays(self) -> None:
        origin = self.world.car.position
        for ray in self.world.observation["rays"]:
            hit = ray["hit"]
            normalized = float(ray["normalized_distance"])
            color = self._ray_color(normalized)
            line = shapes.Line(*origin, *hit, thickness=2, color=color)
            dot = shapes.Circle(hit[0], hit[1], 4.0, color=color)
            line.opacity = 150
            line.draw()
            dot.draw()

    def _draw_car(self) -> None:
        car = self.world.car
        body = shapes.Rectangle(car.x, car.y, car.length, car.width, color=(230, 235, 240))
        body.anchor_position = (car.length * 0.5, car.width * 0.5)
        body.rotation = math.degrees(car.heading)
        body.draw()

        front = angle_to_vector(car.heading)
        nose = (car.x + front[0] * car.length * 0.62, car.y + front[1] * car.length * 0.62)
        heading_line = shapes.Line(car.x, car.y, nose[0], nose[1], thickness=4, color=(55, 134, 225))
        heading_line.draw()

        vx, vy = car.velocity
        speed = math.hypot(vx, vy)
        if speed > 1.0:
            scale = min(0.45, 60.0 / max(speed, 1.0))
            velocity_line = shapes.Line(
                car.x,
                car.y,
                car.x + vx * scale,
                car.y + vy * scale,
                thickness=3,
                color=(237, 94, 78),
            )
            velocity_line.draw()

        if bool(self.world.observation["off_track"]):
            warning = shapes.Circle(car.x, car.y, car.length * 0.85, color=(215, 58, 58))
            warning.opacity = 80
            warning.draw()

    def _draw_overlay(self) -> None:
        obs = self.world.observation
        lines = [
            "AI sandbox: no keyboard control",
            f"Reward {float(obs['total_reward']):7.2f}   Frame {float(obs['frame_reward']):6.2f}",
            f"Speed {float(obs['speed']):6.1f}   Drift {float(obs['drift_score']):6.2f}",
            f"Markers {int(obs['markers_collected'])}/{int(obs['markers_total'])}   Off-track {int(obs['off_track_count'])}",
            f"Lap {int(obs['lap'])}   Progress {float(obs['progress']) * 100.0:5.1f}%",
            f"Ray inputs {len(obs['rays'])}   Heading error {math.degrees(float(obs['heading_error'])):6.1f} deg",
        ]
        x = 18
        y = WINDOW_HEIGHT - 24
        for index, text in enumerate(lines):
            label = pyglet.text.Label(
                text,
                x=x,
                y=y - index * 22,
                font_name="Consolas",
                font_size=12,
                color=(238, 241, 244, 255),
            )
            label.draw()

    @staticmethod
    def _ray_color(normalized_distance: float) -> tuple[int, int, int]:
        if normalized_distance < 0.28:
            return (232, 78, 64)
        if normalized_distance < 0.55:
            return (232, 178, 75)
        return (93, 202, 128)


def run_pyglet_app(world: RacingWorld) -> None:
    RacingWindow(world)
    pyglet.app.run()
