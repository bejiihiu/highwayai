from __future__ import annotations

import math

import pyglet
from pyglet import shapes

from racing_ai.math2d import angle_to_vector
from racing_ai.track import RewardMarker
from racing_ai.world import RacingWorld


WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 760


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
        self.static_batch = pyglet.graphics.Batch()
        self._build_static_track()
        pyglet.clock.schedule_interval(self.update, 1.0 / 60.0)

    def update(self, dt: float) -> None:
        self.world.update(dt)

    def on_draw(self) -> None:
        pyglet.gl.glClearColor(
            self.background_color[0] / 255.0,
            self.background_color[1] / 255.0,
            self.background_color[2] / 255.0,
            1.0,
        )
        self.clear()
        self.static_batch.draw()
        self._draw_markers()
        self._draw_rays()
        self._draw_car()
        self._draw_overlay()
        self.fps_display.draw()

    def capture_frame_image(self) -> pyglet.image.ImageData:
        """Return the current color buffer for agents that want pixel observations."""
        return pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()

    def _build_static_track(self) -> None:
        road_color = (45, 48, 51)
        border_color = (218, 222, 226)
        inner_warning = (156, 58, 58)
        centerline_color = (225, 201, 93)

        for start, end in self.world.track.center_segments:
            shapes.Line(*start, *end, thickness=self.world.track.width, color=road_color, batch=self.static_batch)

        for start, end in self.world.track.inner_segments:
            shapes.Line(*start, *end, thickness=5, color=border_color, batch=self.static_batch)
            shapes.Line(*start, *end, thickness=2, color=inner_warning, batch=self.static_batch)

        for start, end in self.world.track.outer_segments:
            shapes.Line(*start, *end, thickness=5, color=border_color, batch=self.static_batch)

        for index, (start, end) in enumerate(self.world.track.center_segments):
            if index % 8 in (0, 1, 2):
                shapes.Line(*start, *end, thickness=3, color=centerline_color, batch=self.static_batch)

        start_a, start_b = self.world.track.start_line()
        shapes.Line(*start_a, *start_b, thickness=8, color=(245, 245, 245), batch=self.static_batch)
        shapes.Line(start_a[0], start_a[1], start_b[0], start_b[1], thickness=3, color=(35, 35, 35), batch=self.static_batch)

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
