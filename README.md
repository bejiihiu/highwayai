# Pyglet Racing AI Sandbox

Minimal top-down racing sandbox for AI experiments. The default agent returns zero
actions, so the car starts on the line and stays there until you connect a model.

## Setup

```powershell
python -m pip install -r requirements.txt
python main.py
```

For a quick visual movement smoke test without adding keyboard controls:

```powershell
python main.py --demo-agent
```

Camera controls: mouse wheel zoom, `WASD` / arrow keys, edge-scroll near the
window border, middle-mouse drag, and `Space` to snap back to the car.

Run logic tests:

```powershell
python -m unittest discover -s tests
```

## AI contract

`RacingWorld.step(action, dt)` accepts:

```python
{"throttle": 0.0, "steer": 0.0, "brake": 0.0}
```

Action ranges are:

- `throttle`: `-1.0..1.0`
- `steer`: `-1.0..1.0`
- `brake`: `0.0..1.0`

The action format is intentionally small so different teams can plug in rule
agents, RL policies, or imitation models without changing the simulator.

`RacingWorld.reset()` starts a fresh episode-like run and returns the first
observation. `RacingWorld.step(action, dt)` advances the world and returns an
observation dictionary with:

- motion state: `speed`, `forward_speed`, `lateral_speed`, `heading`,
  `track_heading`, `heading_error`, `slip_angle`, `drift_intensity`
- track state: `distance_to_center`, `signed_distance_to_center`,
  `edge_clearance`, `off_track`, `progress`, `lap`
- boundary state: `edge_collision`, `edge_collision_count`,
  `last_edge_impact`, `off_track_count`
- rewards and markers: `frame_reward`, `total_reward`, `marker_reward`,
  `markers_collected`, `markers_total`, `nearest_markers`
- sensors: `rays`, `car_position`, `car_velocity`

`off_track` means the car center is outside the track width. `edge_collision`
means the car body touched an edge; it can happen before `off_track` because
the car has width. Edge contacts softly push the car back inside, damp outward
velocity, and add reward penalties. Going fully off-track still reports the
existing off-track state and penalty.

Reward signals are deliberately simple:

- collect marker rewards by driving through visible markers
- earn small drift points only while on track
- lose reward for new edge impacts, sliding along the edge, and going off track

The Pyglet window also exposes `capture_frame_image()` for future pixel-based
observations.

## Next improvements

Suggested follow-up work before handing this to multiple AI authors:

- add explicit `done` / `done_reason` episode termination fields
- add configurable time limits and reset options
- add deterministic seeds for repeatable experiments
- add a simple baseline agent for comparison
- export per-step logs for reward, progress, collisions, and lap timing
