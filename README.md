# Pyglet Racing AI Sandbox

Top-down racing sandbox with a realism-focused RWD physics model and hybrid-action
RL stack (PPO default, SAC optional).

## Setup

```powershell
python -m pip install -r requirements.txt
python main.py
```

Quick visual movement test:

```powershell
python main.py --demo-agent
```

Train PPO (default):

```powershell
python main.py --train --algo ppo --episodes 5000
```

Train SAC:

```powershell
python main.py --train --algo sac --episodes 5000
```

Play from checkpoint:

```powershell
python main.py --play ai/checkpoints/best.pt --algo ppo
```

Run tests:

```powershell
python -m unittest discover -s tests
```

## Action Contract

`RacingWorld.step(action, dt)` accepts a hybrid control dict:

```python
{
  "throttle": 0.0,   # -1.0..1.0
  "steer": 0.0,      # -1.0..1.0
  "brake": 0.0,      # 0.0..1.0
  "clutch": 0.0,     # 0.0..1.0 (0 engaged, 1 disengaged)
  "handbrake": 0.0,  # 0.0..1.0
  "gear_up": 0.0,    # 0 or 1
  "gear_down": 0.0   # 0 or 1
}
```

Agents can still provide only `throttle/steer/brake`; missing keys default to zero.

## Observation Highlights

`RacingWorld.step(...)` returns motion, track, and sensor state plus gearbox diagnostics,
including: `rpm`, `gear`, `clutch`, `shift_attempted`, `shift_applied`, `shift_blocked`,
`shift_block_reason`, `stall_event`, `money_shift_event`, `engine_load`, `clutch_slip`,
`traction`, `drive_force`, and `wheel_torque`.

Camera controls: mouse wheel zoom, `WASD` / arrow keys, edge-scroll, middle-mouse drag,
and `Space` to snap back to the car.
