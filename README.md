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

Run logic tests:

```powershell
python -m unittest discover -s tests
```

## AI hook

`RacingWorld.step(action, dt)` accepts:

```python
{"throttle": 0.0, "steer": 0.0, "brake": 0.0}
```

It returns an observation dictionary with speed, heading error, off-track state,
reward totals, marker data, and ray distances. The Pyglet window also exposes
`capture_frame_image()` for future pixel-based observations.
