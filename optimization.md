# Optimization notes

This documents why the original `index.py` is resource-hungry and what
`optimized.py` does differently.

## The problem

`index.py` detects when a hand touches your face and sleeps the display. It
works, but it pins a CPU core and heats the machine. Three causes, in order of
impact:

1. **`mp.solutions.Holistic` runs every frame.** Holistic bundles three heavy
   models — FaceMesh (468 landmarks), Pose, and two Hands — and runs all of them
   on every frame. The task only needs a face bounding box and hand positions,
   so nearly all of that compute is wasted.
2. **Uncapped loop.** `while True` with no sleep processes frames as fast as the
   CPU allows, holding a core at ~100%. Detecting a hand on a face does not need
   30–60 fps.
3. **Full-resolution frames.** Inference cost scales with pixel count, and the
   frame is fed in at full camera resolution.

Minor: `cv2.waitKey(1)` never returns `q` because there is no `imshow` window,
so the intended quit key does nothing — the script can only be killed with
Ctrl+C.

## What `optimized.py` changes

| Change | Effect |
| --- | --- |
| **FaceDetection (BlazeFace) + Hands(`model_complexity=0`)** instead of Holistic | Drops FaceMesh + Pose; uses the lite hand model. Provides exactly the box + landmarks the comparison needs. Main saving. |
| **Face-first gating** | The cheap face detector runs first; the expensive hand model runs only when a face is present. When you are out of frame, the hand step is skipped entirely. |
| **FPS cap (`TARGET_FPS = 6`)** | Sleeps off the remainder of each frame budget so the CPU idles between frames instead of spinning. |
| **Downscale to 480px wide** + request 640×480 camera feed | Fewer pixels per inference and less data to decode/move. |
| **Trigger cooldown** | Avoids firing `pmset` repeatedly while a touch persists. |
| **Explicit cleanup** | Releases the camera and closes both models on exit. |

Detection logic is unchanged in spirit: normalized-coordinate bounding-box
check. A small `FACE_PADDING` is added because a face-detection box is tighter
than the old FaceMesh min/max extent.

## Tuning knobs

Top of `optimized.py`:

- `TARGET_FPS` — lower = cooler and lower-power, but slower to react.
- `PROCESS_WIDTH` — inference resolution.
- `FACE_PADDING` — how forgiving the face area is.
- `TRIGGER_COOLDOWN` — seconds before the display can be slept again.

## Run

```bash
cd /Users/serhiilk/Projects/hands_tracker
venv/bin/python optimized.py
```

Stop with Ctrl+C.

## Verifying the gain

This has not been benchmarked on real hardware. To measure, run each script for a
minute and compare CPU in Activity Monitor / `top`. Change one variable at a time
(e.g. FPS cap vs. model swap) if you want to know which factor dominates — it
depends partly on your camera's native frame rate.
