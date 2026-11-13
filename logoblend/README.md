# LogoBlend

A small separate model that makes a pasted logo look **printed on the object** (a shirt, a poster, a box).
It is independent of GenTron: it does not touch the image or video model and needs no retraining of them.

## What it does (and does not do)

- It does **not** draw the logo. `add_logo.py` pastes the exact logo and tracks it with the surface.
- For every pixel of the logo area, LogoBlend predicts: **shading** (the cloth's light), a **small bend** (follow folds),
  and **visibility** (a hand, hair or object in front hides the logo).
- The final pixel is computed from the real logo and those predictions, so the logo's shape and text stay exact and a
  logo it has never seen works. No per-user training.

```
GenTron video -> find + track the garment (CLIPSeg now, SAM 2 later) -> add_logo.py places the exact logo
              -> LogoBlend lights / bends / occludes it, frame by frame -> final video
```

## Files

| file | purpose |
|---|---|
| `model.py` | `BlendNet` (small U-Net) and `compose()` (the exact-logo formula) |
| `data.py` | the training-data simulator, procedural scenes, folder dataset |
| `train.py` | training loop |
| `prepare_masks.py` | garment masks for a folder of real photos (CLIPSeg) |
| `apply.py` | `LogoBlender`, used by `add_logo.py --blend_ckpt` |

## Train

First test, no data needed (procedural cloth, random logos):
```
python -m logoblend.train --data synthetic --steps 3000 --size 128 --out logoblend_synth.pt
```
Real training (recommended): a folder of photos of people wearing plain or printed clothes (any source you have the
right to use), garment masks, and optionally a folder of many different logos.
```
python -m logoblend.prepare_masks --images photos --masks masks --prompt "a t-shirt"
# look at masks/ and delete the bad ones
python -m logoblend.train --data folder --images photos --masks masks --logos logos --steps 20000 --size 128 --out logoblend.pt
```
Use as many different logos as you can (hundreds+): the model must learn "any logo", not your customers' logos.
`--resume file.pt` continues a run. Checkpoints are written atomically.

## Use

```
python add_logo.py --video clip.mp4 --logo my_logo.png --blend_ckpt logoblend.pt
```

## Honest limits

- **The training target is simulated**, not photographed. The simulator (`data.simulate`) fakes how a print looks
  (lighting from the image under the logo, a small bend along folds, occluders on top). The network learns the
  simulator and how to read the scene, which makes it robust (it handles occluders and messy lighting that the
  hand-made blend in `add_logo.py` cannot), but it will not look more realistic than the simulator. To go further, add
  real photos of printed garments and improve `simulate()`.
- It needs a **consistent subject across frames**: it works frame by frame, so if the video model shows a different
  person every frame, the logo still jumps.
- The segmentation (garment mask) is the other weak link: CLIPSeg masks are crude. Better masks give better results.
- Trained here only on procedural data and tested on CPU. Real-photo quality has not been measured.
