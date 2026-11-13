# text-to-video

Text-to-video with Wan 2.2 TI2V-5B (open weights, Apache 2.0) and an automatic company logo.

- `generate_wan.py`  make a video, then add the logo from `logos/` automatically
- `download_wan.py`  download the full model into `models/` (not stored in git)
- `wan2_2/`          the official Wan 2.2 model code (see `wan2_2/SOURCE.md`)
- `add_logo.py`, `logoblend/`  logo placement

The earlier GenTron implementation is kept under the git tag `gentron-final`.
