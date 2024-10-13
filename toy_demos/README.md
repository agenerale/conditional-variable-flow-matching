This directory contains example code to duplicate 2D mappings with discrete and continuous conditioning variables. Each script runs through variants of conditional flow and score matching approaches detailed in the paper, specifically, CVFM, CVSFM, COT-FM, COT-SFM, CFM, CSFM.

Each mapping can be run sweeping through the various approaches and generating respective gifs contained in `../imgs/`.

```bash
# Discrete conditioning - 8 Gaussians to 2 Moons
python toy_demos/8gauss_2moon.py

# Discrete conditioning - 8 Gaussians to 2 Moons
python toy_demos/8gauss_2moon.py

# Continuous conditioning - 2 Moons to 2 Moons
python toy_demos/2moons_2moons.py
```
