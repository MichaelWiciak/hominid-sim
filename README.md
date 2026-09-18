# hominid-sim

An evolution and extinction simulator for humans and Neanderthals, built with
Panda3D and PyTorch.

Every individual carries 22 inheritable traits (strength, intelligence, tool
use, cold tolerance...). A neural network trained per scenario predicts each
one's chance of survival; populations evolve across hundreds of thousands of
years — until they go extinct.

Written as my A-level Computer Science project at 16/17, ~3,000 lines in a
single file. The code is left exactly as it was handed in.

## Run

```sh
cd src
python hominid_sim.py
```

Requires Python 3, Panda3D, PyTorch, numpy, and `names`.

## Layout

- `src/` — the entire simulation: script, pre-trained models, models/textures
- `LICENSE` — CC0