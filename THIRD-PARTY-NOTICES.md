# Third-Party Notices

ComfyUI-Lee-RIFE's own code is licensed under the MIT License (see `LICENSE`).
It bundles and builds upon the third-party components listed below. Each is used
under its own license; the original copyright and license notices are reproduced
here as required.

\---

## 1\) `rife\_arch.py` — RIFE architecture (IFNet) implementation

Vendored **verbatim** from **ComfyUI-Frame-Interpolation** by Fannovel16:
https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

That file in turn derives from RIFE / Practical-RIFE and vs-rife — see the
attribution header inside `rife\_arch.py`:

* https://github.com/hzwer/Practical-RIFE
* https://github.com/HolyWu/vs-rife

License: **MIT**

```
MIT License

Copyright (c) 2023 Fannovel16

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

\---

## 2\) Model weights — `models/rife47/49.pth` (RIFE v4.7/v4.9)

Source: **RIFE / Practical-RIFE** by Zhewei Huang (hzwer) and contributors:

* https://github.com/hzwer/Practical-RIFE
* https://github.com/megvii-research/ECCV2022-RIFE

The RIFE models are released by their authors under the **MIT License**
(the Practical-RIFE README states the model files are distributed under the
same MIT license as the project). © the RIFE / Practical-RIFE authors. For the
authoritative copyright line, see the `LICENSE` file in the repositories above.

MIT permission terms (same text as above) apply to these weights:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND ...
```

If you use this in academic work, please cite RIFE:

```
@inproceedings{huang2022rife,
  title     = {Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author    = {Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```

\---

## 3\) ComfyUI (host application)

This is a custom node for ComfyUI (https://github.com/comfyanonymous/ComfyUI),
which is licensed under GPL-3.0. ComfyUI is **not** bundled or redistributed with
this package — it is the host that loads this node at runtime. This node's own
code is provided under the MIT License (see `LICENSE`).

