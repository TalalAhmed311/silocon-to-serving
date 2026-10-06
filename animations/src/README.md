# Animation sources

Every animation in `animations/*.html` is one **self-contained** file, with no network requests. Each file inlines three shared pieces from this folder:

- `core.css`: theme tokens (light and dark, following `prefers-color-scheme`, with a manual toggle) and a layout that works at phone width
- `core.js`: play, pause, step and reset; a speed slider; keyboard shortcuts (space, ←, →, r); a progress bar
- `controls.html`: the control bar markup

Each animation then defines `STEPS` (an array of `{caption, ...state}`) and `draw(svg, step, index)`.

`python tools/build_animations.py` re-inlines these three files into every animation, between the `/*CORE-CSS*/`, `/*CORE-JS*/` and `<!--CONTROLS-->` markers. Run it after editing anything in this folder.
