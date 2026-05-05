# Results gallery

Drop your best curated outputs here. Suggested set for a portfolio-strength gallery:

- **`hero.png`** — the single most striking result; gets featured in the main README.
- **`single_style_picasso.png`** — full Picasso takeover, demonstrates style fidelity.
- **`single_style_starrynight.png`** — full Van Gogh takeover, shows the swirl-in-eye motif placement.
- **`spatial_mix.png`** — different style per region (e.g., Picasso duck on Starry Night sky), shows clean mask boundaries.
- **`intensity_mix.png`** — same style, different per-region weights, shows painterly emphasis control.
- **`tile_comparison/`** — same content + style at three Tile values (0.0, 0.4, 0.8) showing color preservation tradeoff.

Each image should be a PNG at the same dimensions as the input content image (the pipeline preserves aspect ratio and resizes back to input dimensions on output).

For each image, consider also saving the **seed** and slider configuration in a sidecar `.json` so the result is reproducible. Format suggestion:

```json
{
  "content": "Sir_Quack.jpg",
  "style_a": "picasso_old_guitarist.jpg",
  "style_b": "starry_night.jpg",
  "weights": [0.7, 0.7],
  "tile_scale": 0.4,
  "controlnet_scale": 1.0,
  "canny": [100, 200],
  "guidance": 7.0,
  "steps": 30,
  "seed": 42
}
```
