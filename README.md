---
title: MixStyleGAN
emoji: 🎨
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# MixStyleGAN

> **Mix two artistic styles into a single painting** — with weight sliders, paintable region masks, and color-preserving structure transfer.

Upload a content image and two style references (e.g., Van Gogh's *Starry Night* and Picasso's *Old Guitarist*). Two sliders control how much of each style is mixed in. Optionally paint a region: Style A applies inside the painted area, Style B applies outside.

No training required — any two paintings work as style references.

<!-- Add a hero image to examples/results/hero.png -->

## What's interesting about it

Cross-style mixing is harder than it looks. Naive approaches produce muddy ghosts:

- **Train one GAN per style, average pixels** → ghosting and washed-out colors. (See `MixStyleGAN.py`, the legacy CycleGAN scaffold this project started from.)
- **Train paired LoRAs and merge** → style interference; the two LoRAs fight for the same attention layers.

This project takes a third path: pretrained **Stable Diffusion 1.5** + **dual ControlNet** (Canny for structure, Tile for color/composition) + **dual IP-Adapter** (one per style image) with optional **spatial masks** for region-specific application. No training; arbitrary style references; clean per-region control.

## Live demo

- **Hugging Face Space:** *(deploy then add link)*
- **Colab notebook:** [`MixStyleGAN_Colab.ipynb`](MixStyleGAN_Colab.ipynb) — open in Colab, set Runtime → T4 GPU, Run All.

## Examples

Drop a curated selection of your best outputs into [`examples/results/`](examples/results/). Suggested set:

- A single-artist takeover (e.g., full Picasso).
- A cross-artist mix with spatial mask (e.g., Picasso duck on Starry Night sky).
- A subtle-vs-dramatic intensity demo (same style at different weights per region).
- A mask-debug strip (content / mask / canny / result).

## How it works

```
                       ┌───────────────────────┐
                       │  Content image        │
                       └──┬─────────────────┬──┘
                          │                 │
                          ▼                 ▼
                 ┌─────────────┐   ┌─────────────────┐
                 │ Canny edges │   │ Tile (raw RGB)  │
                 └──────┬──────┘   └────────┬────────┘
                        │                   │
              ControlNet-Canny      ControlNet-Tile
              (structure lock)      (color/composition)
                        │                   │
                        └─────────┬─────────┘
                                  │
                                  ▼
                  ┌──────────────────────────────┐
   Style A ─────► │     Stable Diffusion 1.5     │
                  │  (UNet with 2 IP-Adapters)   │ ◄───── Style B
                  └──────────────┬───────────────┘
                                 │
                       ip_adapter_masks: [A, B]
                                 │
                                 ▼
                          Stylized result
```

**The four knobs that actually matter:**

| Slider | What it controls |
|---|---|
| Style A / B weight | How strongly each style imprints (in its region, if a mask is drawn). |
| Color preservation (Tile) | How much of the original palette persists. 0 = full style takeover; 0.4 = sweet spot; 0.8+ = mostly original colors. |
| Structure (Canny scale) | How rigidly the result follows the input's edges. |
| Canny low/high thresholds | Edge-detection sensitivity. Tune for soft-gradient vs sharp-contrast inputs. |

**Three modes:**

1. **Different styles, same weight** → cross-style mix.
2. **Same style image, different per-region weights** → painterly emphasis (subject readable, background dramatic).
3. **Different styles, spatial mask** → Style A inside the painted region, Style B outside. Mask boundary stays clean.

## Quickstart

### Run on Colab (recommended for first-time users)

1. Open [`MixStyleGAN_Colab.ipynb`](MixStyleGAN_Colab.ipynb) in Google Colab.
2. `Runtime > Change runtime type > T4 GPU`.
3. `Runtime > Run all`. The last cell prints a `gradio.live` URL.
4. Open the URL → upload content + two style references → drag sliders → Generate.

### Run on Hugging Face Spaces

This repository is configured as a Gradio Space (see frontmatter at the top of this file). To deploy your own:

```bash
huggingface-cli login
huggingface-cli repo create mixstylegan --type=space --space_sdk=gradio
git remote add space https://huggingface.co/spaces/<your-username>/mixstylegan
git push space main
```

The Space will pick up `app.py`, install `requirements.txt`, and start the Gradio UI on a free GPU tier.

### Run locally

Requires NVIDIA GPU with ~6 GB VRAM minimum (8+ recommended).

```bash
git clone https://github.com/<your-username>/MixStyleGAN.git
cd MixStyleGAN
pip install -r requirements.txt
python app.py
```

## Repository layout

```
.
├── app.py                          # Gradio entry point (HF Spaces / local)
├── mixstylegan_pipeline.py         # Pipeline, helpers, generate function
├── MixStyleGAN_Colab.ipynb         # Self-contained Colab notebook
├── examples/
│   └── results/                    # Curated outputs gallery
├── MixStyleGAN.py                  # Legacy CycleGAN scaffold (research baseline)
├── requirements.txt                # Pinned dependencies
└── README.md                       # This file (also the HF Space README)
```

## Limitations

- **Resolution.** SD 1.5 generates at ~512 short side. Outputs are resized to input dimensions but don't gain real detail — Lanczos upscale, not super-resolution. SDXL upgrade is the cleanest path to higher fidelity.
- **Small color regions get overridden.** Tiny saturated details (a coral bowtie, a red highlight) tend to lose their color when the dominant style palette is very different. Mitigation: push **Color preservation** higher (0.6+) for runs with palette-dominant styles like Picasso's Blue Period.
- **CLIP encodes style vocabulary, not objects.** A Picasso reference of *The Old Guitarist* won't paint a guitar into your output — by design. The IP-Adapter base variant (used here) is specifically chosen for clean style-only transfer.
- **Edge density vs style motif interaction.** Specific style motifs (Van Gogh's swirls) only manifest where Canny edges are sparse — they don't override densely-edged regions like a face. This is mostly a feature, but explains why some regions read as "Van Gogh-y" without showing literal swirls.

## Roadmap

- [ ] **SDXL upgrade** — drop-in replacement for SD 1.5 base, ~1024 native resolution.
- [ ] **Real-ESRGAN final upscale** — actual high-res output, not just dimension-matching.
- [ ] **Multi-region masks** — 3+ style regions instead of A-inside / B-outside.
- [ ] **Soft-edge masks** — gradient transitions at mask boundaries.
- [ ] **Saved presets** — content + style + slider config bundles.
- [ ] **HF Space deployment** with a public demo link.

## Research angle

The strongest empirical thread that surfaced during development is **the interaction between ControlNet edge density and IP-Adapter style-motif manifestation** — specifically, that style features have a fragility hierarchy (palette and brushwork-direction transfer at any weight; specific motifs like swirls only crystallize at high weight, and only in regions where ControlNet has nothing to enforce). This is a falsifiable claim and would be the seed of a workshop paper if formalized with controlled experiments. See [`docs/research-direction.md`](docs/research-direction.md) for notes (TBD).

## Acknowledgements

Built on top of:

- [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [ControlNet](https://github.com/lllyasviel/ControlNet) (Canny, Tile)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [diffusers](https://github.com/huggingface/diffusers), [transformers](https://github.com/huggingface/transformers), [Gradio](https://github.com/gradio-app/gradio)

## License

MIT — see [`LICENSE`](LICENSE).
