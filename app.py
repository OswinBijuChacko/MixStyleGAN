"""
MixStyleGAN — Gradio app entry point.

Used as the deployment artifact for Hugging Face Spaces and as the local-launch
script for anyone with a CUDA GPU. The pipeline definition lives in
mixstylegan_pipeline.py; this file is just the UI and the launch call.
"""

import gradio as gr

from mixstylegan_pipeline import (
    generate as _generate,
    load_pipeline,
    preview_canny,
)


def generate(*args, **kwargs):
    """Thin wrapper to translate ValueError into a Gradio-friendly error."""
    try:
        return _generate(*args, **kwargs)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="MixStyleGAN — 2-Style Painting Blender") as demo:
    gr.Markdown("## MixStyleGAN — Mix Two Painting Styles")
    gr.Markdown(
        "Upload a content image and two style references. Optionally paint a "
        "mask: Style A applies inside the painted region, Style B applies "
        "outside. Leave the mask empty for global mixing."
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                content_in = gr.Image(label="Content", type="pil", height=220)
                mask_editor = gr.ImageEditor(
                    label="Paint Style A region (transparent = Style B)",
                    type="pil",
                    sources=["upload"],
                    transforms=[],
                    layers=False,
                    height=260,
                    brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed", default_size=40),
                )

            with gr.Row():
                style_a_in = gr.Image(label="Style A", type="pil", height=160)
                style_b_in = gr.Image(label="Style B", type="pil", height=160)

            with gr.Row():
                weight_a = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Style A weight")
                weight_b = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Style B weight")

            tile_scale = gr.Slider(
                0.0, 1.0, value=0.4, step=0.05,
                label="Color preservation (Tile)",
                info="Pulls original image colors through. 0 = full style takeover; "
                     "0.3–0.5 = sweet spot; 0.8+ = mostly original palette.",
            )

            run = gr.Button("Generate", variant="primary", size="lg")

            with gr.Accordion("Edge sensitivity (Canny)", open=False):
                with gr.Row():
                    canny_low = gr.Slider(0, 255, value=100, step=5, label="Low threshold")
                    canny_high = gr.Slider(0, 255, value=200, step=5, label="High threshold")

            with gr.Accordion("Advanced", open=False):
                prompt = gr.Textbox(label="Prompt (optional)", value="a painting", lines=1)
                with gr.Row():
                    steps = gr.Slider(15, 50, value=30, step=1, label="Steps")
                    controlnet_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Canny scale")
                    guidance = gr.Slider(1.0, 12.0, value=7.0, step=0.5, label="Guidance")
                seed = gr.Number(label="Seed", value=42, precision=0)

        with gr.Column(scale=2):
            output = gr.Image(label="Result", type="pil", height=480)
            mode_label = gr.Markdown()
            with gr.Row():
                mask_view = gr.Image(label="Mask (debug)", type="pil", height=180)
                canny_view = gr.Image(label="Canny (live)", type="pil", height=180)

    # Auto-load uploaded content into the mask editor as background
    content_in.change(lambda img: img, content_in, mask_editor)

    # Live Canny preview when content or thresholds change
    for trigger in (content_in, canny_low, canny_high):
        trigger.change(
            preview_canny,
            [content_in, canny_low, canny_high],
            canny_view,
        )

    run.click(
        generate,
        [
            content_in, mask_editor, style_a_in, style_b_in,
            weight_a, weight_b, prompt, steps, controlnet_scale, guidance,
            canny_low, canny_high, tile_scale, seed,
        ],
        [output, canny_view, mask_view, mode_label],
    )


if __name__ == "__main__":
    # Eager-load the pipeline on startup so the first user request doesn't
    # eat the ~30 s model load time.
    load_pipeline()
    # share=True useful for local runs; HF Spaces ignores it and provides
    # its own public URL.
    demo.launch(share=True)
