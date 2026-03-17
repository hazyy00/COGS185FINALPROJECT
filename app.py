"""Stable Diffusion Parameter Explorer — Streamlit App."""

import time
import torch
import streamlit as st
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from evaluate import compute_clip_score

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "PNDM": PNDMScheduler,
    "DPM++2M": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
}

MODELS = {
    "SD v1.5": "runwayml/stable-diffusion-v1-5",
    "SD v2.1": "stabilityai/stable-diffusion-2-1",
}


@st.cache_resource(show_spinner="Loading model...")
def load_pipeline(model_id: str, scheduler_name: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.scheduler = SCHEDULERS[scheduler_name].from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.safety_checker = None
    return pipe


def generate_image(pipe, prompt, negative_prompt, cfg, steps, seed) -> tuple[Image.Image, float]:
    generator = torch.Generator().manual_seed(seed)
    t0 = time.time()
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=generator,
    )
    return result.images[0], round(time.time() - t0, 2)


# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SD Parameter Explorer", layout="wide")
st.title("Stable Diffusion Parameter Explorer")
st.caption("Systematic study of generation parameters for CS 185 Final Project")

with st.sidebar:
    st.header("Parameters")
    model_name = st.selectbox("Model", list(MODELS.keys()))
    scheduler_name = st.selectbox("Scheduler", list(SCHEDULERS.keys()))
    cfg_scale = st.slider("Guidance Scale (CFG)", 1.0, 20.0, 7.5, 0.5)
    steps = st.slider("Denoising Steps", 5, 50, 20)
    seed = st.number_input("Seed", value=42, step=1)

prompt = st.text_area("Prompt", "a majestic lion in the savanna at sunset, photorealistic")
negative_prompt = st.text_input("Negative Prompt", "blurry, low quality, distorted")

col1, col2 = st.columns([1, 3])
with col1:
    generate_btn = st.button("Generate", type="primary", use_container_width=True)

if generate_btn:
    pipe = load_pipeline(MODELS[model_name], scheduler_name)
    with st.spinner("Generating..."):
        image, elapsed = generate_image(pipe, prompt, negative_prompt, cfg_scale, steps, int(seed))
        clip = compute_clip_score(image, prompt)

    col_img, col_metrics = st.columns([2, 1])
    with col_img:
        st.image(image, caption=f"Seed {seed}", use_container_width=True)
    with col_metrics:
        st.metric("CLIP Score", f"{clip:.4f}")
        st.metric("Generation Time", f"{elapsed}s")
        st.metric("CFG Scale", cfg_scale)
        st.metric("Steps", steps)
        st.metric("Scheduler", scheduler_name)
        st.metric("Model", model_name)

        # Save button
        img_bytes = image.tobytes()
        st.download_button(
            "Download Image",
            data=open("/tmp/sd_output.png", "wb") and image.save("/tmp/sd_output.png") or open("/tmp/sd_output.png", "rb").read(),
            file_name=f"sd_cfg{cfg_scale}_steps{steps}_seed{seed}.png",
            mime="image/png",
        )

st.divider()
st.subheader("Batch Experiment Mode")
st.caption("Vary one parameter across a range and compare results side by side.")

with st.expander("Run CFG Scale Sweep"):
    sweep_prompt = st.text_input("Sweep Prompt", "a futuristic city skyline at night", key="sweep_prompt")
    sweep_steps = st.slider("Fixed Steps", 5, 50, 20, key="sweep_steps")
    sweep_seed = st.number_input("Fixed Seed", value=42, key="sweep_seed")
    cfg_values = st.multiselect("CFG values to sweep", [1, 3, 5, 7, 10, 15, 20], default=[3, 7, 10, 15])

    if st.button("Run CFG Sweep"):
        pipe = load_pipeline(MODELS[model_name], scheduler_name)
        cols = st.columns(len(cfg_values))
        for i, cfg_val in enumerate(cfg_values):
            with cols[i]:
                img, t = generate_image(pipe, sweep_prompt, "", float(cfg_val), sweep_steps, int(sweep_seed))
                clip = compute_clip_score(img, sweep_prompt)
                st.image(img, caption=f"CFG={cfg_val}\nCLIP={clip:.3f}\n{t}s", use_container_width=True)
