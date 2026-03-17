"""Batch experiment runner for Stable Diffusion parameter studies."""

import time
import os
import csv
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from PIL import Image
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from evaluate import compute_clip_score

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "PNDM": PNDMScheduler,
    "DPM++2M": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
}

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"


@dataclass
class ExperimentResult:
    experiment: str
    variable: str
    value: str
    prompt: str
    seed: int
    clip_score: float
    gen_time_sec: float
    image_path: str


def _load_pipeline(model_id: str = DEFAULT_MODEL, scheduler_name: str = "DDIM"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if scheduler_name in SCHEDULERS:
        pipe.scheduler = SCHEDULERS[scheduler_name].from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.safety_checker = None  # disable for research use
    return pipe


def _generate(pipe, prompt: str, negative_prompt: str = "", guidance_scale: float = 7.5,
              num_inference_steps: int = 20, seed: int = 42) -> tuple[Image.Image, float]:
    generator = torch.Generator().manual_seed(seed)
    t0 = time.time()
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    elapsed = time.time() - t0
    return result.images[0], elapsed


def run_experiment(
    name: str,
    variable: str,
    values: list,
    prompts: List[str],
    seeds: List[int],
    base_cfg: float = 7.5,
    base_steps: int = 20,
    base_scheduler: str = "DDIM",
    base_model: str = DEFAULT_MODEL,
    negative_prompt: str = "",
) -> List[ExperimentResult]:
    """Run a single-variable experiment across multiple prompts and seeds."""
    results = []
    pipe = _load_pipeline(base_model, base_scheduler)

    for val in values:
        for prompt in prompts:
            for seed in seeds:
                cfg = val if variable == "cfg_scale" else base_cfg
                steps = val if variable == "steps" else base_steps
                scheduler = val if variable == "scheduler" else base_scheduler
                model = val if variable == "model" else base_model

                # Reload pipeline if model or scheduler changed
                if variable in ("model", "scheduler"):
                    pipe = _load_pipeline(model, scheduler)

                image, elapsed = _generate(
                    pipe, prompt, negative_prompt,
                    guidance_scale=cfg,
                    num_inference_steps=steps,
                    seed=seed,
                )
                clip = compute_clip_score(image, prompt)

                # Save image
                safe_val = str(val).replace("/", "-").replace(" ", "_")
                img_name = f"{name}_{variable}_{safe_val}_seed{seed}_{prompt[:20].replace(' ', '_')}.png"
                img_path = os.path.join(RESULTS_DIR, img_name)
                image.save(img_path)

                results.append(ExperimentResult(
                    experiment=name,
                    variable=variable,
                    value=str(val),
                    prompt=prompt,
                    seed=seed,
                    clip_score=round(clip, 4),
                    gen_time_sec=round(elapsed, 2),
                    image_path=img_path,
                ))

    return results


def save_results(results: List[ExperimentResult], filename: str = "results/all_results.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])
    print(f"Saved {len(results)} results to {filename}")


# ── Predefined experiment configs ──────────────────────────────────────────────

PROMPTS = [
    "a majestic lion in the savanna at sunset",
    "a futuristic city skyline with flying cars",
    "a cozy cabin in a snowy forest",
]
SEEDS = [42, 123, 999]

EXPERIMENTS = [
    dict(
        name="exp1_cfg_scale",
        variable="cfg_scale",
        values=[1, 3, 5, 7, 10, 15, 20],
        base_steps=20,
        base_scheduler="DDIM",
    ),
    dict(
        name="exp2_steps",
        variable="steps",
        values=[10, 15, 20, 30, 50],
        base_cfg=7.5,
        base_scheduler="DDIM",
    ),
    dict(
        name="exp3_scheduler",
        variable="scheduler",
        values=["DDIM", "PNDM", "DPM++2M", "Euler"],
        base_cfg=7.5,
        base_steps=20,
    ),
    dict(
        name="exp4_prompt_complexity",
        variable="cfg_scale",  # fixed; prompts vary externally
        values=[7.5],
        base_steps=20,
        base_scheduler="DDIM",
    ),
    dict(
        name="exp5_model",
        variable="model",
        values=[
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
        ],
        base_cfg=7.5,
        base_steps=20,
        base_scheduler="DDIM",
    ),
]


if __name__ == "__main__":
    all_results = []
    for exp_cfg in EXPERIMENTS:
        print(f"\nRunning {exp_cfg['name']} ...")
        results = run_experiment(
            prompts=PROMPTS,
            seeds=SEEDS,
            **exp_cfg,
        )
        all_results.extend(results)

    save_results(all_results)
