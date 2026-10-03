# coding: utf-8
"""Generate the shared synthetic pool for arms B and C (Stable Diffusion v1.5)."""
import csv, json, sys
from pathlib import Path
import numpy as np, torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

POOL   = Path("/workspace/food101_lt_synth"); POOL.mkdir(exist_ok=True)
ALLOC  = Path("/workspace/food101_lt/pool_alloc.json")
MODEL  = "runwayml/stable-diffusion-v1-5"
CACHE  = "/workspace/hf_cache"
STEPS, GUIDANCE, BATCH, SEED = 50, 2.0, 8, 42

need = json.loads(ALLOC.read_text())          # {class_name: n_images}

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, safety_checker=None,
    cache_dir=CACHE).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

g = torch.Generator("cuda")
for ci, (cls, n) in enumerate(sorted(need.items())):
    if n == 0: continue
    d = POOL/cls; d.mkdir(exist_ok=True)
    have = len(list(d.glob("*.jpg")))
    if have >= n:
        print(f"[{ci+1:>3}/101] {cls:<24} {have} already"); continue
    prompt = f"A photo of {cls.replace('_',' ')}, a type of food."
    made = have
    while made < n:
        k = min(BATCH, n - made)
        g.manual_seed(SEED * 100000 + ci * 1000 + made)
        imgs = pipe([prompt]*k, num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE, generator=g).images
        for im in imgs:
            im.resize((512,512)).save(d/f"{made:05d}.jpg", quality=95)
            made += 1
    print(f"[{ci+1:>3}/101] {cls:<24} {made}")

print("pool complete")
