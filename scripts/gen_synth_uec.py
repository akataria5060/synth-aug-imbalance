# coding: utf-8
"""Generate the shared synthetic pool for UEC-256 arms B, C and D."""
import json
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

META  = Path("/workspace/uec256_meta")
POOL  = Path("/workspace/uec256_synth"); POOL.mkdir(exist_ok=True)
CACHE = "/workspace/hf_cache"
STEPS, GUIDANCE, BATCH, SEED = 50, 2.0, 8, 42

need    = {int(k): v for k, v in json.loads((META/"pool_alloc.json").read_text()).items()}
prompts = json.loads((META/"prompts.json").read_text())

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
    safety_checker=None, cache_dir=CACHE).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

g = torch.Generator("cuda")
total = sum(need.values()); done_all = 0

for ci in sorted(need):
    n = need[ci]
    d = POOL/f"{ci:03d}"; d.mkdir(exist_ok=True)
    have = len(list(d.glob("*.jpg")))
    if have >= n:
        done_all += n
        print(f"[{ci:>3}/255] {prompts[str(ci)]['name'][:28]:<28} {have} already", flush=True)
        continue

    prompt = f"A photo of {prompts[str(ci)]['prompt']}."
    made = have
    while made < n:
        k = min(BATCH, n - made)
        g.manual_seed(SEED * 1000000 + ci * 1000 + made)
        imgs = pipe([prompt]*k, num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE, generator=g).images
        for im in imgs:
            im.resize((512, 512)).save(d/f"{made:05d}.jpg", quality=95)
            made += 1
    done_all += made
    print(f"[{ci:>3}/255] {prompts[str(ci)]['name'][:28]:<28} {made}   "
          f"({100*done_all/total:.1f}% of pool)", flush=True)

print("pool complete")
