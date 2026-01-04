from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
)
import torch
from PIL import Image, ImageOps
import numpy as np
import imageio
from pathlib import Path
import cv2
from transformers import pipeline as hf_pipeline
import os
import re

from openai import OpenAI
from pydantic import BaseModel, Field


BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_MODEL_ID = "diffusers/controlnet-depth-sdxl-1.0"
DEVICE = "cuda"


controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL_ID,
    torch_dtype=torch.float16,
)

pipe_shape_attr = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    BASE_MODEL_ID,
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to(DEVICE)

_pipe_background = None

depth_estimator = hf_pipeline("depth-estimation")


class PromptParts(BaseModel):
    shape: str = Field(default="")
    attributes: str = Field(default="")
    style: str = Field(default="")
    background: str = Field(default="")


def _lazy_background_pipe():
    global _pipe_background
    if _pipe_background is None:
        _pipe_background = StableDiffusionXLInpaintPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
        ).to(DEVICE)
    return _pipe_background


def _join_nonempty(*items: str) -> str:
    out = []
    for x in items:
        x = (x or "").strip()
        if x:
            out.append(x)
    return ", ".join(out).strip()


def load_prompts(prompts_path: str) -> list[str]:
    p = Path(prompts_path)
    if not p.exists():
        raise FileNotFoundError(f"prompts file not found: {p}")

    text = p.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n+", text.strip(), flags=re.MULTILINE)
    prompts = []
    for b in blocks:
        s = " ".join([ln.strip() for ln in b.splitlines() if ln.strip()]).strip()
        if s:
            prompts.append(s)
    return prompts


def make_depth_condition_from_mask(depth_mask_path: str) -> Image.Image:
    depth = Image.open(depth_mask_path).convert("RGB").resize((698, 1024))
    inverted_image = ImageOps.invert(depth)
    depth_result = depth_estimator(inverted_image)
    image_depth = depth_result["depth"]
    if image_depth.size != (698, 1024):
        image_depth = image_depth.resize((698, 1024), Image.BILINEAR)
    return image_depth


def decompose_prompt_with_gpt(prompt: str, model: str | None = None) -> PromptParts:
    prompt = (prompt or "").strip()
    if not prompt:
        return PromptParts()

    model = model or os.getenv("PROMPT_DECOMPOSE_MODEL", "gpt-4o-2024-08-06")
    client = OpenAI()

    system = (
        "You are a prompt decomposition agent for virtual try-on.\n"
        "Decompose the user's long clothing prompt into 4 independent slots:\n"
        "1) shape: garment type + silhouette + sleeve/length/fit\n"
        "2) attributes: material/pattern/structure/details\n"
        "3) style: aesthetic adjectives/mood\n"
        "4) background: scene/location/lighting/background cues\n"
        "Rules: If a slot is not mentioned, return an empty string. Keep each slot concise."
    )

    try:
        resp = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            text_format=PromptParts,
        )
        parts: PromptParts = resp.output_parsed
        return PromptParts(
            shape=(parts.shape or "").strip(),
            attributes=(parts.attributes or "").strip(),
            style=(parts.style or "").strip(),
            background=(parts.background or "").strip(),
        )
    except Exception:
        return PromptParts(shape="", attributes=prompt, style="", background="")


def run_batch(input_dir, mask_dir, depth_dir, output_dir, prompts_path):
    input_dir = Path(input_dir)
    mask_dir = Path(mask_dir)
    depth_dir = Path(depth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.jpg"))
    mask_files = sorted(mask_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png"))

    if len(input_files) != len(mask_files) or len(input_files) != len(depth_files):
        raise RuntimeError(
            f"count mismatch: input={len(input_files)}, mask={len(mask_files)}, depth={len(depth_files)}"
        )

    prompts = load_prompts(prompts_path)
    if len(prompts) < len(input_files):
        raise RuntimeError(
            f"prompts.txt has fewer prompts than images: prompts={len(prompts)}, images={len(input_files)}"
        )

    debug = os.getenv("DEBUG_AGENT", "0").strip() in ("1", "true", "True", "YES", "yes")

    for i, (inp, msk, dep) in enumerate(zip(input_files, mask_files, depth_files)):
        prompt = prompts[i].strip()
        if not prompt:
            raise RuntimeError(f"empty prompt at index {i+1} (line/block {i+1})")

        init_image = Image.open(inp).convert("RGB").resize((698, 1024))

        mask_cv = cv2.imread(str(msk))
        if mask_cv is None:
            raise RuntimeError(f"failed to read mask: {msk}")
        mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2RGB)
        mask_cv = cv2.resize(mask_cv, (698, 1024), interpolation=cv2.INTER_AREA)

        kernel = np.ones((17, 17), np.uint8)
        mask_dilated = cv2.dilate(mask_cv, kernel, iterations=1)
        mask_image = Image.fromarray(mask_dilated)

        control_image = make_depth_condition_from_mask(str(dep))

        parts = decompose_prompt_with_gpt(prompt)

        if debug:
            print(f"[{i+1}/{len(input_files)}] {inp.name}")
            print("shape:", parts.shape)
            print("attributes:", parts.attributes)
            print("style:", parts.style)
            print("background:", parts.background)

        stage1_prompt = _join_nonempty(parts.shape, parts.attributes) or prompt
        out1 = pipe_shape_attr(
            prompt=stage1_prompt,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            num_inference_steps=40,
            guidance_scale=7.5,
            num_images_per_prompt=1,
            controlnet_conditioning_scale=0.7,
        )
        cur = out1.images[0]

        if (parts.style or "").strip():
            out2 = pipe_shape_attr(
                prompt=parts.style,
                image=cur,
                mask_image=mask_image,
                control_image=control_image,
                num_inference_steps=25,
                guidance_scale=6.5,
                num_images_per_prompt=1,
                controlnet_conditioning_scale=0.35,
            )
            cur = out2.images[0]

        if (parts.background or "").strip():
            bg_pipe = _lazy_background_pipe()
            out3 = bg_pipe(
                prompt=parts.background,
                image=cur,
                mask_image=mask_image,
                num_inference_steps=30,
                guidance_scale=6.5,
                num_images_per_prompt=1,
            )
            cur = out3.images[0]

        save_path = output_dir / f"{inp.stem}_result.png"
        imageio.imsave(save_path, np.asarray(cur))


if __name__ == "__main__":
    input_dir = "../DeepFashion-MultiModal/ALL/images"
    mask_dir = "../DeepFashion-MultiModal/ALL/mask"
    depth_dir = "../DeepFashion-MultiModal/ALL/depth"
    output_dir = "./results"
    prompts_path = "./prompts.txt"

    run_batch(input_dir, mask_dir, depth_dir, output_dir, prompts_path)
