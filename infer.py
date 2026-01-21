import os
import json
import argparse
from typing import Dict, Optional

import torch
from PIL import Image

from openai import OpenAI
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

import shape_control

import relighting


def decompose_prompt_with_gpt(
    client: OpenAI,
    prompt: str,
    model: str = "gpt-4.1-mini",
) -> Dict[str, str]:
    system = (
        "Decompose the given fashion prompt into STRICT JSON with keys: "
        "shape, attributes, style, background. No extra keys."
    )
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    data = json.loads(resp.choices[0].message.content)
    for k in ["shape", "attributes", "style", "background"]:
        if k not in data or not isinstance(data[k], str):
            raise RuntimeError(f"GPT JSON missing key {k}: {data}")
    return {k: data[k].strip() for k in ["shape", "attributes", "style", "background"]}


def run_sdxl_controlnet_inpaint(
    base_model_id: str,
    controlnet_model_id: str,
    device: str,
    fp16: bool,
    input_image: Image.Image,
    inpaint_mask: Image.Image,
    depth_map: Image.Image,
    prompt: str,
    negative_prompt: Optional[str],
    steps: int,
    guidance: float,
    strength: float,
    seed: int,
) -> Image.Image:
    dtype = torch.float16 if fp16 else torch.float32

    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
    ).to(device)

    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    pipe.enable_vae_slicing()

    g = torch.Generator(device=device).manual_seed(seed)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image.convert("RGB"),
        mask_image=inpaint_mask.convert("L"),
        control_image=depth_map.convert("RGB"),  # depth map을 3채널로
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=g,
    ).images[0]

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--prompt", type=str, required=True)

    ap.add_argument("--input_file", type=str, default="../DeepFashion-MultiModal/ALL/images/01.png")
    ap.add_argument("--mask_file", type=str, default="../DeepFashion-MultiModal/ALL/mask/01.png")

    ap.add_argument("--parsing_wo_cloth", type=str, default=None)
    ap.add_argument("--densepose", type=str, default=None)

    ap.add_argument("--workdir", type=str, default="./outputs/vtonagent")
    ap.add_argument("--gpt_model", type=str, default="gpt-4.1-mini")

    ap.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--controlnet_model_id", type=str, default="diffusers/controlnet-depth-sdxl-1.0")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--negative_prompt", type=str, default="low quality, blurry, deformed, artifacts")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY 환경변수가 필요")

    os.makedirs(args.workdir, exist_ok=True)

    input_img = Image.open(args.input_file).convert("RGB")
    mask_img = Image.open(args.mask_file).convert("L")

    # 1) GPT 분해
    client = OpenAI()
    parts = decompose_prompt_with_gpt(client, args.prompt, model=args.gpt_model)
    print(json.dumps(parts, ensure_ascii=False, indent=2))

    # 2) shape_control 엔진 초기화
    shape_control.init_engine(
        config_path=args.stage1_config,
        sd15_ckpt=args.sd15_ckpt,
        stage1_pth=args.stage1_pth,
        vae_ckpt=args.vae_ckpt,
        clip_path=args.clip_path,
        H=args.shape_H,
        W=args.shape_W,
        ddim_steps=args.shape_ddim_steps,
        sampler=args.shape_sampler,
        scale=args.shape_scale,
        device=args.device,
        fp16=args.fp16,
        seed=123,
    )

    # 3) Stage1 inputs 로드 
    parsing_wo_cloth_img = None
    densepose_img = None
    if args.parsing_wo_cloth and args.densepose:
        parsing_wo_cloth_img = Image.open(args.parsing_wo_cloth).convert("RGB")
        densepose_img = Image.open(args.densepose).convert("RGB")

    # 4) shape -> depth_map 생성
    depth_map = shape_control.get_shape_depth_map(
        shape_text=parts["shape"],
        parsing_wo_cloth=parsing_wo_cloth_img,
        densepose=densepose_img,
        fallback_mask=mask_img,  # Stage1 입력 없을 때라도 파이프라인 동작하게
    )
    depth_path = os.path.join(args.workdir, "shape_depth_map.png")
    depth_map.save(depth_path)

    # 5) SDXL ControlNet Inpaint (prompt = attributes + style)
    sdxl_prompt = f"{parts['attributes']}. {parts['style']}."
    gen = run_sdxl_controlnet_inpaint(
        base_model_id=args.base_model_id,
        controlnet_model_id=args.controlnet_model_id,
        device=args.device,
        fp16=args.fp16,
        input_image=input_img,
        inpaint_mask=mask_img,
        depth_map=depth_map,
        prompt=sdxl_prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance=args.guidance,
        strength=args.strength,
        seed=args.seed,
    )
    mid_path = os.path.join(args.workdir, "sdxl_inpaint.png")
    gen.save(mid_path)

    # 6) Background stage
    final_path = os.path.join(args.workdir, "final.png")
    if hasattr(relighting, "relight_and_save"):
        relighting.relight_and_save(
            image=gen,
            background_prompt=parts["background"],
            out_path=final_path,
            subject_mask=mask_img,
            base_model_id=args.base_model_id,
            device=args.device,
            fp16=args.fp16,
            seed=args.seed + 1,
        )
    else:
        # relighting 구현이 없으면 일단 저장만
        gen.save(final_path)
        print("[WARN] relighting.relight_and_save가 없어서 배경 후처리 없이 저장했음.")

    print("\n[DONE]")
    print("depth_map:", depth_path)
    print("mid:", mid_path)
    print("final:", final_path)


if __name__ == "__main__":
    main()
