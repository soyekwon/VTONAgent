from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from omegaconf import OmegaConf
from transformers import CLIPTokenizer, CLIPTextModel

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


def _seed_everything(seed: int = 123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _modify_weights(w: torch.Tensor, scale: float = 1e-8, n: int = 2) -> torch.Tensor:
    extra_w = scale * torch.randn_like(w)
    new_w = w.clone()
    for _ in range(n):
        new_w = torch.cat((new_w, extra_w.clone()), dim=1)
    return new_w


def _pil_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def _pil_l(img: Image.Image) -> Image.Image:
    return img.convert("L") if img.mode != "L" else img


def _resize_pil(img: Image.Image, size_hw: Tuple[int, int], mode: str = "bilinear") -> Image.Image:
    # size_hw=(H,W)
    H, W = size_hw
    resample = Image.BILINEAR if mode == "bilinear" else Image.NEAREST
    return img.resize((W, H), resample=resample)


def _img_to_tensor_minus1_1(img: Image.Image) -> torch.Tensor:
    arr = np.array(_pil_rgb(img)).astype(np.float32) / 255.0  # [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # 1,3,H,W
    t = t * 2.0 - 1.0
    return t


def _tensor_to_pil_0_255(x: torch.Tensor) -> Image.Image:
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().cpu()
    if x.min() < -0.1:  # assume [-1,1]
        x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _depth_from_mask(mask_l: Image.Image) -> Image.Image:
    import cv2
    m = np.array(_pil_l(mask_l)).astype(np.uint8)
    bin_m = (m > 127).astype(np.uint8)
    dist = cv2.distanceTransform(bin_m, distanceType=cv2.DIST_L2, maskSize=5)
    if dist.max() < 1e-6:
        out = np.zeros_like(m)
    else:
        dist = dist / (dist.max() + 1e-8)
        out = (dist * 255.0).astype(np.uint8)
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return Image.fromarray(out, mode="L")


def _depth_from_parsing_rgb(parsing_rgb: Image.Image) -> Image.Image:
    g = np.array(parsing_rgb.convert("L")).astype(np.float32)
    mn, mx = float(g.min()), float(g.max())
    if mx - mn < 1e-6:
        out = np.zeros_like(g, dtype=np.uint8)
    else:
        out = ((g - mn) / (mx - mn) * 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="L")

@dataclass
class ShapeControlConfig:
    config_path: str = "configs/Stage1_text_to_parsing.yaml"
    sd15_ckpt: str = "./../pretrain_models/v1-5-pruned-emaonly.ckpt"
    stage1_pth: str = "../pretrain_models/Stage1/model_15000.pth"
    vae_ckpt: str = "./../pretrain_models/vae-ft-mse.ckpt"
    clip_path: str = "./../pretrain_models/clip-vit-large-patch14"

    H: int = 1024
    W: int = 512
    C: int = 4
    f: int = 8

    ddim_steps: int = 50
    ddim_eta: float = 0.0
    scale: float = 1.0  # unconditional guidance scale

    sampler: str = "ddim"  # "ddim" | "plms" | "dpm"
    seed: int = 123

    device: str = "cuda"
    fp16: bool = True


class ShapeControlEngine:
    def __init__(self, cfg: ShapeControlConfig):
        self.cfg = cfg
        _seed_everything(cfg.seed)

        self.device = torch.device(cfg.device)
        self.dtype = torch.float16 if cfg.fp16 else torch.float32

        self._load_all()

    def _load_model_from_config(self, config, ckpt: str, verbose: bool = False):
        print(f"[shape_control] Loading base model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd

        model = instantiate_from_config(config.model)

        # 원본 코드: input weight 수정
        k = "model.diffusion_model.input_blocks.0.0.weight"
        if k in sd:
            print("[shape_control] modifying input weights for compatibility")
            sd[k] = _modify_weights(sd[k], scale=1e-8, n=2)

        m, u = model.load_state_dict(sd, strict=False)
        if verbose and len(m) > 0:
            print("[shape_control] missing keys:", m)
        if verbose and len(u) > 0:
            print("[shape_control] unexpected keys:", u)

        model = model.to(self.device)
        model.eval()
        return model

    def _load_all(self):
        cfg = self.cfg
        config = OmegaConf.load(cfg.config_path)

        self.model = self._load_model_from_config(config, cfg.sd15_ckpt, verbose=False)
        # stage1 가중치 로드
        print(f"[shape_control] Loading Stage1 weights from {cfg.stage1_pth}")
        self.model.load_state_dict(torch.load(cfg.stage1_pth, map_location="cpu"), strict=False)

        # VAE init
        print(f"[shape_control] Init first-stage VAE from {cfg.vae_ckpt}")
        self.model.first_stage_model.init_from_ckpt(cfg.vae_ckpt)

        # CLIP
        print(f"[shape_control] Loading CLIP from {cfg.clip_path}")
        self.clip_model = CLIPTextModel.from_pretrained(cfg.clip_path).to(self.device)
        self.clip_model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_path)

        # Sampler
        if cfg.sampler.lower() == "dpm":
            self.sampler = DPMSolverSampler(self.model)
        elif cfg.sampler.lower() == "plms":
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)

        print("[shape_control] Engine ready.")

    @torch.no_grad()
    def infer_parsing_rgb(
        self,
        text_description: str,
        parsing_wo_cloth: Image.Image,
        densepose: Image.Image,
    ) -> Image.Image:
        
        cfg = self.cfg

        # resize to expected
        parsing_wo_cloth = _resize_pil(_pil_rgb(parsing_wo_cloth), (cfg.H, cfg.W), mode="bilinear")
        densepose = _resize_pil(_pil_rgb(densepose), (cfg.H, cfg.W), mode="bilinear")

        # tokenize
        text = self.tokenizer(
            [text_description],
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = text["input_ids"].to(self.device)

        text_features = self.clip_model(input_ids).last_hidden_state  # [1,77,dim]

        # encode parsing_wo_cloth + densepose to latents
        p = _img_to_tensor_minus1_1(parsing_wo_cloth).to(self.device, dtype=torch.float32)
        d = _img_to_tensor_minus1_1(densepose).to(self.device, dtype=torch.float32)

        # model.encode_first_stage expects [-1,1] typically
        p_lat = self.model.get_first_stage_encoding(self.model.encode_first_stage(p))
        d_lat = self.model.get_first_stage_encoding(self.model.encode_first_stage(d))
        concat_feature = torch.cat([p_lat, d_lat], dim=1)

        c = [concat_feature, text_features]
        shape = [cfg.C, cfg.H // cfg.f, cfg.W // cfg.f]

        samples, _ = self.sampler.sample(
            S=cfg.ddim_steps,
            conditioning=c,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=cfg.scale,
            unconditional_conditioning=None,
            eta=cfg.ddim_eta,
            x_T=None,
            features_adapter=None,
        )

        x = self.model.decode_first_stage(samples)
        # 원본 코드와 동일하게 [0,1]로
        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        out_pil = _tensor_to_pil_0_255(x)
        return out_pil

    def infer_depth_map(
        self,
        shape_text: str,
        parsing_wo_cloth: Optional[Image.Image],
        densepose: Optional[Image.Image],
        fallback_mask: Optional[Image.Image] = None,
    ) -> Image.Image:

        if parsing_wo_cloth is not None and densepose is not None:
            parsing_rgb = self.infer_parsing_rgb(shape_text, parsing_wo_cloth, densepose)
            return _depth_from_parsing_rgb(parsing_rgb)

        if fallback_mask is None:
            return Image.fromarray(np.zeros((self.cfg.H, self.cfg.W), dtype=np.uint8), mode="L")

        return _depth_from_mask(fallback_mask)

_ENGINE: Optional[ShapeControlEngine] = None


def init_engine(
    config_path: str = "configs/Stage1_text_to_parsing.yaml",
    sd15_ckpt: str = "./pretrain_models/v1-5-pruned-emaonly.ckpt",
    stage1_pth: str = "./pretrain_models/Stage1/model_15000.pth",
    vae_ckpt: str = "./pretrain_models/vae-ft-mse.ckpt",
    clip_path: str = "./pretrain_models/clip-vit-large-patch14",
    H: int = 1024,
    W: int = 512,
    C: int = 4,
    f: int = 8,
    ddim_steps: int = 50,
    ddim_eta: float = 0.0,
    scale: float = 1.0,
    sampler: str = "ddim",
    seed: int = 123,
    device: str = "cuda",
    fp16: bool = True,
) -> ShapeControlEngine:
    global _ENGINE
    cfg = ShapeControlConfig(
        config_path=config_path,
        sd15_ckpt=sd15_ckpt,
        stage1_pth=stage1_pth,
        vae_ckpt=vae_ckpt,
        clip_path=clip_path,
        H=H, W=W, C=C, f=f,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        scale=scale,
        sampler=sampler,
        seed=seed,
        device=device,
        fp16=fp16,
    )
    _ENGINE = ShapeControlEngine(cfg)
    return _ENGINE


def get_engine() -> ShapeControlEngine:
    global _ENGINE
    if _ENGINE is None:
        # 기본값으로 초기화
        _ENGINE = ShapeControlEngine(ShapeControlConfig())
    return _ENGINE

def get_shape_depth_map(
    shape_text: str,
    parsing_wo_cloth: Optional[Image.Image] = None,
    densepose: Optional[Image.Image] = None,
    fallback_mask: Optional[Image.Image] = None,
) -> Image.Image:
    eng = get_engine()
    return eng.infer_depth_map(
        shape_text=shape_text,
        parsing_wo_cloth=parsing_wo_cloth,
        densepose=densepose,
        fallback_mask=fallback_mask,
    )


def get_shape_mask(
    shape_text: str,
    parsing_wo_cloth: Optional[Image.Image] = None,
    densepose: Optional[Image.Image] = None,
    fallback_mask: Optional[Image.Image] = None,
) -> Image.Image:

    return get_shape_depth_map(shape_text, parsing_wo_cloth, densepose, fallback_mask)
