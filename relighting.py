from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import torch
import safetensors.torch as sf

from PIL import Image
from enum import Enum

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

from briarmbg import BriaRMBG
from torch.hub import download_url_to_file


@dataclass
class RelightConfig:
    sd15_name: str = "stablediffusionapi/realistic-vision-v51"
    iclight_safetensors_path: str = "./models/iclight_sd15_fc.safetensors"
    iclight_download_url: str = "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors"
    rmbg_name: str = "briaai/RMBG-1.4"

    device: str = "cuda"
    text_dtype = torch.float16
    unet_dtype = torch.float16
    vae_dtype = torch.bfloat16
    rmbg_dtype = torch.float32

    steps: int = 25
    cfg: float = 2.0
    a_prompt: str = "best quality"
    n_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality"

    image_width: int = 512
    image_height: int = 640
    num_samples: int = 1
    highres_scale: float = 1.5
    highres_denoise: float = 0.5
    lowres_denoise: float = 0.9

    rmbg_sigma: float = 0.0

    scheduler: str = "dpmpp_2m_sde_karras"  


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


_ENGINE = None


class RelightingEngine:
    def __init__(self, cfg: RelightConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._load_all()

    def _load_all(self):
        cfg = self.cfg

        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.sd15_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.sd15_name, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(cfg.sd15_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(cfg.sd15_name, subfolder="unet")

        self.rmbg = BriaRMBG.from_pretrained(cfg.rmbg_name)

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                8,
                self.unet.conv_in.out_channels,
                self.unet.conv_in.kernel_size,
                self.unet.conv_in.stride,
                self.unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            new_conv_in.bias = self.unet.conv_in.bias
            self.unet.conv_in = new_conv_in

        self._unet_original_forward = self.unet.forward

        def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
            c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
            new_sample = torch.cat([sample, c_concat], dim=1)
            kwargs["cross_attention_kwargs"] = {}
            return self._unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

        self.unet.forward = hooked_unet_forward

        os.makedirs(os.path.dirname(cfg.iclight_safetensors_path), exist_ok=True)
        if not os.path.exists(cfg.iclight_safetensors_path):
            download_url_to_file(url=cfg.iclight_download_url, dst=cfg.iclight_safetensors_path)

        sd_offset = sf.load_file(cfg.iclight_safetensors_path)
        sd_origin = self.unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        self.unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged

        self.text_encoder = self.text_encoder.to(device=self.device, dtype=cfg.text_dtype)
        self.vae = self.vae.to(device=self.device, dtype=cfg.vae_dtype)
        self.unet = self.unet.to(device=self.device, dtype=cfg.unet_dtype)
        self.rmbg = self.rmbg.to(device=self.device, dtype=cfg.rmbg_dtype)

        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())

        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        self.euler_a_scheduler = EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
        )
        self.dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1,
        )

        scheduler = self._pick_scheduler(cfg.scheduler)

        self.t2i_pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        )
        self.i2i_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        )

        self.t2i_pipe.to(self.device)
        self.i2i_pipe.to(self.device)

    def _pick_scheduler(self, name: str):
        name = (name or "").lower()
        if name == "ddim":
            return self.ddim_scheduler
        if name in ("euler_a", "euler-ancestral"):
            return self.euler_a_scheduler
        return self.dpmpp_2m_sde_karras_scheduler

    @torch.inference_mode()
    def _encode_prompt_inner(self, txt: str):
        tokenizer = self.tokenizer
        text_encoder = self.text_encoder

        max_length = tokenizer.model_max_length
        chunk_length = tokenizer.model_max_length - 2
        id_start = tokenizer.bos_token_id
        id_end = tokenizer.eos_token_id
        id_pad = id_end

        def pad(x, p, i):
            return x[:i] if len(x) >= i else x + [p] * (i - len(x))

        tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i : i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        conds = text_encoder(token_ids).last_hidden_state
        return conds

    @torch.inference_mode()
    def encode_prompt_pair(self, positive_prompt: str, negative_prompt: str):
        c = self._encode_prompt_inner(positive_prompt)
        uc = self._encode_prompt_inner(negative_prompt)

        c_len = float(len(c))
        uc_len = float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat = int(math.ceil(max_count / c_len))
        uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)

        return c, uc

    @torch.inference_mode()
    def pytorch2numpy(self, imgs, quant=True):
        results = []
        for x in imgs:
            y = x.movedim(0, -1)
            if quant:
                y = y * 127.5 + 127.5
                y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
            else:
                y = y * 0.5 + 0.5
                y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)
            results.append(y)
        return results

    @torch.inference_mode()
    def numpy2pytorch(self, imgs: List[np.ndarray]):
        h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # 127 -> strictly 0.0
        h = h.movedim(-1, 1)
        return h

    def resize_and_center_crop(self, image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        pil_image = Image.fromarray(image)
        original_width, original_height = pil_image.size
        scale_factor = max(target_width / original_width, target_height / original_height)
        resized_width = int(round(original_width * scale_factor))
        resized_height = int(round(original_height * scale_factor))
        resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
        left = (resized_width - target_width) / 2
        top = (resized_height - target_height) / 2
        right = (resized_width + target_width) / 2
        bottom = (resized_height + target_height) / 2
        cropped_image = resized_image.crop((left, top, right, bottom))
        return np.array(cropped_image)

    def resize_without_crop(self, image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        return np.array(resized_image)

    @torch.inference_mode()
    def run_rmbg(self, img: np.ndarray, sigma: float = 0.0):
        H, W, C = img.shape
        assert C == 3
        k = (256.0 / float(H * W)) ** 0.5
        feed = self.resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
        feed = self.numpy2pytorch([feed]).to(device=self.device, dtype=torch.float32)

        alpha = self.rmbg(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)

        result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
        return result.clip(0, 255).astype(np.uint8), alpha

    def _alpha_from_subject_mask(self, subject_mask_l: Image.Image, H: int, W: int) -> np.ndarray:
        """
        subject_mask: 255=foreground(보호), 0=background
        alpha: float HWC(1ch expanded), 0~1
        """
        m = subject_mask_l.convert("L").resize((W, H), Image.NEAREST)
        a = (np.array(m).astype(np.float32) / 255.0).clip(0, 1)
        a = a[..., None]
        return a

    @torch.inference_mode()
    def _make_initial_bg(self, bg_source: BGSource, image_width: int, image_height: int) -> Optional[np.ndarray]:
        if bg_source == BGSource.NONE:
            return None

        if bg_source == BGSource.LEFT:
            gradient = np.linspace(255, 0, image_width)
            image = np.tile(gradient, (image_height, 1))
        elif bg_source == BGSource.RIGHT:
            gradient = np.linspace(0, 255, image_width)
            image = np.tile(gradient, (image_height, 1))
        elif bg_source == BGSource.TOP:
            gradient = np.linspace(255, 0, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
        elif bg_source == BGSource.BOTTOM:
            gradient = np.linspace(0, 255, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
        else:
            raise ValueError("Invalid BGSource")

        return np.stack((image,) * 3, axis=-1).astype(np.uint8)

    @torch.inference_mode()
    def process(
        self,
        input_fg: np.ndarray,
        prompt: str,
        image_width: int,
        image_height: int,
        num_samples: int,
        seed: int,
        steps: int,
        a_prompt: str,
        n_prompt: str,
        cfg: float,
        highres_scale: float,
        highres_denoise: float,
        lowres_denoise: float,
        bg_source: BGSource,
    ) -> List[np.ndarray]:

        input_bg = self._make_initial_bg(bg_source, image_width, image_height)

        rng = torch.Generator(device=self.device).manual_seed(int(seed))

        fg = self.resize_and_center_crop(input_fg, image_width, image_height)

        concat_conds = self.numpy2pytorch([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
        concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor

        conds, unconds = self.encode_prompt_pair(
            positive_prompt=prompt + ", " + a_prompt,
            negative_prompt=n_prompt,
        )

        if input_bg is None:
            latents = self.t2i_pipe(
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=steps,
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
        else:
            bg = self.resize_and_center_crop(input_bg, image_width, image_height)
            bg_latent = self.numpy2pytorch([bg]).to(device=self.vae.device, dtype=self.vae.dtype)
            bg_latent = self.vae.encode(bg_latent).latent_dist.mode() * self.vae.config.scaling_factor

            latents = self.i2i_pipe(
                image=bg_latent,
                strength=lowres_denoise,
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=int(round(steps / lowres_denoise)),
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

        pixels = self.vae.decode(latents).sample
        pixels = self.pytorch2numpy(pixels)

        pixels = [
            self.resize_without_crop(
                image=p,
                target_width=int(round(image_width * highres_scale / 64.0) * 64),
                target_height=int(round(image_height * highres_scale / 64.0) * 64),
            )
            for p in pixels
        ]

        pixels_t = self.numpy2pytorch(pixels).to(device=self.vae.device, dtype=self.vae.dtype)
        latents = self.vae.encode(pixels_t).latent_dist.mode() * self.vae.config.scaling_factor
        latents = latents.to(device=self.unet.device, dtype=self.unet.dtype)

        image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

        fg = self.resize_and_center_crop(input_fg, image_width, image_height)
        concat_conds = self.numpy2pytorch([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
        concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor

        latents = self.i2i_pipe(
            image=latents,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type="latent",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

        pixels = self.vae.decode(latents).sample
        return self.pytorch2numpy(pixels)

    @torch.inference_mode()
    def process_relight(
        self,
        input_fg: np.ndarray,
        prompt: str,
        image_width: int,
        image_height: int,
        num_samples: int,
        seed: int,
        steps: int,
        a_prompt: str,
        n_prompt: str,
        cfg: float,
        highres_scale: float,
        highres_denoise: float,
        lowres_denoise: float,
        bg_source: BGSource,
        subject_mask_l: Optional[Image.Image] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        
        H, W, _ = input_fg.shape

        if subject_mask_l is None:
            fg_processed, alpha = self.run_rmbg(input_fg, sigma=self.cfg.rmbg_sigma)
            alpha = alpha[..., None]  # H,W,1
        else:
            alpha = self._alpha_from_subject_mask(subject_mask_l, H, W)
            fg_processed = 127 + (input_fg.astype(np.float32) - 127.0) * alpha
            fg_processed = fg_processed.clip(0, 255).astype(np.uint8)

        results = self.process(
            input_fg=fg_processed,
            prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            cfg=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
            lowres_denoise=lowres_denoise,
            bg_source=bg_source,
        )
        return fg_processed, results


def _get_engine(cfg: Optional[RelightConfig] = None) -> RelightingEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RelightingEngine(cfg or RelightConfig())
    return _ENGINE


def relight_and_save(
    image: Image.Image,
    background_prompt: str,
    out_path: str,
    subject_mask: Optional[Image.Image] = None,
    # overrides (optional)
    device: str = "cuda",
    seed: int = 12345,
    image_width: int = 512,
    image_height: int = 640,
    num_samples: int = 1,
    steps: int = 25,
    cfg: float = 2.0,
    a_prompt: str = "best quality",
    n_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality",
    highres_scale: float = 1.5,
    highres_denoise: float = 0.5,
    lowres_denoise: float = 0.9,
    bg_source: str = "None",
    # model paths (optional overrides)
    sd15_name: str = "stablediffusionapi/realistic-vision-v51",
    iclight_safetensors_path: str = "./models/iclight_sd15_fc.safetensors",
    iclight_download_url: str = "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
    rmbg_name: str = "briaai/RMBG-1.4",
    scheduler: str = "dpmpp_2m_sde_karras",
    fp16: bool = True,
):

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cfg_obj = RelightConfig(
        sd15_name=sd15_name,
        iclight_safetensors_path=iclight_safetensors_path,
        iclight_download_url=iclight_download_url,
        rmbg_name=rmbg_name,
        device=device,
        steps=steps,
        cfg=cfg,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        image_width=image_width,
        image_height=image_height,
        num_samples=num_samples,
        highres_scale=highres_scale,
        highres_denoise=highres_denoise,
        lowres_denoise=lowres_denoise,
        scheduler=scheduler,
    )

    if not fp16:
        cfg_obj.text_dtype = torch.float32
        cfg_obj.unet_dtype = torch.float32
        cfg_obj.vae_dtype = torch.float32

    engine = _get_engine(cfg_obj)

    img_np = np.array(image.convert("RGB")).astype(np.uint8)

    try:
        bgsrc = BGSource(bg_source)
    except Exception:
        bgsrc = BGSource.NONE

    fg_pre, outs = engine.process_relight(
        input_fg=img_np,
        prompt=background_prompt,
        image_width=image_width,
        image_height=image_height,
        num_samples=num_samples,
        seed=seed,
        steps=steps,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        cfg=cfg,
        highres_scale=highres_scale,
        highres_denoise=highres_denoise,
        lowres_denoise=lowres_denoise,
        bg_source=bgsrc,
        subject_mask_l=subject_mask.convert("L") if subject_mask is not None else None,
    )

    if len(outs) == 0:
        raise RuntimeError("Relighting produced no outputs.")

    base, ext = os.path.splitext(out_path)
    for i, arr in enumerate(outs):
        p = out_path if i == 0 else f"{base}_{i}{ext}"
        Image.fromarray(arr).save(p)

    return out_path
