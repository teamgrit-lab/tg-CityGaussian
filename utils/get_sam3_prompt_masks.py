import add_pypath

import argparse
import importlib
import inspect
import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import AsyncImageReader, AsyncImageSaver
from distibuted_tasks import configure_arg_parser, get_task_list_with_args


def _filter_kwargs(callable_obj, kwargs):
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    signature = inspect.signature(callable_obj)
    if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _call_with_supported_kwargs(callable_obj, kwargs):
    return callable_obj(**_filter_kwargs(callable_obj, kwargs))


def _build_sam3_model(sam3_module, args, device):
    if hasattr(sam3_module, "sam_model_registry"):
        if args.sam3_arch not in sam3_module.sam_model_registry:
            raise ValueError(
                "SAM3 arch '{}' not found in sam_model_registry. Available: {}".format(
                    args.sam3_arch,
                    list(sam3_module.sam_model_registry.keys()),
                )
            )
        model_builder = sam3_module.sam_model_registry[args.sam3_arch]
        model = _call_with_supported_kwargs(
            model_builder,
            {
                "checkpoint": args.sam3_ckpt,
                "ckpt": args.sam3_ckpt,
                "model_path": args.sam3_ckpt,
                "config": args.sam3_config,
                "model_cfg": args.sam3_config,
                "cfg": args.sam3_config,
            },
        )
    elif hasattr(sam3_module, "build_sam3"):
        model = _call_with_supported_kwargs(
            sam3_module.build_sam3,
            {
                "checkpoint": args.sam3_ckpt,
                "ckpt": args.sam3_ckpt,
                "model_path": args.sam3_ckpt,
                "config": args.sam3_config,
                "model_cfg": args.sam3_config,
                "cfg": args.sam3_config,
                "model_type": args.sam3_arch,
            },
        )
    else:
        raise RuntimeError(
            "SAM3 module must expose sam_model_registry or build_sam3()."
        )

    if hasattr(model, "to"):
        model = model.to(device)
    return model


def _build_mask_generator(sam3_module, model, args):
    generator_cls = None
    for class_name in [
        "SamAutomaticMaskGenerator",
        "SAM3AutomaticMaskGenerator",
        "AutomaticMaskGenerator",
    ]:
        if hasattr(sam3_module, class_name):
            generator_cls = getattr(sam3_module, class_name)
            break
    if generator_cls is None:
        raise RuntimeError(
            "SAM3 module does not provide an automatic mask generator class."
        )

    generator_kwargs = _filter_kwargs(
        generator_cls,
        {
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "box_nms_thresh": args.box_nms_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "crop_n_layers": args.crop_n_layers,
            "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
            "min_mask_region_area": args.min_mask_region_area,
        },
    )
    return generator_cls(model, **generator_kwargs)


def _load_clip_model(args, device):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip_torch is required for prompt-based masking. "
            "Install it with: pip install open_clip_torch"
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


def _read_prompts(args):
    prompts = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                prompts.append(line)
    prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
    if not prompts and args.mode == "text":
        raise ValueError("No prompts provided for text mode.")
    if args.prompt_template:
        if "{}" not in args.prompt_template:
            raise ValueError("prompt_template must include '{}' placeholder.")
        prompts = [args.prompt_template.format(prompt) for prompt in prompts]
    return prompts


def _expand_bbox(x0, y0, x1, y1, width, height, expand_ratio):
    if expand_ratio <= 0:
        return x0, y0, x1, y1
    box_w = x1 - x0
    box_h = y1 - y0
    pad = int(max(box_w, box_h) * expand_ratio)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)
    return x0, y0, x1, y1


def _crop_masked_region(rgb_image, mask, expand_ratio, mask_background):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    height, width = rgb_image.shape[:2]
    x0, y0, x1, y1 = _expand_bbox(x0, y0, x1, y1, width, height, expand_ratio)
    crop = rgb_image[y0:y1, x0:x1]
    if mask_background:
        mask_crop = mask[y0:y1, x0:x1]
        crop = crop.copy()
        crop[~mask_crop] = 0
    return crop


def _mask_matches_prompts(
    mask,
    rgb_image,
    clip_model,
    clip_preprocess,
    text_features,
    device,
    threshold,
    expand_ratio,
    mask_background,
):
    crop = _crop_masked_region(rgb_image, mask, expand_ratio, mask_background)
    if crop is None:
        return False, None

    image_tensor = clip_preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        similarity = image_features @ text_features.T
        best_score, best_index = similarity.max(dim=-1)
    best_score = best_score.item()
    return best_score >= threshold, (best_score, int(best_index.item()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default="semantic/masks_png")
    parser.add_argument("--sam3-ckpt", "-c", type=str, required=True)
    parser.add_argument("--sam3-arch", type=str, default="vit_h")
    parser.add_argument("--sam3-config", type=str, default=None)
    parser.add_argument("--sam3-module", type=str, default="sam3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, choices=["text", "auto"], default="text")
    parser.add_argument("--prompt", "-p", nargs="+", default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--prompt-template", type=str, default="{}")
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.95)
    parser.add_argument("--crop-expand", type=float, default=0.0)
    parser.add_argument("--mask-background", dest="mask_background", action="store_true")
    parser.add_argument("--no-mask-background", dest="mask_background", action="store_false")
    parser.set_defaults(mask_background=True)
    parser.add_argument("--invert", action="store_true", default=False)
    parser.add_argument("--preview", action="store_true", default=False)
    parser.add_argument("--preview-alpha", type=float, default=0.5)
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    parser.add_argument("--ext", "-e", nargs="+", default=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])
    configure_arg_parser(parser)
    args = parser.parse_args()

    image_path = args.image_path
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(image_path.rstrip("/")))

    mask_dir = args.mask_dir
    if not os.path.isabs(mask_dir):
        mask_dir = os.path.join(output_path, mask_dir)
    mask_preview_dir = mask_dir + "_preview"
    os.makedirs(mask_dir, exist_ok=True)
    if args.preview:
        os.makedirs(mask_preview_dir, exist_ok=True)

    prompts = _read_prompts(args)

    try:
        sam3_module = importlib.import_module(args.sam3_module)
    except ImportError as exc:
        raise ImportError(
            "Failed to import SAM3 module '{}'. "
            "Install your SAM3 package or pass --sam3-module to point to it.".format(
                args.sam3_module
            )
        ) from exc
    model = _build_sam3_model(sam3_module, args, args.device)
    mask_generator = _build_mask_generator(sam3_module, model, args)

    clip_model = None
    clip_preprocess = None
    text_features = None
    if args.mode == "text":
        clip_model, clip_preprocess, clip_tokenizer = _load_clip_model(args, args.device)
        with torch.no_grad():
            tokenized = clip_tokenizer(prompts).to(args.device)
            text_features = clip_model.encode_text(tokenized)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

    print("Finding image files...")
    image_name_set = {}
    for ext in args.ext:
        for path in glob(os.path.join(image_path, f"**/*.{ext}"), recursive=True):
            image_name_set[path] = True
    image_list = list(image_name_set.keys())
    assert len(image_list) > 0, "Not a image can be found"
    image_list.sort()

    image_list = get_task_list_with_args(args, image_list)
    print("Generating masks from {} images".format(len(image_list)))

    image_reader = AsyncImageReader(image_list)
    mask_saver = AsyncImageSaver()
    preview_saver = AsyncImageSaver(is_rgb=True)
    try:
        with tqdm(range(len(image_list))) as t:
            for _ in t:
                image_full_path, bgr = image_reader.queue.get()
                image_name = image_full_path[len(image_path):].lstrip("/")
                t.set_description("Masking: {}".format(image_name))

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                masks = mask_generator.generate(rgb)

                height, width = rgb.shape[:2]
                selected_mask = np.zeros((height, width), dtype=bool)
                for mask_info in masks:
                    segmentation = mask_info["segmentation"]
                    if isinstance(segmentation, torch.Tensor):
                        segmentation = segmentation.detach().cpu().numpy()
                    if segmentation.dtype != np.bool_:
                        segmentation = segmentation.astype(bool)
                    if segmentation.all() or (~segmentation).all():
                        continue
                    area = int(segmentation.sum())
                    if area < args.min_mask_area:
                        continue
                    area_ratio = area / float(height * width)
                    if area_ratio > args.max_mask_area_ratio:
                        continue

                    if args.mode == "auto":
                        selected_mask |= segmentation
                        continue

                    matched, _ = _mask_matches_prompts(
                        segmentation,
                        rgb,
                        clip_model,
                        clip_preprocess,
                        text_features,
                        args.device,
                        args.text_threshold,
                        args.crop_expand,
                        args.mask_background,
                    )
                    if matched:
                        selected_mask |= segmentation

                mask_output = np.zeros((height, width), dtype=np.uint8) if args.invert else np.full((height, width), 255, dtype=np.uint8)
                if args.invert:
                    mask_output[selected_mask] = 255
                else:
                    mask_output[selected_mask] = 0

                mask_path = os.path.join(mask_dir, "{}.png".format(image_name))
                mask_saver.save(mask_output, mask_path)

                if args.preview:
                    preview = rgb.astype(np.float32)
                    overlay_color = np.array([255, 0, 0], dtype=np.float32)
                    preview[selected_mask] = (
                        preview[selected_mask] * (1.0 - args.preview_alpha)
                        + overlay_color * args.preview_alpha
                    )
                    preview = np.clip(preview, 0, 255).astype(np.uint8)
                    preview_path = os.path.join(mask_preview_dir, "{}.png".format(image_name))
                    preview_saver.save(preview, preview_path)
    finally:
        image_reader.stop()
        mask_saver.stop()
        preview_saver.stop()


if __name__ == "__main__":
    main()
