from __future__ import annotations

import math
import os

import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from vision_io import process_vision_info


class _InternVLBase:
    def build_transform(self, input_size):
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=336, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_video(self, video_path, frame_indices, input_size=448, max_num=1):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def get_answer(self, video_path, query, sorted_frame_idx):
        pixel_values, num_patches_list = self.load_video(video_path, sorted_frame_idx)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
        question = video_prefix + query
        response, _history = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self.generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
        return response

    def get_text_answer(self, query):
        response, _history = self.model.chat(
            self.tokenizer,
            None,
            query,
            self.generation_config,
            num_patches_list=None,
            history=None,
            return_history=True,
        )
        return response


class InternVL8B(_InternVLBase):
    def __init__(self):
        path = "OpenGVLab/InternVL2_5-8B"
        device_map = self.split_model("InternVL2_5-8B")
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.generation_config = dict(max_new_tokens=64, do_sample=True)

    def get_model_name(self):
        return "intern_8b"

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            "InternVL2_5-1B": 24,
            "InternVL2_5-2B": 24,
            "InternVL2_5-4B": 36,
            "InternVL2_5-8B": 32,
            "InternVL2_5-26B": 48,
            "InternVL2_5-38B": 64,
            "InternVL2_5-78B": 80,
        }[model_name]
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for _ in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
        device_map["language_model.model.rotary_emb"] = 0
        return device_map


class InternVL3538B(_InternVLBase):
    def __init__(self):
        path = "OpenGVLab/InternVL3_5-38B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.generation_config = dict(max_new_tokens=64, do_sample=True)

    def get_model_name(self):
        return "intern_3538b"


class InternVL378B(_InternVLBase):
    def __init__(self):
        path = "OpenGVLab/InternVL3-78B"
        device_map = self.split_model("OpenGVLab/InternVL3-78B")
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.generation_config = dict(max_new_tokens=64, do_sample=True)

    def get_model_name(self):
        return "intern_78b"

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        num_layers = config.llm_config.num_hidden_layers
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for _ in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.model.rotary_emb"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
        return device_map


class _Qwen25Base:
    def __init__(self, model_path: str, model_name: str, device_map):
        self._model_name = model_name
        self.processor = AutoProcessor.from_pretrained(
            model_path,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )

    def get_model_name(self):
        return self._model_name

    def get_text_answer(self, text):
        messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
        rendered_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=rendered_text, images=None, videos=None, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def get_answer(self, video_path, text, sample_idx=None, multi_image_path=None):
        if multi_image_path:
            messages = [
                {"role": "user", "content": [{"type": "image", "image": image_path} for image_path in multi_image_path]}
            ]
            messages[0]["content"].append({"type": "text", "text": text})
            images, videos = process_vision_info(messages)
        elif video_path is None:
            messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
            images, videos = None, None
        else:
            messages = [
                [{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": text}]}]
            ]
            images, videos = process_vision_info(messages, sample_idx)

        rendered_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=rendered_text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class Qwen2_5_7bAgent(_Qwen25Base):
    def __init__(self):
        super().__init__(
            model_path="Qwen/Qwen2.5-VL-7B-Instruct",
            model_name="qwen2_5_7b",
            device_map={"": 0},
        )


class Qwen2_5_72bAgent(_Qwen25Base):
    def __init__(self):
        super().__init__(
            model_path="Qwen/Qwen2.5-VL-72B-Instruct",
            model_name="qwen2_5_72b",
            device_map="auto",
        )


__all__ = [
    "InternVL8B",
    "InternVL3538B",
    "InternVL378B",
    "Qwen2_5_7bAgent",
    "Qwen2_5_72bAgent",
]
