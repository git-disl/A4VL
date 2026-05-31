from __future__ import annotations

import copy
import json
import os
import random
import re
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from tqdm import tqdm

from cutting_points import get_cutting_points
from model_backends import InternVL3538B, InternVL378B, Qwen2_5_72bAgent
from modules.modeling import CLIP4Clip
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

SPECIAL_TOKEN = {
    "CLS_TOKEN": "<|startoftext|>",
    "SEP_TOKEN": "<|endoftext|>",
    "MASK_TOKEN": "[MASK]",
    "UNK_TOKEN": "[UNK]",
    "PAD_TOKEN": "[PAD]",
}

N_CHUNKS = 6
ALPHA = 0

CLIP_TASK_ARGS = {
    "video_dim": 1024,
    "max_words": 60,
    "max_frames": 16,
    "feature_framerate": 1,
    "margin": 0.1,
    "hard_negative_rate": 0.5,
    "negative_weighting": 1,
    "n_pair": 1,
    "text_num_hidden_layers": 16,
    "visual_num_hidden_layers": 16,
    "cross_num_hidden_layers": 4,
    "linear_patch": "2d",
    "sim_header": "seqTransf",
}


@dataclass
class PipelineConfig:
    name: str
    video_root: str
    result_path: str
    action_frame_count: int
    rationale_frame_count: int
    preview_bins: int
    priority_order: list[str]
    valid_choices: set[str]
    build_prompts: Callable[[dict], tuple[str, str, str]]
    build_history_prompt: Callable[[dict, str, str], str]
    load_annotations: Callable[[str], list[dict]]
    use_cutpoints_for_whole_sampling: bool
    allow_segment_sampling: bool
    clip_score_subsample: int | None = None
    force_global_context_keywords: tuple[str, ...] = ()
    strict_choice_check: bool = False
    agent_order: tuple[str, ...] = ("intern_3538b", "intern_78b", "qwen2_5_72b")
    reuse_block_sampling_for_intern_78b_only: bool = True


def _build_agent_pool() -> dict[str, object]:
    return {
        "intern_3538b": InternVL3538B(),
        "intern_78b": InternVL378B(),
        "qwen2_5_72b": Qwen2_5_72bAgent(),
    }


class AspClipSelector:
    def __init__(self, model_path: str = "pytorch_model_0.0011.bin.25"):
        model_state_dict = torch.load(model_path, map_location="cpu")
        self.model = CLIP4Clip.from_pretrained(
            "cross-base",
            cache_dir="",
            state_dict=model_state_dict,
            task_config=CLIP_TASK_ARGS,
        )
        self.model.to("cuda")
        self.tokenizer = ClipTokenizer()

    def _tokenize_text(self, video_id: str, sentence: str):
        pairs_text = np.zeros((1, 77), dtype=np.int64)
        pairs_mask = np.zeros((1, 77), dtype=np.int64)
        pairs_segment = np.zeros((1, 77), dtype=np.int64)

        words = self.tokenizer.tokenize(sentence)
        words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
        if len(words) > 76:
            words = words[:76]
        words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < 77:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)
        pairs_segment[0] = np.array(segment_ids)
        return (
            torch.Tensor(pairs_text).cuda().long(),
            torch.Tensor(pairs_mask).cuda().long(),
            torch.Tensor(pairs_segment).cuda().long(),
        )

    def _extract_frames(self, video_path: str, frame_indices: list[int]):
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(frame_indices).asnumpy()
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )
        return torch.stack([transform(frame) for frame in frames])

    def score(self, video_path: str, text: str, sample_idx: list[int]) -> float:
        input_ids, input_mask, segment_ids = self._tokenize_text(video_path, text)
        video = self._extract_frames(video_path, sample_idx).cuda()
        video_mask = torch.Tensor([[1] * 16]).cuda()
        visual_output = self.model.get_visual_output(video, video_mask=video_mask, shaped=True, video_frame=16)
        text_feat = self.model.get_sequence_output(input_ids, segment_ids, input_mask, shaped=True)
        logits, *_tmp = self.model.get_similarity_logits(
            text_feat,
            visual_output,
            input_mask,
            video_mask,
            loose_type=True,
            eval="myeval",
        )
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().float()
            if logits.numel() == 1:
                return float(logits.item())
            return float(logits.mean().item())
        return float(logits)


def _sec_to_frame_idx(sec: float, fps: float, total_frames: int) -> int:
    return int(max(0, min(total_frames - 1, round(sec * max(fps, 1e-6)))))


def _uniform_indices(fs: int, fe: int, k: int) -> list[int]:
    if k <= 0 or fe <= fs:
        return []
    idxs = []
    bin_size = (fe - fs) // k
    for i in range(k):
        start = fs + i * bin_size
        end = fs + (i + 1) * bin_size if i < k - 1 else fe
        if start >= end:
            continue
        idxs.append(random.randint(start, end - 1))
    return sorted(idxs)


def _sample_frame_indices(
    video_path: str,
    sample_frame: int,
    cutting_points: list[float],
    use_cutpoints_for_whole_sampling: bool,
    allow_segment_sampling: bool,
    round_id: int = 0,
    judge_whole: bool = False,
):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    if len(vr) <= 96 or judge_whole:
        if not use_cutpoints_for_whole_sampling:
            return sorted(random.sample(range(len(vr)), sample_frame))

        fps = float(vr.get_avg_fps())
        duration = len(vr) / max(fps, 1e-6)
        boundaries = [0.0] + sorted({c for c in cutting_points if 0.0 < c < duration}) + [duration]
        n_local_chunks = max(1, len(boundaries) - 1)

        sorted_frame_idx = []
        for i in range(n_local_chunks):
            s_sec, e_sec = boundaries[i], boundaries[i + 1]
            fs = _sec_to_frame_idx(s_sec, fps, len(vr))
            fe = _sec_to_frame_idx(e_sec, fps, len(vr)) + 1
            fe = max(fe, fs + 1)
            sorted_frame_idx.extend(_uniform_indices(fs, fe, max(sample_frame // n_local_chunks, 1)))

        if len(sorted_frame_idx) < sample_frame:
            sorted_frame_idx.extend(random.sample(range(len(vr)), sample_frame - len(sorted_frame_idx)))
        return sorted(sorted_frame_idx)

    if not allow_segment_sampling:
        raise NotImplementedError

    inter = len(vr) // N_CHUNKS
    if 1 <= round_id <= N_CHUNKS:
        sp = round_id * inter
        try:
            sorted_frame_idx = random.sample(range(sp - inter, sp - 1), sample_frame)
        except Exception:
            sorted_frame_idx = random.sample(range(len(vr)), sample_frame)
    else:
        sorted_frame_idx = random.sample(range(len(vr)), sample_frame)
    return sorted(sorted_frame_idx)


def _select_clip_frames(
    clip_selector: AspClipSelector,
    video_path: str,
    perception_clue: str,
    question: str,
    cutting_points: list[float],
    action_frame_count: int,
    prior_block_samples: dict | None,
    clip_score_subsample: int | None,
):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    video_secs = total_frames / max(fps, 1e-6)

    cut_secs = sorted({c for c in cutting_points if 0.0 < c < video_secs})
    boundaries = [0.0] + cut_secs + [video_secs]
    n_local_chunks = max(1, len(boundaries) - 1)

    selected_action_frames = []
    best_block_similarity = -1e9
    selected_block_id = 0
    block_frame_samples = []
    block_similarity_scores = []
    block_frame_ranges = []

    for i in range(n_local_chunks):
        s_sec, e_sec = boundaries[i], boundaries[i + 1]
        fs = _sec_to_frame_idx(s_sec, fps, total_frames)
        fe = _sec_to_frame_idx(e_sec, fps, total_frames) + 1
        fe = max(fe, fs + 1)
        block_frame_ranges.append((fs, fe))

        if prior_block_samples:
            sorted_frame_idx = prior_block_samples["all_samp"][i]
        else:
            sorted_frame_idx = _uniform_indices(fs, fe, action_frame_count)

        if not sorted_frame_idx:
            block_frame_samples.append([])
            block_similarity_scores.append(-1e9)
            continue

        scoring_idx = sorted_frame_idx
        if clip_score_subsample is not None:
            scoring_idx = sorted(random.sample(sorted_frame_idx, min(clip_score_subsample, len(sorted_frame_idx))))

        s1 = clip_selector.score(video_path, perception_clue, scoring_idx)
        s2 = 0.0
        if ALPHA != 0:
            s2 = clip_selector.score(video_path, question, scoring_idx)
        block_similarity = s1 + ALPHA * s2
        block_similarity_scores.append(block_similarity)

        if block_similarity > best_block_similarity:
            best_block_similarity = block_similarity
            selected_action_frames = sorted_frame_idx
            selected_block_id = i + 1

        block_frame_samples.append(sorted_frame_idx)

    print("Clip scores per block:", block_similarity_scores)
    high_confidence_blocks = sorted(
        [i for i, score in enumerate(block_similarity_scores) if score > 0.8],
        key=lambda x: -block_similarity_scores[x],
    )
    if len(high_confidence_blocks) > 0 and action_frame_count > 0:
        good_scores = np.array([block_similarity_scores[i] for i in high_confidence_blocks], dtype=float)
        good_scores = good_scores - good_scores.max()
        probs = np.exp(good_scores)
        probs = probs / max(probs.sum(), 1e-12)
        print("probs for high-confidence blocks:", probs.tolist())

        if len(high_confidence_blocks) >= action_frame_count:
            top_order = np.argsort(-np.array([block_similarity_scores[i] for i in high_confidence_blocks]))[
                :action_frame_count
            ]
            allocated_frames = {high_confidence_blocks[int(t)]: 1 for t in top_order}
        else:
            allocated_frames = {idx: 0 for idx in high_confidence_blocks}
            raw = probs * action_frame_count
            floors = np.floor(raw).astype(int)
            for idx, add in zip(high_confidence_blocks, floors):
                allocated_frames[idx] += int(add)
            leftover = action_frame_count - int(floors.sum())
            if leftover > 0:
                remainders = raw - floors
                order = np.argsort(-remainders)
                for j in range(int(leftover)):
                    allocated_frames[high_confidence_blocks[int(order[j])]] += 1

        print("Final allocation of frames per high-confidence block:", allocated_frames)
        final_indices = []
        for block_id in allocated_frames:
            frames_to_draw = int(allocated_frames[block_id])
            fs, fe = block_frame_ranges[block_id]
            if frames_to_draw > 0:
                final_indices.extend(_uniform_indices(fs, fe, frames_to_draw))
        selected_action_frames = sorted(final_indices)

    return selected_action_frames, selected_block_id, block_frame_samples


def _parse_json(text):
    if isinstance(text, list):
        text = text[0]
    text = re.sub(r"[\n\t]", "", text)
    text = text.replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for match in re.findall(r"\{.*?\}|\[.*?\]", text, re.DOTALL):
            try:
                return json.loads(match.replace("'", '"'))
            except json.JSONDecodeError:
                continue
        print("No valid JSON found in the text.")
        return None


def _prune_agent_pool(priority_order: list[str], cross_review_scores: dict):
    aggregate_scores = {}
    for agent_name in cross_review_scores.keys():
        total_score = 0
        for reviewer_scores in cross_review_scores.values():
            if agent_name in reviewer_scores:
                total_score += int(reviewer_scores[agent_name])
        aggregate_scores[agent_name] = total_score

    min_score = min(aggregate_scores.values())
    lowest_scoring_agents = [key for key, score in aggregate_scores.items() if score == min_score]
    if len(lowest_scoring_agents) > 1:
        for priority_key in priority_order:
            if priority_key in lowest_scoring_agents:
                pruned_agent_name = priority_key
                break
    else:
        pruned_agent_name = lowest_scoring_agents[0]

    surviving_agent_names = [key for key in cross_review_scores.keys() if key != pruned_agent_name]
    return surviving_agent_names, pruned_agent_name, aggregate_scores


def _majority_if_any(answer_set: set[str], answer_dict: dict[str, str], expected_count: int):
    values = list(answer_dict.values())
    for ans in answer_set:
        if values.count(ans) == expected_count:
            return ans
    return None


def _candidate_text(anno: dict, answer_char: str):
    if answer_char not in anno.get("valid_choices", []):
        return "Invalid Answer"
    idx = ord(answer_char) - ord("A")
    if idx < 0 or idx >= len(anno["candidates"]):
        return "Invalid Answer"
    return anno["candidates"][idx]


def _run_cross_review(agent_set, anno, answer_dict, rationale_dict):
    agent_names = [agent.get_model_name() for agent in agent_set]
    peer_agent_names = copy.deepcopy(agent_names)
    for agent_name in rationale_dict.keys():
        if isinstance(rationale_dict[agent_name], list):
            rationale_dict[agent_name] = rationale_dict[agent_name][0]
    cross_review_scores = {}

    def process_agent(agent):
        agent_name = agent.get_model_name()
        local_peer_names = copy.deepcopy(peer_agent_names)
        local_peer_names.remove(agent_name)

        if len(local_peer_names) == 1:
            answer_format = {agent_name: "<1-10>", local_peer_names[0]: "<1-10>"}
            discuss_prompt = f"""Given the answers and the reasoning for judgment from this model and two other models, please rate this model and the other two models. The score ranges from 1-10. Output in dictionary format.
            The question is: {anno['question']}, 
            The answer of this model is {answer_dict[agent_name]}, the reason is {rationale_dict[agent_name]}.
            The answer of {local_peer_names[0]} model is {answer_dict[local_peer_names[0]]}, the reason is {rationale_dict[local_peer_names[0]]}.
            You do not need to explain your answer, just give me scores as your answer following the answer_format.
            Please strictly follow the answer format! The answer_format is:
            {answer_format}
            """
        else:
            answer_format = {
                agent_name: "<1-10>",
                local_peer_names[0]: "<1-10>",
                local_peer_names[1]: "<1-10>",
            }
            discuss_prompt = f"""Given the answers and the reasoning for judgment from this model and two other models. 
            The question is: {anno['question']}
            The answer of this model is {answer_dict[agent_name]}, the reason is {rationale_dict[agent_name]}.
            The answer of {local_peer_names[0]} model is {answer_dict[local_peer_names[0]]}, the reason is {rationale_dict[local_peer_names[0]]}.
            The answer of {local_peer_names[1]} model is {answer_dict[local_peer_names[1]]}, the reason is {rationale_dict[local_peer_names[1]]}.
            Please score the performance of this model an other two models base on their reasoning. The score ranges from 1-10. Output in dict format.
            You do not need to explain your answer, just give me scores as your answer following the answer_format.
            Please strictly follow the answer format! The answer_format is:
            {answer_format}
            """

        is_valid = False
        while not is_valid:
            score_dict = _parse_json(agent.get_text_answer(discuss_prompt))
            try:
                for key, value in score_dict.items():
                    try:
                        score_dict[key] = int(value)
                    except Exception:
                        score_dict[key] = "no_val"
                for value in score_dict.values():
                    if not isinstance(value, int) or value < 1 or value > 10:
                        raise ValueError("Score out of range")
                is_valid = True
            except Exception:
                is_valid = False
                print("Invalid JSON format or values.")
                print(score_dict)
        cross_review_scores[agent_name] = score_dict

    threads = [threading.Thread(target=process_agent, args=(agent,)) for agent in agent_set]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return cross_review_scores


def run_pipeline(config: PipelineConfig):
    result_anno = config.load_annotations(config.result_path)
    for anno in result_anno:
        anno["valid_choices"] = sorted(config.valid_choices)

    clip_selector = AspClipSelector()
    agent_pool = _build_agent_pool()
    agent_set_ori = [agent_pool[name] for name in config.agent_order]

    all_num = 0
    correct = 0

    for _, anno in tqdm(enumerate(result_anno), desc=f"{config.name}: processing items"):
        if "final_answer" in anno:
            continue

        agent_set = agent_set_ori
        all_num += 1
        print(anno["video_path"])
        print(anno["question"])
        print(anno["candidates"])

        video_path = os.path.join(config.video_root, anno["video_path"])
        cutting_points = get_cutting_points(video_path)
        print(cutting_points)
        anno["cutting_points"] = cutting_points

        for agent in agent_set:
            agent_name = agent.get_model_name()
            video = VideoReader(video_path)
            frame_count = len(video)
            if agent_name not in anno:
                anno[agent_name] = {}
            anno[agent_name]["preview_frames"] = []
            bin_size = frame_count // config.preview_bins
            for i in range(config.preview_bins):
                start = i * bin_size
                end = (i + 1) * bin_size if i < config.preview_bins - 1 else frame_count
                anno[agent_name]["preview_frames"].extend(random.sample(range(start, end), 1))
            anno[agent_name]["watch_samp"] = anno[agent_name]["preview_frames"]

        first_answer_set, first_answer_dict, first_sample_dict = _run_perception_action_round(
            agent_set=agent_set,
            anno=anno,
            config=config,
            clip_selector=clip_selector,
            perception_clue_updates=None,
            is_first_round=True,
        )
        anno["first_samp"] = first_sample_dict
        print(chr(ord("A") + anno["correct_choice"]))

        consensus = _majority_if_any(first_answer_set, first_answer_dict, expected_count=3)
        if isinstance(consensus, str):
            anno["first_round"] = {"answer_dict": first_answer_dict}
            anno["final_answer"] = str(consensus)
            if anno["final_answer"] == chr(ord("A") + anno["correct_choice"]):
                correct += 1
            print(f"first round, correct{correct}, all_num {all_num}, {correct / all_num}")
            with open(config.result_path, "w") as f:
                json.dump(result_anno, f)
            continue

        if "first_round" in anno:
            first_round_state = anno["first_round"]
            perception_clue_updates = first_round_state.get("perception_clue_updates", first_round_state.get("history_info"))
            rationale_dict = first_round_state.get("rationale_dict", first_round_state.get("reason_dict"))
            cross_review_scores = _run_cross_review(agent_set, anno, first_answer_dict, rationale_dict)
            anno["first_round"]["cross_review_scores"] = cross_review_scores
            anno["first_round"]["discuss_dict"] = cross_review_scores
            surviving_agent_names, pruned_agent_name, aggregate_scores = _prune_agent_pool(config.priority_order, cross_review_scores)
        else:
            rationale_dict = _collect_action_rationales(agent_set, anno, first_answer_dict, config)
            success = False
            for _retry in range(3):
                cross_review_scores = _run_cross_review(agent_set, anno, first_answer_dict, rationale_dict)
                try:
                    surviving_agent_names, pruned_agent_name, aggregate_scores = _prune_agent_pool(config.priority_order, cross_review_scores)
                    success = True
                    break
                except Exception as e:
                    print(f"Error in _prune_agent_pool: {e}")
                    continue
            if not success:
                raise RuntimeError("Failed to process agent back after multiple retries.")

            perception_clue_updates = _refine_perception_clues(
                agent_set,
                anno,
                surviving_agent_names,
                pruned_agent_name,
                aggregate_scores,
                rationale_dict,
                first_answer_dict,
                config,
            )
            anno["first_round"] = {
                "answer_dict": first_answer_dict,
                "scores": aggregate_scores,
                "aggregate_scores": aggregate_scores,
                "reason_dict": rationale_dict,
                "rationale_dict": rationale_dict,
                "discuss_dict": cross_review_scores,
                "cross_review_scores": cross_review_scores,
                "history_info": perception_clue_updates,
                "perception_clue_updates": perception_clue_updates,
            }

        agent_set = [agent for agent in agent_set if agent.get_model_name() in surviving_agent_names]
        second_answer_set, second_answer_dict, second_sample_dict = _run_perception_action_round(
            agent_set=agent_set,
            anno=anno,
            config=config,
            clip_selector=clip_selector,
            perception_clue_updates=perception_clue_updates,
            is_first_round=False,
        )
        anno["second_samp"] = second_sample_dict
        print(second_answer_set)

        if len(second_answer_set) == 1:
            resolved = str(next(iter(second_answer_set)))
            if resolved == chr(ord("A") + anno["correct_choice"]):
                correct += 1
            print(f"second round, correct{correct}, all_num {all_num}, {correct / all_num}")
            anno["final_answer"] = resolved
            with open(config.result_path, "w") as f:
                json.dump(result_anno, f)
            continue

        print(second_answer_dict, chr(ord("A") + anno["correct_choice"]))
        rationale_dict = _collect_action_rationales(agent_set, anno, second_answer_dict, config)
        cross_review_scores = _run_cross_review(agent_set, anno, second_answer_dict, rationale_dict)
        surviving_agent_names, pruned_agent_name, aggregate_scores = _prune_agent_pool(config.priority_order, cross_review_scores)
        perception_clue_updates = _refine_perception_clues(
            agent_set,
            anno,
            surviving_agent_names,
            pruned_agent_name,
            aggregate_scores,
            rationale_dict,
            second_answer_dict,
            config,
        )
        anno["second_round"] = {
            "answer_dict": second_answer_dict,
            "scores": aggregate_scores,
            "aggregate_scores": aggregate_scores,
            "reason_dict": rationale_dict,
            "rationale_dict": rationale_dict,
            "discuss_dict": cross_review_scores,
            "cross_review_scores": cross_review_scores,
            "history_info": perception_clue_updates,
            "perception_clue_updates": perception_clue_updates,
        }

        consensus = _majority_if_any(second_answer_set, second_answer_dict, expected_count=2)
        if isinstance(consensus, str):
            anno["final_answer"] = str(consensus)
            if anno["final_answer"] == chr(ord("A") + anno["correct_choice"]):
                correct += 1
            print(f"second round, correct{correct}, all_num {all_num}, {correct / all_num}")
            with open(config.result_path, "w") as f:
                json.dump(result_anno, f)
            continue

        agent_set = [agent for agent in agent_set if agent.get_model_name() in surviving_agent_names]
        third_answer_set, third_answer_dict, third_sample_dict = _run_perception_action_round(
            agent_set=agent_set,
            anno=anno,
            config=config,
            clip_selector=clip_selector,
            perception_clue_updates=perception_clue_updates,
            is_first_round=False,
        )
        anno["third_samp"] = third_sample_dict
        anno["last_round"] = {"answer_dict": third_answer_dict}

        print(third_answer_dict)
        final_answer = third_answer_dict[agent_set[0].get_model_name()]
        print(final_answer)
        anno["final_answer"] = str(final_answer)
        if final_answer == chr(ord("A") + anno["correct_choice"]):
            correct += 1

        print(third_answer_dict, chr(ord("A") + anno["correct_choice"]))
        print(f"third round, correct{correct}, all_num {all_num}, {correct / all_num}")
        with open(config.result_path, "w") as f:
            json.dump(result_anno, f)

    with open(config.result_path, "w") as f:
        json.dump(result_anno, f)


def _run_perception_action_round(
    agent_set,
    anno,
    config: PipelineConfig,
    clip_selector: AspClipSelector,
    perception_clue_updates,
    is_first_round,
):
    answer_dict = {}
    sample_dict = {}
    global_context_gate_prompt, perception_clue_prompt, action_answer_prompt = config.build_prompts(anno)
    video_path = os.path.join(config.video_root, anno["video_path"])

    def process_agent(agent):
        agent_name = agent.get_model_name()
        cutting_points = anno.get("cutting_points", [])
        preview_frames = anno[agent_name].get("preview_frames", anno[agent_name].get("watch_samp", []))

        if is_first_round:
            requires_global_context = agent.get_answer(video_path, global_context_gate_prompt, preview_frames)
            requires_global_context = (
                requires_global_context[0] if isinstance(requires_global_context, list) else requires_global_context
            )
            if any(token in anno["question"].lower() for token in config.force_global_context_keywords):
                requires_global_context = "Yes"
            anno[agent_name]["watch"] = requires_global_context
            anno[agent_name]["needs_global_context"] = requires_global_context
        else:
            persisted_gate = anno[agent_name].get("needs_global_context", anno[agent_name].get("watch", "No"))
            requires_global_context = persisted_gate[0] if isinstance(persisted_gate, list) else persisted_gate

        if "Yes" in requires_global_context:
            if is_first_round and "sample_idx" in anno[agent_name]:
                sample_idx = anno[agent_name]["sample_idx"]
            else:
                sample_idx = _sample_frame_indices(
                    video_path=video_path,
                    sample_frame=config.action_frame_count,
                    cutting_points=cutting_points,
                    use_cutpoints_for_whole_sampling=config.use_cutpoints_for_whole_sampling,
                    allow_segment_sampling=config.allow_segment_sampling,
                    round_id=0,
                    judge_whole=True,
                )
            result = agent.get_answer(video_path, action_answer_prompt, sample_idx)
            if is_first_round:
                perception_clue = agent.get_answer(video_path, perception_clue_prompt, preview_frames)
                anno[agent_name]["info"] = perception_clue
                anno[agent_name]["perception_clue"] = perception_clue
            sample_dict[agent_name] = {"all_samp": [sample_idx]}
        else:
            if is_first_round:
                perception_clue = agent.get_answer(video_path, perception_clue_prompt, preview_frames)
                anno[agent_name]["info"] = perception_clue
                anno[agent_name]["perception_clue"] = perception_clue
            else:
                perception_clue = perception_clue_updates[agent_name]

            if isinstance(perception_clue, list):
                perception_clue = perception_clue[0]

            sample_dict_in = None
            if "sample_dict" in anno[agent_name]:
                if is_first_round:
                    sample_dict_in = anno[agent_name]["sample_dict"]
                elif not config.reuse_block_sampling_for_intern_78b_only or agent_name == "intern_78b":
                    sample_dict_in = anno[agent_name]["sample_dict"]

            best_idx, select_block, sample_all_frames = _select_clip_frames(
                clip_selector=clip_selector,
                video_path=video_path,
                perception_clue=perception_clue,
                question=anno["question"],
                cutting_points=cutting_points,
                action_frame_count=config.action_frame_count,
                prior_block_samples=sample_dict_in,
                clip_score_subsample=config.clip_score_subsample,
            )
            result = agent.get_answer(video_path, action_answer_prompt, best_idx)
            if is_first_round:
                sample_dict[agent_name] = {"all_samp": sample_all_frames, "block": select_block, "score_idx": best_idx}
            else:
                sample_dict[agent_name] = {"all_samp": sample_all_frames, "block": select_block, "sample_idx": best_idx}

        if isinstance(result, list):
            result = result[0]
        answer_dict[agent_name] = result.split("Answer: ")[-1][0]

    threads = [threading.Thread(target=process_agent, args=(agent,)) for agent in agent_set]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return set(answer_dict.values()), answer_dict, sample_dict


def _collect_action_rationales(agent_set, anno, answer_dict, config: PipelineConfig):
    video_path = os.path.join(config.video_root, anno["video_path"])
    rationale_dict = {}

    def process_agent(agent):
        agent_name = agent.get_model_name()
        answer_char = answer_dict[agent_name]
        if config.strict_choice_check and answer_char not in config.valid_choices:
            rationale_dict[agent_name] = "I am wrong."
            return

        candidate_idx = ord(answer_char) - ord("A")
        predicted_answer = anno["candidates"][candidate_idx] if 0 <= candidate_idx < len(anno["candidates"]) else "Invalid Answer"
        reason_prompt = (
            "Given the video frames you've seen, and the question along with your answer, deeply analyze the logical "
            "steps and evidence from the frames that led you to provide this particular answer. "
            f"The Question is: {anno['question']}\n, The predict answer is {predicted_answer}\n."
        )
        if "sample_idx" not in anno[agent_name]:
            rand_block = random.randint(1, N_CHUNKS)
            local_sample_idx = _sample_frame_indices(
                video_path=video_path,
                sample_frame=config.rationale_frame_count,
                cutting_points=anno.get("cutting_points", []),
                use_cutpoints_for_whole_sampling=config.use_cutpoints_for_whole_sampling,
                allow_segment_sampling=config.allow_segment_sampling,
                round_id=rand_block,
                judge_whole=True,
            )
        else:
            local_sample_idx = anno[agent_name]["sample_idx"]

        try:
            result = agent.get_answer(video_path, reason_prompt, local_sample_idx)
        except Exception:
            rand_block = random.randint(1, N_CHUNKS)
            local_sample_idx = _sample_frame_indices(
                video_path=video_path,
                sample_frame=config.rationale_frame_count,
                cutting_points=anno.get("cutting_points", []),
                use_cutpoints_for_whole_sampling=config.use_cutpoints_for_whole_sampling,
                allow_segment_sampling=config.allow_segment_sampling,
                round_id=rand_block,
                judge_whole=True,
            )
            result = agent.get_answer(video_path, reason_prompt, local_sample_idx)
        if isinstance(result, list):
            result = result[0]
        rationale_dict[agent_name] = result

    threads = [threading.Thread(target=process_agent, args=(agent,)) for agent in agent_set]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return rationale_dict


def _refine_perception_clues(agent_set, anno, surviving_agent_names, pruned_agent_name, aggregate_scores, rationale_dict, answer_dict, config: PipelineConfig):
    discussion_summary_prompt = " Discussion History Summary:\n"
    for agent_name in surviving_agent_names:
        candidate = _candidate_text(anno, answer_dict[agent_name])
        discussion_summary_prompt += (
            f"{agent_name}'s answer: {candidate}\n Reason: {rationale_dict[agent_name]}\n "
            f"The final score is {aggregate_scores[agent_name]}.\n"
        )

    removed_answer = _candidate_text(anno, answer_dict[pruned_agent_name])
    discussion_summary_prompt += (
        f"Removed Answer ({pruned_agent_name})\n Answer: {removed_answer}\n Reason {rationale_dict[pruned_agent_name]}\n "
        "However, this reason was deemed unconvincing, so this answer was removed from the discussion."
    )

    perception_clue_updates = {}
    for agent in agent_set:
        agent_name = agent.get_model_name()
        previous_clue = anno[agent_name].get("perception_clue", anno[agent_name].get("info", ""))
        clue_update_prompt = config.build_history_prompt(anno, discussion_summary_prompt, previous_clue)
        perception_clue_updates[agent_name] = agent.get_text_answer(clue_update_prompt)
        if isinstance(perception_clue_updates[agent_name], list):
            perception_clue_updates[agent_name] = perception_clue_updates[agent_name][0]
    return perception_clue_updates
