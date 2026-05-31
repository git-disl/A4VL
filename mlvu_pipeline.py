import json
import os

import pandas as pd

from pipeline_runtime import PipelineConfig, run_pipeline
from prompt_templates import build_mlvu_history_prompt, build_mlvu_prompts


def load_mlvu_annotations(result_path: str):
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            return json.load(f)

    table = pd.read_csv("../vidagent/dataset/MLVU/test.csv")
    records = []
    for _, row in table.iterrows():
        records.append(
            {
                "video_path": f'{row["video_id"]}',
                "question": row["question"],
                "candidates": [row["a0"], row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]],
                "A": row["a0"],
                "B": row["a1"],
                "C": row["a2"],
                "D": row["a3"],
                "E": row["a4"],
                "F": row["a5"],
                "correct_choice": row["answer"],
            }
        )
    return records


if __name__ == "__main__":
    run_pipeline(
        PipelineConfig(
            name="mlvu",
            video_root="../vidagent/dataset/MLVU/videos",
            result_path="mlvu_results.json",
            action_frame_count=32,
            rationale_frame_count=32,
            preview_bins=8,
            priority_order=["qwen2_5_72b", "intern_78b", "intern_3538b"],
            valid_choices=set("ABCDEF"),
            build_prompts=build_mlvu_prompts,
            build_history_prompt=build_mlvu_history_prompt,
            load_annotations=load_mlvu_annotations,
            use_cutpoints_for_whole_sampling=True,
            allow_segment_sampling=False,
            clip_score_subsample=16,
            force_global_context_keywords=("how many", "count"),
            strict_choice_check=True,
            agent_order=("intern_78b", "qwen2_5_72b", "intern_3538b"),
        )
    )
