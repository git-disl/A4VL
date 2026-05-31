import json
import os

import pandas as pd

from pipeline_runtime import PipelineConfig, run_pipeline
from prompt_templates import build_nextqa_history_prompt, build_nextqa_prompts


def load_nextqa_annotations(result_path: str):
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            return json.load(f)

    table = pd.read_csv("../vidagent/dataset/NExTVideo/test.csv")
    records = []
    for _, row in table.iterrows():
        records.append(
            {
                "video_path": f'{row["video"]}.mp4',
                "question": row["question"],
                "candidates": [row["a0"], row["a1"], row["a2"], row["a3"], row["a4"]],
                "correct_choice": row["answer"],
            }
        )
    return records


if __name__ == "__main__":
    run_pipeline(
        PipelineConfig(
            name="nextqa",
            video_root="../vidagent/dataset/NExTVideo/videos",
            result_path="nextqa_results.json",
            action_frame_count=16,
            rationale_frame_count=16,
            preview_bins=4,
            priority_order=["intern_3538b", "qwen2_5_72b", "intern_78b"],
            valid_choices=set("ABCDE"),
            build_prompts=build_nextqa_prompts,
            build_history_prompt=build_nextqa_history_prompt,
            load_annotations=load_nextqa_annotations,
            use_cutpoints_for_whole_sampling=True,
            allow_segment_sampling=True,
        )
    )
