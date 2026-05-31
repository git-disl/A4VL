import json
import os

import pandas as pd

from pipeline_runtime import PipelineConfig, run_pipeline
from prompt_templates import build_egoschema_history_prompt, build_egoschema_prompts

def load_egoschema_annotations(result_path: str):
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            return json.load(f)

    table = pd.read_csv("../vidagent/dataset/egoschema/test.csv")
    records = []
    for _, row in table.iterrows():
        records.append(
            {
                "video_path": f'{row["q_uid"]}.mp4',
                "question": row["question"],
                "candidates": [row["option0"], row["option1"], row["option2"], row["option3"], row["option4"]],
                "A": row["option0"],
                "B": row["option1"],
                "C": row["option2"],
                "D": row["option3"],
                "E": row["option4"],
                "correct_choice": row["answer"],
            }
        )
    return records


if __name__ == "__main__":
    run_pipeline(
        PipelineConfig(
            name="egoschema",
            video_root="../vidagent/dataset/egoschema/videos",
            result_path="egoschema_results.json",
            action_frame_count=16,
            rationale_frame_count=16,
            preview_bins=4,
            priority_order=["intern_3538b", "qwen2_5_72b", "intern_78b"],
            valid_choices=set("ABCDE"),
            build_prompts=build_egoschema_prompts,
            build_history_prompt=build_egoschema_history_prompt,
            load_annotations=load_egoschema_annotations,
            use_cutpoints_for_whole_sampling=True,
            allow_segment_sampling=True,
        )
    )
