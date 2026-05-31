# A4VL

Official implementation for our CVPR'26 paper A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning.

## Required environment (py39)
- Python `3.9.19`

Required modules tested in this repo:

```text
torch==2.4.0+cu121
torchvision==0.15.2+cu117
transformers==4.51.3
accelerate==1.7.0
safetensors==0.4.5
tokenizers==0.21.1
sentencepiece==0.1.99
decord==0.6.0
pandas==2.2.3
numpy==1.26.4
Pillow==9.5.0
tqdm==4.66.5
opencv-python==4.11.0
ruptures==1.1.10
requests==2.31.0
packaging==24.1
ftfy==6.3.1
regex==2.5.147
einops==0.6.1
boto3==1.42.35
botocore==1.42.35
```

## Things to modify on your machine
- `nextqa_pipeline.py`
  - Update dataset annotation path: `../vidagent/dataset/NExTVideo/test.csv`
  - Update video root: `video_root="../vidagent/dataset/NExTVideo/videos"`
  - Update output file if needed: `result_path="nextqa_results.json"`
- `egoschema_pipeline.py`
  - Update dataset annotation path: `../vidagent/dataset/egoschema/test.csv`
  - Update video root: `video_root="../vidagent/dataset/egoschema/videos"`
  - Update output file if needed: `result_path="egoschema_results.json"`
- `mlvu_pipeline.py`
  - Update dataset annotation path: `../vidagent/dataset/MLVU/test.csv`
  - Update video root: `video_root="../vidagent/dataset/MLVU/videos"`
  - Update output file if needed: `result_path="mlvu_results.json"`
- `cutting_points.py`
  - Replace absolute cache/env defaults to paths valid on your machine:
    - `TORCH_HOME`
    - `HUGGINGFACE_HUB_CACHE`
    - `KERAS_HOME`
- `pipeline_runtime.py`
  - Ensure ASP-CLIP checkpoint path is valid: `model_path="pytorch_model_0.0011.bin.25"` in `AspClipSelector`.
  - If the checkpoint is stored elsewhere, update that path.
- `run.sbatch` (if using SLURM)
  - Update cluster-specific fields: GPU type/count, partition/QOS, walltime, memory.
  - Update environment name: `conda activate py39`.
  - Update cache path export: `HF_HOME=...`.
- `eval.py`
  - Update input JSON path.
- Dataset files: usually save the questions in test.csv (except for LongVideoBench), and in videos folder save video files.

## Benchmark pipelines
- `nextqa_pipeline.py`
- `egoschema_pipeline.py`
- `mlvu_pipeline.py`

## Core modules
- `pipeline_runtime.py`: shared multi-round debate pipeline engine (all benchmark logic)
- `model_backends.py`: model client classes (InternVL and Qwen2.5-VL)
- `prompt_templates.py`: benchmark prompt templates and history prompts
- `vision_io.py`: image/video preprocessing and `process_vision_info`
- `cutting_points.py`: event-based video partition and output cutting points
- `modules/` and `pytorch_model_0.0011.bin.25`: ASP-CLIP code + weights
- `llava/`: local LLaVA implementation used by backends

## Results
Our runtime results are stored in results/. Due to a later optimization of the code, the output structure of the current code might be slightly different from the json files in results/ directory. However, the accuracy should match. Json files in results/ directory are exactly those analyzed in paper experiments. Now nextqa_results.json is the result from the current code. We are working on validating other benchmarks.

## Unified launcher
```bash
cd A4VL
python main.py --list
python main.py nextqa
python main.py ego
python main.py mlvu
python main.py all
python eval.py
```
