#!/bin/bash
#SBATCH -J lvagent             # Job name
#SBATCH -N 1                            # Number of nodes
#SBATCH -N1 --gres=gpu:6 -C H200            # Request 1 GPU
#SBATCH --mem=200G                      # Increase memory
#SBATCH -t 240                          # Duration of the job
#SBATCH -o main_egoschema-%j.out                # Combined output and error messages file
#SBATCH -q coc-grade

cd $SLURM_SUBMIT_DIR                    # Correctly change to the submit directory

module load anaconda3
conda activate py311

export HF_HOME="/home/hice1/yxu846/scratch/models"
export HUGGINGFACE_HUB_CACHE="/home/hice1/yxu846/scratch/models/hub"
export TRANSFORMERS_CACHE="/home/hice1/yxu846/scratch/models/hub"
export HF_HUB_DISABLE_XET=1

srun python main.py egoschema
