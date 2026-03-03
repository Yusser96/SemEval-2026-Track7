#!/bin/sh
export PYTHONPATH=./src:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0
#,1,2,3,4,5

cd SemEval2026

export HF_TOKEN=""
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"






python3 get_flores_data.py


bash pipeline_qwen_vectors.sh



bash pipeline_qwen_sem_eval.sh