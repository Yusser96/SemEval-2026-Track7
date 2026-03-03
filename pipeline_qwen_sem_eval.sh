#!/bin/bash
export PYTHONPATH=./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
#,1,2,3,4,5

cd SemEval2026

export HF_TOKEN=""
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=0





# MODEL_ID="Qwen/Qwen2.5-72B-Instruct" # L 0 58

MODEL_ID="Qwen/Qwen3-32B" # batch 16 # L 6 49

# MODEL_ID="Qwen/Qwen3-8B" # batch 64 # L 5 22



ALPHA="$1";
echo "alpha:" $ALPHA;


languages_saq=("am-ET" "en-ET" "ar-DZ" "en-DZ" "ar-EG" "en-EG" "ar-MA" "en-MA" "ar-SA" "en-SA" "as-AS" "en-AS" "az-AZ" "en-AZ" "eu-PV" "en-PV" "bg-BG" "en-BG" "zh-CN" "en-CN" "zh-SG" "en-AU" "en-GB" "en-US" "fr-FR" "en-FR" "el-GR" "en-GR" "ha-NG" "en-NG" "id-ID" "en-ID" "ga-IE" "en-IE" "ja-JP" "en-JP" "ko-KP" "en-KP" "ko-KR" "en-KR" "ms-SG" "zh-TW" "en-TW" "fa-IR" "en-IR" "es-EC" "en-EC" "es-MX" "en-MX" "es-ES" "en-ES" "su-JB" "en-JB" "sv-SE" "en-SE" "tl-PH" "en-PH" "ta-SG" "ta-LK" "en-LK" "en-SG")

languages_mcq=("ga-IE" "ar-MA" "tl-PH" "eu-PV" "ja-JP" "en-AU" "fr-FR" "ar-SA" "sv-SE" "zh-SG" "es-EC" "bg-BG" "ta-LK" "ar-EG" "en-GB" "en-US" "ko-KR" "ar-DZ" "zh-CN" "id-ID" "es-ES" "fa-IR" "es-MX" "as-AS" "el-GR" "am-ET" "ha-NG" "az-AZ" "ko-KP" "su-JB")

# --use_sys_prompt
# --enable_thinking

python3 vllm_sem_eval.py \
  --model_name $MODEL_ID \
  --dataset_path "vectors" \
  --output_dir "eval_results/sem_eval" \
  --dims "${languages_mcq[@]}" \
  --layers '-1' \
  --alpha $ALPHA \
  --batch_size 32 \
  --max_new_tokens 32 \
  --task_name "mcq"


python3 vllm_sem_eval.py \
  --model_name $MODEL_ID \
  --dataset_path "vectors" \
  --output_dir "eval_results/sem_eval" \
  --dims "${languages_saq[@]}" \
  --layers '-1' \
  --alpha $ALPHA \
  --batch_size 32 \
  --max_new_tokens 128 \
  --task_name "seq"





python3 vllm_sem_eval.py \
  --model_name $MODEL_ID \
  --dataset_path "vectors" \
  --output_dir "eval_results/sem_eval-sys_prompt" \
  --dims "${languages_mcq[@]}" \
  --layers '-1' \
  --alpha $ALPHA \
  --batch_size 32 \
  --max_new_tokens 32 \
  --task_name "mcq" \
  --use_sys_prompt




python3 vllm_sem_eval.py \
  --model_name $MODEL_ID \
  --dataset_path "vectors" \
  --output_dir "eval_results/sem_eval-sys_prompt" \
  --dims "${languages_saq[@]}" \
  --layers '-1' \
  --alpha $ALPHA \
  --batch_size 32 \
  --max_new_tokens 128 \
  --task_name "seq" \
  --use_sys_prompt





### thinking doesn't work



python3 vllm_sem_eval.py \
  --model_name $MODEL_ID \
  --dataset_path "vectors" \
  --output_dir "eval_results/sem_eval-thinking" \
  --dims "${languages_mcq[@]}" \
  --layers '-1' \
  --alpha $ALPHA \
  --batch_size 32 \
  --max_new_tokens 512 \
  --task_name "mcq" \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --min_p 0.0 \
  --enable_thinking



python3 vllm_sem_eval.py \
  --model_name $MODEL_ID \
  --dataset_path "vectors" \
  --output_dir "eval_results/sem_eval-thinking" \
  --dims "${languages_saq[@]}" \
  --layers '-1' \
  --alpha $ALPHA \
  --batch_size 32 \
  --max_new_tokens 512 \
  --task_name "seq" \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --min_p 0.0 \
  --enable_thinking




python3 vllm_sem_eval.py \
  --model_name $MODEL_ID \
  --dataset_path "vectors" \
  --output_dir "eval_results/sem_eval-sys_prompt-thinking" \
  --dims "${languages_mcq[@]}" \
  --layers '-1' \
  --alpha $ALPHA \
  --batch_size 32 \
  --max_new_tokens 512 \
  --task_name "mcq" \
  --use_sys_prompt \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --min_p 0.0 \
  --enable_thinking




python3 vllm_sem_eval.py \
  --model_name $MODEL_ID \
  --dataset_path "vectors" \
  --output_dir "eval_results/sem_eval-sys_prompt-thinking" \
  --dims "${languages_saq[@]}" \
  --layers '-1' \
  --alpha $ALPHA \
  --batch_size 32 \
  --max_new_tokens 512 \
  --task_name "seq" \
  --use_sys_prompt \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --min_p 0.0 \
  --enable_thinking


