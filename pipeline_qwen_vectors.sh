#!/bin/sh
export PYTHONPATH=./src:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0
#,1,2,3,4,5

cd SemEval2026

export HF_TOKEN=""
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=0




#python3 list_saes.py

MODEL_ID="Qwen/Qwen2.5-72B-Instruct"

# MODEL_ID="Qwen/Qwen3-32B" # batch 16

# MODEL_ID="Qwen/Qwen3-8B" # batch 64



#python3  -m pip install git+https://github.com/ericwtodd/nnterp.git


languages=('amh_Ethi' 'kab_Latn' 'arz_Arab' 'ary_Arab' 'ars_Arab' 'azj_Latn' 'bul_Cyrl' 'ell_Grek' 'azj_Latn' 'bul_Cyrl' 'zho_Hans' 'kab_Latn' 'arb_Latn' 'spa_Latn' 'amh_Ethi' 'fra_Latn' 'eng_Latn' 'ell_Grek' 'ind_Latn' 'gle_Latn' 'pes_Arab' 'jpn_Jpan' 'kor_Hang' 'sin_Sinh' 'arb_Latn' 'hau_Latn' 'tgl_Latn' 'ars_Arab' 'swe_Latn' 'zho_Hant' 'eng_Latn' 'spa_Latn' 'pes_Arab' 'fra_Latn' 'gle_Latn' 'hau_Latn' 'ind_Latn' 'jpn_Jpan' 'kor_Hang' 'swe_Latn' 'sin_Sinh' 'tgl_Latn' 'zho_Hans' 'zho_Hant' 'eus_Latn' 'asm_Beng' 'jav_Latn' 'zsm_Latn' 'tam_Taml')



echo "${languages[@]}"

for DIM in "${languages[@]}"
do
  FILE="SemEval2026/activations/$MODEL_ID/model_resid_post_activation_binary.$DIM"
  if [ -f $FILE ]; then
      echo "File $FILE exists."
  else
      echo "Running language: $DIM" # vllm2

      echo $DIM
      python3  collect_sae_activations.py \
      --model_name $MODEL_ID \
      --batch_size 32 \
      --dataset_path "data/flores200_dataset_low_res.json" \
      --dim $DIM \
      --output_dir activations \
      --max_samples 1000
  fi
done

languages=('amh_Ethi' 'kab_Latn' 'arz_Arab' 'ary_Arab' 'ars_Arab' 'azj_Latn' 'bul_Cyrl' 'ell_Grek' 'azj_Latn' 'bul_Cyrl' 'zho_Hans' 'kab_Latn' 'arb_Latn' 'spa_Latn' 'amh_Ethi' 'fra_Latn' 'eng_Latn' 'ell_Grek' 'ind_Latn' 'gle_Latn' 'pes_Arab' 'jpn_Jpan' 'kor_Hang' 'sin_Sinh' 'arb_Latn' 'hau_Latn' 'tgl_Latn' 'ars_Arab' 'swe_Latn' 'zho_Hant' 'eng_Latn' 'spa_Latn' 'pes_Arab' 'fra_Latn' 'gle_Latn' 'hau_Latn' 'ind_Latn' 'jpn_Jpan' 'kor_Hang' 'swe_Latn' 'sin_Sinh' 'tgl_Latn' 'zho_Hans' 'zho_Hant' 'eus_Latn' 'asm_Beng' 'jav_Latn' 'zsm_Latn' 'tam_Taml')


python3 create_steer_vector.py \
  --model_name $MODEL_ID \
  --dataset_path "activations" \
  --dims "${languages[@]}" \
  --output_dir vectors




