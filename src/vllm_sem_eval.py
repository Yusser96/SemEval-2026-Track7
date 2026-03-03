#!/usr/bin/env python

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
# os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA_VLLM_V1"

import multiprocessing as mp

import argparse

import torch
from torch import Tensor

from sae_lens import SAE
from vllm import LLM, SamplingParams
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

import json
from vllm_hooks import apply_steervec_intervention, restore_original_forwards_saes


from tasks.sem_eval import (get_prompts, get_flores_language_mapping, clean_results)




def run_task(model, task_name, batch_size, source_lang, steer_lang,sampling_params, use_sys_prompt=False, enable_thinking=False):

    results = []

    tokenizer = model.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    prompts = get_prompts(source_lang,tokenizer=tokenizer,track=task_name,use_sys_prompt=use_sys_prompt,enable_thinking=enable_thinking)

    for i, (prompt,meta) in enumerate(prompts):
        outputs = model.generate(prompt, sampling_params)
        pred  = outputs[0].outputs[0].text.strip()

        pred_cleaned = clean_results(pred, task_name)

        results.append({
            "id": i,
            "prompt":prompt,
            "pred": pred,
            "pred_cleaned":pred_cleaned,
            "meta":meta
        })


    
    # for i, pred in enumerate(pred):
    #     pred_post = post_process_text(pred)

    #     # print(detected_lang)

    #     results.append({
    #         "id": i,
    #         "prompt":prompts[i][0],
    #         "model_translation": pred,
    #     })


    return results



def load_steer_vec(vectors_path, dim, layer, use_sae):

    file_name = "model_resid_post_activation"
    if use_sae:
        file_name="sae_activation" 
    
    all_svectors = torch.load(f'{vectors_path}/{file_name}_vectors_diffmean', weights_only=False)

    return torch.Tensor(all_svectors[layer][dim]).to(device)



def main():
    parser = argparse.ArgumentParser(
        description="Gemma-3 SAE steering using resid_post and SAE latent space"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-4b-it",
        help="HF / TransformerLens model name compatible with HookedSAETransformer",
    )
    # parser.add_argument(
    #     "--dim",
    #     type=str,
    #     default=None,
    #     help="Dimension key in JSON (if dataset is a dict of dim-keys)",
    # )
    parser.add_argument(
        "--dims",
        nargs='+',
        default=None,
        required=True,
        help="Dimensions keys",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to activation_stats.pt for target corpus", 
    )
    parser.add_argument(
        "--layers",
        nargs='+',
        type=int,
        default=14,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scaling for resid_post steering vector",
    )


    parser.add_argument(
        "--use_sys_prompt", 
        action='store_true', 
        help="if used it will use_sys_prompt"
    )

    parser.add_argument(
        "--enable_thinking", 
        action='store_true', 
        help="if used it will enable_thinking"
    )

    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="mcq, seq",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save activation stats (will be created if missing)",
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    torch.set_grad_enabled(False)

    

    # Load model
    print(f"Loading model {args.model_name}...")
    # model: HookedSAETransformer = HookedSAETransformer.from_pretrained(
    #     args.model_name,
    #     device=device,
    #     torch_dtype=dtype,
    # )
    print(torch.cuda.device_count())
    model = LLM(args.model_name, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True) # , dtype=dtype

    n_layers = model.llm_engine.model_config.hf_config.num_hidden_layers


    sampling_params = SamplingParams(
        temperature=0, #args.temperature,
        # top_p=args.top_p,
        # top_k=args.top_k,
        # min_p=args.min_p,
        max_tokens=args.max_new_tokens,
        # skip_special_tokens=True
    )


    if args.layers[0] == -1:
        args.layers = range(n_layers)


    

    # Load stats and build steering vectors
    vectors_path = os.path.join(args.dataset_path, args.model_name)

    locale2lang = get_flores_language_mapping()


    for source_lang in args.dims:
        print("source_lang:",source_lang)
        def bse_exp(source_lang):
            try:
                base_res = run_task(model, 
                                    args.task_name, 
                                    args.batch_size, 
                                    source_lang, 
                                    source_lang,
                                    sampling_params,
                                    use_sys_prompt=args.use_sys_prompt, 
                                    enable_thinking=args.enable_thinking)
            except Exception as e:
                raise Exception(f"*** {source_lang}: {e}")
                # continue
            return base_res

        # base_res = bse_exp(source_lang)
        base_res = None
            
        def steer_exp(steer_lang, base_res):
            for layer in args.layers:

                

                out_path = os.path.join(args.output_dir, str(args.alpha), args.model_name, str(layer))
                output_path = os.path.join(out_path,"eval",args.task_name)
                os.makedirs(output_path, exist_ok=True)

                output_path = os.path.join(output_path,f"{source_lang}-{steer_lang}.json")

                if os.path.exists(output_path):
                    print(f"file exists: {output_path}")
                    continue

                if base_res is None:
                    base_res = bse_exp(source_lang)

                
                
                steer_vec = load_steer_vec(vectors_path, locale2lang[steer_lang], layer, False)


                original_forwards = apply_steervec_intervention(args.model_name, model, layer, steer_vec, alpha=args.alpha, sae=None)
                
                

                res = run_task(model, 
                               args.task_name, 
                               args.batch_size, 
                               source_lang, 
                               steer_lang,
                               sampling_params,
                               use_sys_prompt=args.use_sys_prompt, 
                               enable_thinking=args.enable_thinking)
                

                # Clean up hooks / SAEs (optional in a one-off script but good practice)
                restore_original_forwards_saes(model, original_forwards)



                

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"mod":res, "base":base_res}, f, indent=4, ensure_ascii=False)

        
        steer_lang = source_lang
        steer_exp(steer_lang,base_res)

        # for steer_lang in args.dims:
        #     # if source_lang == steer_lang:
        #     #     continue
                
        #     steer_exp(steer_lang)

    


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
