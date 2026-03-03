#!/usr/bin/env python
import os
import json
import argparse
import multiprocessing as mp
from collections import defaultdict
from functools import partial  # kept in case you want to extend hooks later

import torch
from torch import Tensor
from tqdm import tqdm

# from sae_lens import SAE, HookedSAETransformer
# # from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory


# from nnterp import NNInterpModel
from nnsight import LanguageModel




def load_json_dim_dataset(path: str, dim: str | None = None, n: int | None = None) -> list[str]:
    """
    Very simple loader matching your JSON format:

    {
        "good": ["sentence 1", "sentence 2", ...],
        "bad": [...],
        ...
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if dim is None:
        raise ValueError("dim must be provided to select a key from the JSON dataset.")

    texts = data[dim]

    if n is not None:
        texts = texts[-n:]
    return texts



def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Collect Gemma-3 SAE and resid_post activations with sae_lens"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-4b-it",
        help="HF / TransformerLens model name compatible with HookedSAETransformer",
    )
    # parser.add_argument(
    #     "--sae_release",
    #     type=str,
    #     required=True,
    #     help="SAE Lens release name for this model (e.g. 'gemma-3-4b-res-myrelease')",
    # )
    # parser.add_argument(
    #     "--sae_width",
    #     type=str,
    #     required=True,
    #     help="SAE Lens release width for this model (e.g. '16k')",
    # )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to JSON dataset file",
    )
    parser.add_argument(
        "--dim",
        type=str,
        default=None,
        help="Dimension key in JSON (if dataset is a dict of dim-keys)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save activation stats (will be created if missing)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of examples to process",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size in number of prompts",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    texts = load_json_dim_dataset(args.dataset_path, dim=args.dim, n=args.max_samples)
    texts = [t["prompt"] for t in texts]
    print(f"Loaded {len(texts)} samples from {args.dataset_path}")

    # Pick device + dtype
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    torch.set_grad_enabled(False)

    # Load model
    print(f"Loading model {args.model_name} on {device}...")

    # model: HookedSAETransformer = HookedSAETransformer.from_pretrained(
    #     args.model_name,
    #     device=device,
    #     torch_dtype=dtype,
    # )
    # d_model = model.cfg.d_model
    # n_layers = model.cfg.n_layers

    # model = StandardizedTransformer(
    #     args.model_name,
    #     device=device,
    #     torch_dtype=dtype,
    # )
    model = LanguageModel(
        args.model_name,
        device_map="auto",
        dispatch=True
    )

    try:
        d_model = model.config.hidden_size
        n_layers = model.config.num_hidden_layers
    except:
        d_model = model.config.text_config.hidden_size
        n_layers = model.config.text_config.num_hidden_layers

    print(f"Model has {n_layers} layers, d_model={d_model}")

   


    ##################################################################
    # 2) Initialise accumulators
    ##################################################################
    model_resid_post_over_zero = torch.zeros(
        n_layers, d_model, dtype=torch.float32, device=device
    )
    model_resid_post_over_zero_binary = torch.zeros(
        n_layers, d_model, dtype=torch.int64, device=device
    )


    total_tokens = 0  # number of (batch, seq) positions processed

    ##################################################################
    # 3) Run the model once per batch, then feed resid_post to SAEs
    ##################################################################

    # The above comprehension is a bit awkward; a clearer one:
    # needed_hook_names = list({hook_name for (_, hook_name) in saes_by_layer_and_hook.keys()})

    # needed_hook_names = [f"blocks.{layer_idx}.hook_resid_post" for layer_idx in range(n_layers)]
    # needed_hook_names = [f"blocks.{layer_idx}.resid_post" for layer_idx in range(n_layers)]
    needed_hook_names = [f"model.transformer.h.{layer_idx}.output" for layer_idx in range(n_layers)]

    def batches(iterable, batch_size):
        for i in range(0, len(iterable), batch_size):
            yield iterable[i: i + batch_size]

    print("Collecting activations with cached resid_post...")
    for batch_texts in tqdm(
        list(batches(texts, args.batch_size)),
        total=(len(texts) + args.batch_size - 1) // args.batch_size,
    ):

        toks = model.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
            
        )

        input_ids = toks["input_ids"].to(device)
        attention_mask = toks["attention_mask"].to(device)

        
        pad_mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq, 1]

        # Run model once, cache only the hooks we care about
        with torch.no_grad():

            with model.trace(input_ids, 
            attention_mask=attention_mask, 
            retain=needed_hook_names
            ):
                for layer_idx in range(n_layers):
                    layers = (
                        model.model.layers
                        if hasattr(model, "model") and hasattr(model.model, "layers")
                        else model.language_model.layers
                    )
                    resid = layers[layer_idx].output[0].save()

                    # print(resid.shape)
                
                    tmp_resid = resid * pad_mask_expanded.to(resid.device)

                    # Update model resid stats for this layer
                    model_resid_post_over_zero[layer_idx] += tmp_resid.sum(dim=(0, 1)).to(model_resid_post_over_zero.device)
                    model_resid_post_over_zero_binary[layer_idx] += (tmp_resid > 0).sum(dim=(0, 1)).to(model_resid_post_over_zero_binary.device)


        # batch_size, seq_len = toks.shape
        #print(toks.shape)
        # num_positions = batch_size * seq_len
        total_tokens += int(pad_mask_expanded.sum())


        # Free cache for this batch
        # del cache
        torch.cuda.empty_cache()

    ##################################################################
    # 4) Save stats
    ##################################################################
    output_path = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving activation stats to {output_path}")

    model_resid_post_over_zero_output = dict(
        n=total_tokens,
        over_zero=model_resid_post_over_zero.to("cpu"),
    )
    model_resid_post_over_zero_binary_output = dict(
        n=total_tokens,
        over_zero=model_resid_post_over_zero_binary.to("cpu"),
    )



    torch.save(
        model_resid_post_over_zero_output,
        f"{output_path}/model_resid_post_activation.{args.dim}",
    )
    torch.save(
        model_resid_post_over_zero_binary_output,
        f"{output_path}/model_resid_post_activation_binary.{args.dim}",
    )

    print("Done.")


if __name__ == "__main__":
    main()
