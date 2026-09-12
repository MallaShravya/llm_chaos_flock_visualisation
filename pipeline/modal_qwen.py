import modal

app = modal.App("llm-chaos-export-embeddings")

hf_cache = modal.Volume.from_name(
    "llm-chaos-hf-cache",
    create_if_missing=True,
)

output_volume = modal.Volume.from_name(
    "llm-chaos-output",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "numpy",
    )
)

CACHE_PATH = "/cache/huggingface"
OUTPUT_PATH = "/output"


@app.function(
    image=image,
    gpu="T4",
    timeout=900,
    volumes={
        "/cache": hf_cache,
        "/output": output_volume,
    },
)
def export_runs():
    import os
    import json
    import torch

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )

    os.environ["HF_HOME"] = CACHE_PATH

    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    SEED = 42
    TEMPERATURE = 0.8
    TOP_P = 0.9
    MAX_NEW_TOKENS = 300

    prompt_a = (
        "Describe a bird flying over a city, including what it sees below "
        "and where it eventually goes."
    )

    # Exactly one extra trailing space
    prompt_b = (
        "Describe a bird flying over a city, including what it sees below "
        "and where it eventually goes. "
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        cache_dir=CACHE_PATH,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        cache_dir=CACHE_PATH,
        dtype=torch.float16,
    ).to("cuda")

    model.eval()

    # Static token embedding matrix
    embedding_matrix = (
        model
        .get_input_embeddings()
        .weight
        .detach()
        .cpu()
        .float()
    )

    print(
        "Embedding matrix shape:",
        tuple(embedding_matrix.shape),
    )

    embedding_dim = embedding_matrix.shape[1]

    def generate(prompt):
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
        ).to("cuda")

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

        raw_generated_ids = output[
            0,
            inputs.input_ids.shape[1]:
        ].tolist()

        stopped_naturally = (
            len(raw_generated_ids) < MAX_NEW_TOKENS
            or (
                raw_generated_ids
                and raw_generated_ids[-1]
                in tokenizer.all_special_ids
            )
        )

        generated_ids = [
            token_id
            for token_id in raw_generated_ids
            if token_id not in tokenizer.all_special_ids
        ]

        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return (
            generated_ids,
            text,
            stopped_naturally,
        )

    def build_records(ids):
        records = []

        for index, token_id in enumerate(ids):
            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

            embedding = (
                embedding_matrix[token_id]
                .tolist()
            )

            records.append(
                {
                    "index": index,
                    "token_id": token_id,
                    "token": token_text,
                    "embedding": embedding,
                }
            )

        return records

    def first_divergence(ids_a, ids_b):
        common_length = min(
            len(ids_a),
            len(ids_b),
        )

        for i in range(common_length):
            if ids_a[i] != ids_b[i]:
                return i

        if len(ids_a) != len(ids_b):
            return common_length

        return None

    print("\nGenerating run A...")
    ids_a, text_a, stopped_a = generate(
        prompt_a
    )

    print("Generating run B...")
    ids_b, text_b, stopped_b = generate(
        prompt_b
    )

    divergence = first_divergence(
        ids_a,
        ids_b,
    )

    print("\nBuilding embedding records...")

    records_a = build_records(ids_a)
    records_b = build_records(ids_b)

    run_a = {
        "run": "A",
        "model": MODEL,
        "prompt": prompt_a,
        "generated_text": text_a,
        "token_count": len(ids_a),
        "stopped_naturally": stopped_a,
        "embedding_dimension": embedding_dim,
        "generation_settings": {
            "seed": SEED,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "first_divergence": divergence,
        "tokens": records_a,
    }

    run_b = {
        "run": "B",
        "model": MODEL,
        "prompt": prompt_b,
        "generated_text": text_b,
        "token_count": len(ids_b),
        "stopped_naturally": stopped_b,
        "embedding_dimension": embedding_dim,
        "generation_settings": {
            "seed": SEED,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "first_divergence": divergence,
        "tokens": records_b,
    }

    metadata = {
        "model": MODEL,
        "embedding_dimension": embedding_dim,
        "run_a_token_count": len(ids_a),
        "run_b_token_count": len(ids_b),
        "first_divergence": divergence,
        "prompt_difference": (
            "Run B contains one extra trailing "
            "space after the final period."
        ),
    }

    os.makedirs(
        OUTPUT_PATH,
        exist_ok=True,
    )

    path_a = f"{OUTPUT_PATH}/run_a.json"
    path_b = f"{OUTPUT_PATH}/run_b.json"
    path_metadata = (
        f"{OUTPUT_PATH}/metadata.json"
    )

    with open(
        path_a,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_a,
            f,
            ensure_ascii=False,
        )

    with open(
        path_b,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_b,
            f,
            ensure_ascii=False,
        )

    with open(
        path_metadata,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    output_volume.commit()

    print("\nEXPORT COMPLETE")
    print("------------------------------")
    print("Run A tokens:", len(ids_a))
    print("Run B tokens:", len(ids_b))
    print("Embedding dimension:", embedding_dim)
    print("First divergence:", divergence)
    print("Run A stopped naturally:", stopped_a)
    print("Run B stopped naturally:", stopped_b)

    print("\nFiles written:")
    print(path_a)
    print(path_b)
    print(path_metadata)


@app.local_entrypoint()
def main():
    export_runs.remote()