MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
TOKENIZER_REVISION = MODEL_REVISION
DATASET_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
SETTINGS = dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=0,
                repetition_penalty=1.0, max_new_tokens=1024,
                num_return_sequences=1, num_beams=1)
CONTEXT_LIMIT = 32768
CONFIGURATIONS = {"single": (1, 0), "debate3": (3, 2), "debate5": (5, 1)}
RUN_SEEDS = [101, 202, 303]
DEV_SEED = 404
