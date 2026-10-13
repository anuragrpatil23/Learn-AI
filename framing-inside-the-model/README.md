# Framing, from inside the model

Work towards explaining, inside the model, the finding of Yun et al. 2026 ("This Treatment Works, Right?", arXiv 2604.05051): a medical question framed positively or negatively, with the same evidence supplied, gets a different answer more often than the same question asked twice. Their data and judgments: github.com/hyesunyun/LLMHealthFramingEffect.

The model read is Qwen3-30B-A3B-Instruct-2507, one of the paper's eight, with the sparse autoencoders the Qwen team published for every layer (Qwen-Scope, `Qwen/SAE-Res-Qwen3-30B-A3B-Base-W32K-L0_50`).

So far:

- `rebuild_check.py` asks whether those sparse autoencoders, trained on the base model, fit the instruct model. It runs the lab's own prompts through a model and measures, at each layer, how much of the stream the sparse autoencoder fails to rebuild.
- `rebuild_instruct.json` and `rebuild_base.json` are the results on the same 24 prompts. They agree to within a point or two on layers 0 to 46; the last layer does not carry over.

Nothing about the framing effect itself has been measured here yet.
