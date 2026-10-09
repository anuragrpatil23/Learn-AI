"""Do the sparse autoencoders published for Qwen3 30B base fit Qwen3 30B Instruct?

The framing study reads Qwen3-30B-A3B-Instruct-2507, the model the lab ran. The sparse autoencoders the Qwen team
published (Qwen-Scope) were trained on the base model. This runs the lab's own prompts through a model and, for
each of the 48 layers, measures how much of the stream that layer's sparse autoencoder fails to rebuild. Run it
once for the instruct model and once for the base model, on the same prompts, and compare.

A sparse autoencoder here is: keep the 50 largest of (x @ W_enc.T + b_enc), zero the rest, then
x_hat = f @ W_dec.T + b_dec.

Two things are measured for every row (one token position at one layer):
  share lost   |x - x_hat|^2 / |x|^2, for that row alone. The median over rows is reported, so that a few
               unusual rows cannot decide the answer.
  outsized     a row whose length is more than 10 times the layer's median. These models put enormous values
               at a handful of positions; they are counted and reported apart.
The usual single number, squared error over all rows as a share of the rows' squared distance from their
average, is also given, with the outsized rows left out.

Usage: HF_HOME=<cache> python rebuild_check.py --lab <path to LLMHealthFramingEffect> --model <model> --name instruct
"""
import argparse, glob, json, os, time
import torch, jinja2
from transformers import AutoTokenizer, AutoModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--lab", required=True)
p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
p.add_argument("--chat-from", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="whose chat template wraps the prompt; the same for both models so they read the same tokens")
p.add_argument("--sae", default="Qwen/SAE-Res-Qwen3-30B-A3B-Base-W32K-L0_50")
p.add_argument("--name", default="instruct", help="results go to rebuild_<name>.json, and the sampled rows to rows_<name>.pt")
p.add_argument("--reviews", type=int, default=12)
p.add_argument("--per-prompt", type=int, default=200, help="token positions kept from each prompt, chosen at random, plus the last")
p.add_argument("--max-tokens", type=int, default=4000, help="skip prompts longer than this, to keep the check quick")
args = p.parse_args()
torch.set_grad_enabled(False)
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
ROWS = "rows_%s.pt" % args.name

if not os.path.exists(ROWS):
    # ---------- the lab's prompts ----------
    data = json.load(open(os.path.join(args.lab, "code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json")))
    template = jinja2.Template(open(os.path.join(args.lab, "code/prompts/question_answering.jinja2")).read())
    tok = AutoTokenizer.from_pretrained(args.chat_from)

    def prompt(review, question):
        abstracts = [{"title": r["Title"], "abstract": r["Abstract"]} for r in review["Inputs"]]
        text = template.render(question=question, abstracts=abstracts)
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)

    prompts = []
    for review in data:
        q = review["Questions"]["effectiveness"]
        pair = [prompt(review, q["positive_question"]), prompt(review, q["negative_question"])]
        if max(len(tok(t).input_ids) for t in pair) <= args.max_tokens:
            prompts += pair
        if len(prompts) >= 2 * args.reviews:
            break
    print(len(prompts), "prompts from", len(prompts) // 2, "reviews", flush=True)

    # ---------- the model ----------
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to(dev).eval()
    print("%s loaded in %.0f s on %s, %s" % (args.model, time.time() - t0, dev, next(model.parameters()).dtype), flush=True)
    g = torch.Generator().manual_seed(0)
    kept, where, lengths, seconds = [], [], [], []           # kept[prompt] is (layers, positions kept, width)
    for n, text in enumerate(prompts):
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        t1 = time.time()
        hidden = model(ids, output_hidden_states=True).hidden_states      # [0] is the embedding; [i + 1] is the stream after layer i
        seconds.append(time.time() - t1); T = ids.shape[1]; lengths.append(T)
        pos = torch.cat([torch.randperm(T - 1, generator=g)[:args.per_prompt].sort().values, torch.tensor([T - 1])])
        kept.append(torch.stack([x[0, pos.to(dev)].float().cpu() for x in hidden[1:]]))
        where.append(torch.stack([torch.full_like(pos, n), pos, (pos == T - 1).long()], 1))
    print("tokens per prompt: %d to %d; seconds per prompt: %.1f on average" % (min(lengths), max(lengths), sum(seconds) / len(seconds)), flush=True)
    torch.save({"rows": torch.cat(kept, 1), "where": torch.cat(where), "lengths": lengths, "seconds": seconds, "model": args.model}, ROWS)
    del model

saved = torch.load(ROWS)
rows, where = saved["rows"], saved["where"]                  # rows: (layers, all kept positions, width)
is_last, is_first = where[:, 2] == 1, where[:, 1] == 0

# ---------- each layer's sparse autoencoder ----------
folder = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub",
                                "models--" + args.sae.replace("/", "--"), "snapshots", "*"))[0]
K = json.load(open(os.path.join(folder, "config.json")))["k"]
results = []
print("layer  outsized rows  median share lost  at answer position  usual number (outsized left out)")
for layer in range(rows.shape[0]):
    sae = {k: v.float() for k, v in torch.load(os.path.join(folder, "layer%d.sae.pt" % layer), map_location="cpu").items()}
    x = rows[layer]
    pre = x @ sae["W_enc"].T + sae["b_enc"]
    top = pre.topk(K, dim=-1)
    x_hat = torch.zeros_like(pre).scatter_(-1, top.indices, top.values) @ sae["W_dec"].T + sae["b_dec"]
    err, size = ((x - x_hat) ** 2).sum(-1), (x ** 2).sum(-1)
    length = size.sqrt(); big = length > 10 * length.median()
    ok = ~big
    lost = err / size
    usual = float(err[ok].sum() / ((x[ok] - x[ok].mean(0)) ** 2).sum())
    r = {"layer": layer, "outsized": int(big.sum()), "outsized_at_first_position": int((big & is_first).sum()),
         "median_share_lost": round(float(lost[ok].median()), 4), "median_share_lost_at_answer": round(float(lost[is_last & ok].median()), 4),
         "usual_not_rebuilt": round(usual, 4), "median_length": round(float(length.median()), 2)}
    results.append(r)
    print("%5d  %13d  %17.3f  %18.3f  %10.3f" % (layer, r["outsized"], r["median_share_lost"], r["median_share_lost_at_answer"], usual), flush=True)
json.dump({"model": saved["model"], "sae": args.sae, "rows": int(rows.shape[1]), "tokens": saved["lengths"],
           "seconds_per_prompt": saved["seconds"], "layers": results}, open("rebuild_%s.json" % args.name, "w"), indent=1)
