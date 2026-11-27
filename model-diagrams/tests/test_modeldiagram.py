"""Tests for modeldiagram. Uses the configs and tokenizers in the Hugging Face cache; no weights are
downloaded (the values test builds a small GPT-2 with random weights). The Qwen test builds the 9B model on
the meta device, which takes under a minute; skip it with -m "not slow"."""
import copy
import json

import pytest
import torch
import yaml

from modeldiagram import build, check, kinds, trace, values


def gpt2_spec(batch=3, tokens=11):
    cfg, kind, model = trace.load("gpt2", meta=True)
    ids = torch.zeros(batch, tokens, dtype=torch.long, device="meta")
    b = build.build_llm(trace.run_llm(model, ids), "gpt2", ids)
    return b, b.spec(dict(model="gpt2", revision=None, generator_commit="test", date="2026-11-26"))


@pytest.fixture(scope="module")
def gpt2():
    return gpt2_spec()


def test_unknown_class_stops_by_name():
    with pytest.raises(kinds.UnknownModuleClass, match="MysteryBlock"):
        kinds.role_of("MysteryBlock", "layers.0")


def test_names_decide_the_two_ambiguous_classes():
    assert kinds.role_of("Embedding", "wte") == "embed"
    assert kinds.role_of("Embedding", "wpe") == "position"
    assert kinds.role_of("Linear", "lm_head", top_level=True) == "unembed"
    assert kinds.role_of("Linear", "lm_head", top_level=False) == "linear"


def test_an_unknown_class_in_a_model_stops_the_tracer():
    class Odd(torch.nn.Module):
        def forward(self, x):
            return x
    m = torch.nn.Sequential(Odd())
    with pytest.raises(kinds.UnknownModuleClass, match="Sequential|Odd"):
        trace.Tracer(m)


def test_gpt2_spec_connects_and_has_the_reference_steps(gpt2):
    b, spec = gpt2
    assert check.check_spec(spec) == []
    assert [o["kind"] for o in spec["ops"]] == ["input", "embed", "position", "norm", "attention", "add", "norm", "mlp", "add",
                                                "norm", "unembed", "output"]
    assert spec["axes"] == dict(batch=3, tokens=11, vocab=50257, width=768, positions=1024, heads=12, head_width=64, mlp_width=3072)
    assert spec["blocks"] == [dict(id="block", name="Block", ops=["norm1", "attn", "add1", "norm2", "mlp", "add2"], repeat=12)]
    assert spec["stream"] == ["position.out", "add1.out", "add2.out", "after_blocks"]
    attn = next(o for o in spec["ops"] if o["kind"] == "attention")
    assert attn["mask"] == "causal" and attn["heads"] == 12 and "kv_heads" not in attn
    unembed = next(o for o in spec["ops"] if o["kind"] == "unembed")
    assert unembed["weights"] == ["embed.weight"]      # GPT-2 reads its scores off the token table
    # the parts the layout reads by name
    ids = {t["id"] for t in spec["tensors"]}
    for t in ["attn.q", "attn.k", "attn.v", "attn.scores", "attn.weights", "attn.mixed", "mlp.pre", "mlp.act",
              "attn.q.weight", "attn.out.weight", "mlp.up.weight", "mlp.down.weight", "norm1.weight", "position.rows"]:
        assert t in ids, t
    T = {t["id"]: t for t in spec["tensors"]}
    assert T["attn.scores"]["axes"] == ["batch", "heads", "tokens", "tokens"]
    assert T["attn.q.weight"]["axes"] == ["width", "width"]
    assert T["mlp.up.weight"]["axes"] == ["mlp_width", "width"]     # stored (out, in)


def test_every_step_says_something_with_the_models_numbers(gpt2):
    _, spec = gpt2
    text = {s["id"]: s["text"] for s in spec["steps"]}
    assert "50,257" in text["embed"]
    assert "12 heads" in text["attn"] and "64 numbers" in text["attn"]
    assert "3,072" in text["mlp"] and "GELU" in text["mlp"]
    assert "12 times" in text["block"]


def test_the_written_yaml_reads_back_the_same(gpt2, tmp_path):
    from modeldiagram.cli import write_yaml
    _, spec = gpt2
    write_yaml(spec, tmp_path / "x.model.yaml", ["test"])
    back = yaml.safe_load((tmp_path / "x.model.yaml").read_text())
    assert back["tensors"] == spec["tensors"] and back["ops"] == spec["ops"] and back["steps"] == spec["steps"]


@pytest.mark.parametrize("break_it, says", [
    (lambda s: s["ops"][4]["in"].__setitem__(0, "nowhere"), "not in tensors"),
    (lambda s: s["steps"].append(dict(id="ghost", text="boo")), "neither an op nor a block"),
    (lambda s: s["ops"][4].__setitem__("kind", "teleport"), "not one of the known kinds"),
    (lambda s: next(t for t in s["tensors"] if t["id"] == "add1.out")["axes"].__setitem__(2, "mlp_width"), "keeps its shape"),
    (lambda s: s["ops"][4].__setitem__("kv_heads", 5), "heads 12 over kv_heads 5"),
    (lambda s: s["ops"].insert(3, s["ops"].pop(5)), "before any op makes it"),
])
def test_check_names_what_is_broken(gpt2, break_it, says):
    spec = copy.deepcopy(gpt2[1])
    break_it(spec)
    assert any(says in p for p in check.check_spec(spec)), check.check_spec(spec)


def test_values_from_a_small_gpt2_fit_its_spec(tmp_path):
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=64, n_head=4, n_positions=64)).eval()
    model.config._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok("Q: Why is the sky blue?\nA:", return_tensors="pt").input_ids
    full = torch.cat([ids, torch.tensor([[464, 6766, 318]])], 1)
    b = build.build_llm(trace.run_llm(model, full), "tiny", full)
    spec = b.spec(dict(model="tiny"))
    assert check.check_spec(spec) == []
    # the tensors the pass made by plain arithmetic hold the right numbers
    assert torch.allclose(b.data["add1.out"], b.data["position.out"] + b.data["attn.out"], atol=1e-5)
    assert torch.allclose(b.data["add2.out"], b.data["add1.out"] + b.data["mlp.out"], atol=1e-5)
    assert torch.allclose(b.data["output.out"].sum(-1), torch.ones(1, full.shape[1]), atol=1e-5)
    assert torch.allclose(b.data["attn.weights"].sum(-1), torch.ones(1, 4, full.shape[1]), atol=1e-5)
    v = values.trim(b, tok, full, ids.shape[1], "test-run", "0" * 64)
    w = values.split_weights(v, b)
    assert check.check_values(spec, json.dumps(v), "test") == []
    assert check.check_values(spec, json.dumps(w), "weights", weights_file=True) == []
    assert "attn.q.weight" in w["slices"] and "attn.q.weight" not in v["slices"] and "unembed.top.weight" in v["slices"]
    assert v["prompt_tokens"] == 11 and len(v["top_by_pass"]) == 4
    assert v["slices"]["attn.scores"]["rows"][0][1] == -1e9     # masked: a later key
    assert len(v["attention"]["attn"]) == 4
    sc = v["attention_scores"]["attn"]                          # every head's scores, head 0 matching its slice
    assert len(sc) == 4 and sc["0"] == v["slices"]["attn.scores"]["rows"] and sc["3"] != sc["0"]
    assert len(v["slices"]["mlp.pre"]["rows"][0]) == values.drawn(256)      # only the columns the engine draws
    assert values.drawn(768) == 8 and values.drawn(3072) == 16 and values.drawn(10_000_000) == 32


@pytest.mark.slow
def test_qwen35_spec_connects():
    cfg, kind, model = trace.load("Qwen/Qwen3.5-9B", meta=True)
    assert kind == "vlm"
    roots, at = trace.run_vlm_meta(model, cfg, 825, [1, 60, 48], 61)
    b = build.build_vlm(roots, "Qwen/Qwen3.5-9B", torch.zeros(1, 825, dtype=torch.long, device="meta"), at)
    spec = b.spec(dict(model="Qwen/Qwen3.5-9B"))
    assert check.check_spec(spec) == []
    blocks = {x["id"]: x for x in spec["blocks"]}
    assert (blocks["vis"]["repeat"], blocks["mem"]["repeat"], blocks["att"]["repeat"]) == (27, 24, 8)
    assert "every fourth block" in blocks["att"]["pattern"]
    ops = {o["id"]: o for o in spec["ops"]}
    assert ops["vis_attn"]["mask"] == "none" and ops["att_attn"]["mask"] == "causal"
    assert (ops["att_attn"]["heads"], ops["att_attn"]["kv_heads"]) == (16, 4)
    assert ops["mem_memory"]["kind"] == "memory"
    assert spec["axes"]["patches"] == 2880 and spec["axes"]["v_merged"] == 720 and spec["axes"]["mem_heads"] == 32
    assert ops["vis_join"]["kind"] == "join" and ops["join"]["in"] == ["embed.out", "vis_linear2.out"]
    assert spec["streams"]["vision"][0] == "vis_position.out"


def small_qwen():
    """Qwen3.5 with its real layout (vision blocks, three memory blocks then one attention block, the
    hand-over) at a tiny size, random weights, on the CPU."""
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B")
    v, t = cfg.vision_config, cfg.text_config
    v.depth, v.hidden_size, v.num_heads, v.intermediate_size, v.out_hidden_size = 2, 64, 4, 96, 64
    v.num_position_embeddings = 64
    t.num_hidden_layers, t.layer_types = 8, ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]
    t.hidden_size, t.intermediate_size, t.num_attention_heads, t.num_key_value_heads, t.head_dim = 64, 96, 4, 2, 16
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 8, 8
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(cfg, attn_implementation="eager").float().eval()
    grid = [1, 8, 8]
    n_img = 64 // 4
    ids = torch.tensor([[11, 12, 13, 14, cfg.vision_start_token_id] + [cfg.image_token_id] * n_img + [cfg.vision_end_token_id, 15, 16, 17]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=torch.randn(64, 1536),
                  image_grid_thw=torch.tensor([grid]), mm_token_type_ids=(ids == cfg.image_token_id).long())
    return cfg, model, inputs


def test_vlm_job_traces_dumps_and_trims_a_small_qwen(tmp_path):
    from modeldiagram import vlm_job
    cfg, model, inputs = small_qwen()
    roots = trace.run_vlm(model, inputs)
    ids = inputs["input_ids"]
    b = build.build_vlm(roots, "small", ids, dict(image_at=5))
    spec = b.spec(dict(model="small"))
    assert check.check_spec(spec) == []
    blocks = {x["id"]: x["repeat"] for x in spec["blocks"]}
    assert blocks == {"vis": 2, "mem": 6, "att": 2}
    # only the drawn blocks kept their insides: block 2 (memory, not drawn) has no tensors below it
    lang = roots[2]
    layers = [k for k in lang.kids if k.cls == "Qwen3_5DecoderLayer"]
    assert layers[0].kids[0].inp is not None and layers[1].kids[0].inp is None and layers[1].out is not None
    # the join really holds the picture's rows where the placeholders were
    j = b.data["join.out"][0]
    assert torch.allclose(j[5:5 + 16], b.data["vis_linear2.out"], atol=1e-5)
    rows = vlm_job.chosen_rows(None, None, ["x"], cfg, "first", 184)["x"]
    rows["language"] = list(range(ids.shape[1]))
    rec = dict(tokens=ids[0].tolist(), words=[str(i) for i in ids[0].tolist()], reply=" A", image_at=5, grid=[1, 8, 8], rows=rows,
               vocab_words={}, tensors=vlm_job.dump_one(b, rows))
    torch.save(dict(rows_rule="first", inputs={"x-hint": rec}), tmp_path / "dump.pt")
    vlm_job.trim(tmp_path / "dump.pt", tmp_path, run_id="test")
    v = json.loads((tmp_path / "values" / "x-hint.json").read_text())
    assert check.check_values(spec, json.dumps(v), "x") == []
    assert check.check_values(spec, (tmp_path / "weights.json").read_text(), "w", weights_file=True) == []
    assert len(v["attention"]["att_attn"]) == 4 and len(v["attention"]["vis_attn"]) == 4      # every head
    assert "att_attn.scores" in v["slices"] and "vis_attn.k.weight" in json.loads((tmp_path / "weights.json").read_text())["slices"]
