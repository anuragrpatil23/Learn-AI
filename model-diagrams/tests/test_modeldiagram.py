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
    assert check.check_values(spec, json.dumps(v), "test") == []
    assert v["prompt_tokens"] == 11 and len(v["top_by_pass"]) == 4
    assert v["slices"]["attn.scores"]["rows"][0][1] == -1e9     # masked: a later key
    assert len(v["attention"]["attn"]) == 4
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
