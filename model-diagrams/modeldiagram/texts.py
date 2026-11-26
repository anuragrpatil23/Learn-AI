"""The walk-through: one plain sentence per step, said the way it would be said aloud, with the model's own
numbers in it. Written from the kind of each op and the sizes the pass found, so a new model gets its sentences
without anyone writing them; a sentence that reads badly for a model is fixed here, for every model."""


def n(x):
    return f"{int(x):,}"


ORD = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 8: "eighth"}


def pattern(where, total):
    """Which of the blocks share one layout, said plainly: 'every fourth block, from block 4 (4, 8, ..., 32)'."""
    if len(where) > 2:
        step = where[1] - where[0]
        if all(b - a == step for a, b in zip(where, where[1:])):
            return f"every {ORD.get(step, str(step) + 'th')} block, from block {where[0]} ({where[0]}, {where[1]}, ..., {where[-1]})"
    runs = []
    for w in where:
        if runs and w == runs[-1][1] + 1:
            runs[-1][1] = w
        else:
            runs.append([w, w])
    say = lambda r: str(r[0]) if r[0] == r[1] else f"{r[0]} to {r[1]}"
    shown = [say(r) for r in runs] if len(runs) <= 4 else [say(runs[0]), say(runs[1]), "and so on", say(runs[-1])]
    return "blocks " + ", ".join(shown)


def say(o, b):
    A, F, k = b.axes, b.facts.get(o["id"], {}), o["kind"]
    vis = o["id"].startswith("vis_")
    W = A["v_width"] if vis else A.get("width", 0)
    if k == "input":
        if "pixels" in o["out"]:
            w = b.tensors["vis_patchify.weight"]["axes"]
            return (f"The picture is cut into squares of {w[-1]} by {w[-2]} pixels, {w[-3]} frames deep, {n(A['patches'])} of them here, "
                    f"each one a row of {n(A['patch_pixels'])} numbers.")
        return f"The text is cut into tokens, which are words or pieces of words, and each token is a number from 0 to {n(A['vocab'] - 1)}."
    if k == "embed":
        return f"Each token picks its own row of {n(W)} numbers from a table with one row for each of the {n(A['vocab'])} tokens."
    if k == "position":
        if vis:
            return "Each patch gets a learnt row for where it sits in the picture, and it is added on."
        return (f"A second table has one row of {n(W)} for each place in the line, up to {n(A['positions'])}, "
                f"and each token's place row is added on, so the order of the words is known.")
    if k == "patchify":
        return f"Each patch's {n(A['patch_pixels'])} pixel numbers go through one layer and come out as {n(W)} numbers."
    if k == "norm":
        final = not o.get("block") and o["id"] in ("norm", "vis_norm")
        where = "Before the hand-over, each patch's row" if vis and final else "At the end, each token's row" if final else \
            ("Each patch's row" if vis else "Each token's row")
        if F.get("rms"):
            return f"{where} is divided by its root mean square, so its size is about one, then scaled by this norm's own {n(W)} weights."
        return f"{where} is moved to average zero and scaled to spread one, then given this norm's own scale and shift."
    if k == "attention":
        H, KV, hw = F["heads"], F["kv_heads"], F["head_width"]
        who = "patch" if vis else "token"
        if F["causal"]:
            s = (f"Each {who} makes a query, a key and a value; its query is compared with the keys of itself and every {who} before it, "
                 f"and it takes a weighted mix of their values.")
        else:
            s = (f"Each {who} makes a query, a key and a value; its query is compared with the keys of every {who} in the picture, "
                 f"before or after it, and it takes a weighted mix of their values.")
        s += f" {H} heads do this side by side, each {hw} numbers wide."
        if KV != H:
            s += f" The {H} heads share {KV} sets of keys and values."
        if "rotary" in o.get("note", ""):
            s += " Queries and keys are first turned by their place, so a comparison knows how far apart two are."
        if F.get("gated"):
            s += " A gate then decides how much of each head's result gets through."
        return s
    if k == "memory":
        return (f"Instead of looking back at every token, this block keeps a running memory: {F['heads']} squares of {F['key']} by {F['value']} numbers. "
                f"Each token lets the memory fade a little, writes into it, and reads from it, "
                f"so the work per token stays the same however long the text gets.")
    if k == "add":
        what = {"attn": "attention", "memory": "the memory", "mlp": "the MLP"}
        src = next((v for key, v in what.items() if any(i.startswith(o["id"].rsplit("add", 1)[0] + key) for i in o["in"])), "the step before")
        return f"What {src} found is added onto the stream, so the stream keeps what it had and gains this."
    if k == "mlp":
        M = A["v_mlp_width"] if vis else A["mlp_width"]
        who = "patch" if vis else "token"
        if F.get("gated"):
            return (f"Each {who}'s row, on its own, goes through two layers of {n(M)}; one is bent by {F['act']} and used as a gate on the other, "
                    f"and the result is brought back to {n(W)}.")
        return f"Each {who}'s row, on its own, is widened to {n(M)} numbers, bent by {F['act']}, and brought back to {n(W)}."
    if k == "join":
        if o["id"] == "join":
            return (f"The picture's {n(F['rows'])} rows take the places of its placeholder tokens, from token {F['at']} on, "
                    f"so words and picture go through the language model as one line of tokens.")
        return (f"Each group of {F['k']} neighbouring patches, 2 by 2, is put side by side as one row of {n(A['v_joined_width'])}, "
                f"so the picture's {n(A['patches'])} rows become {n(A['v_merged'])}.")
    if k == "linear":
        t = b.tensors[o["weights"][0]]["axes"]
        size = lambda a: A[a] if isinstance(a, str) else a
        return f"One layer takes each row from {n(size(t[1]))} numbers to {n(size(t[0]))}."
    if k == "act":
        return f"Every number is bent by {o.get('act', 'the bend')}."
    if k == "unembed":
        table = "every row of the token table" if F.get("tied") else f"each row of its own table of {n(A['vocab'])}"
        return f"Each token's final row is compared with {table}, giving a score for each of the {n(A['vocab'])} tokens."
    if k == "output":
        return "The scores are turned into chances that add up to one, and the most likely token is the next word."
    raise ValueError(f"No sentence for the kind {k}.")


def say_block(blk, b):
    F = b.facts[blk["id"]]
    if blk.get("pattern"):
        return (f"These steps are {F['count']} of the {F['total']} blocks, {blk['pattern']}; "
                f"each has its own weights.")
    return f"These steps run {F['count']} times, one block after another, each with its own weights."


def steps(b):
    out = []
    by_block = {blk["id"]: blk for blk in b.blocks}
    ops = b.ops
    for i, o in enumerate(ops):
        out.append(dict(id=o["id"], name=o["name"], text=say(o, b)))
        blk = o.get("block")
        nxt = ops[i + 1].get("block") if i + 1 < len(ops) else None
        if blk and nxt != blk:
            out.append(dict(id=blk, name=by_block[blk]["name"], text=say_block(by_block[blk], b)))
    return out


def name(o, b):
    """A short plain name for an op, for the step strip and the labels (after vlm-in-one-drawing's step names)."""
    k, vis = o["kind"], o["id"].startswith("vis_")
    F = b.facts.get(o["id"], {})
    final = k == "norm" and not o.get("block")
    norm = "RMS norm" if F.get("rms") else "Layer norm"
    if k == "input":
        return "The picture" if "pixels" in o["out"] else "The tokens"
    if k == "norm":
        if not final:
            return norm
        return f"{norm} before the hand-over" if vis else "Final " + ("RMS norm" if F.get("rms") else "layer norm")
    if k == "attention":
        return "Attention, no mask" if not F.get("causal") else "Gated attention" if F.get("gated") else "Attention"
    if k == "mlp":
        return f"Gated MLP + {F['act']}" if F.get("gated") else f"MLP + {F['act']}"
    if k == "linear":
        out = b.tensors[o["weights"][0]]["axes"][0]
        return "To the language model's width" if vis and out == "width" else "Hand-over layer 1" if vis else "Linear layer"
    if k == "act":
        return f"Bend ({o.get('act', '')})"
    if k == "join":
        return "Into the sequence" if o["id"] == "join" else "Join four patches"
    if k == "position":
        return "Add each patch's place" if vis else "Add each token's place"
    return {"embed": "Token to row", "patchify": "Patch to row", "memory": "Running memory", "add": "Add to the stream",
            "unembed": "Score per token", "output": "Chance of each token"}[k]


def block_name(blk, b):
    F = b.facts[blk["id"]]
    if blk["id"].startswith("vis"):
        return "Vision block"
    return {"memory": "Running-memory block", "attention": "Attention block"}.get(F.get("mixer"), "Block") if blk.get("pattern") else "Block"
