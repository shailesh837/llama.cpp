#!/usr/bin/env python3
"""Make a Qwen3.8 MTP drafter GGUF with a reduced output head (draft shortlist).

The drafter's output head has one row per token (248k rows, read in full for every
drafted token). This writes a copy of the file whose output.weight keeps only the rows
of the N most frequent tokens, plus a tensor d2t with the token id of each row. A
llama.cpp with d2t support in qwen4exp (branch local-speedup) scatters the short logits
back to full size, so a token outside the list can simply not be drafted.

Usage:
  mtp_shortlist.py IN.gguf COUNTS.json N [OUT.gguf] [--classes CLASSES.json NAMES]
                   [--eval URL FILE...]
    COUNTS.json  token frequencies from shortlist-count.py
    N            a number: that many rows, the most frequent tokens first;
                 "seen": every token the corpus contains;
                 "script": every token of the --classes NAMES and nothing else.
                 Rounded up to a multiple of 64. Control and user-defined tokens
                 and the tokens of --classes NAMES are always kept.
    --classes    CLASSES.json from shortlist-scripts.py and a comma-separated list
                 of its class names (e.g. latin,symbol,fragment,greek)
    OUT.gguf     written if given (refuses to overwrite); without it: dry run
    --eval       tokenize FILEs with the llama-server at URL and print how many of
                 their tokens the list covers
Only the header is parsed; tensor data is copied byte for byte. Python stdlib only."""
import json
import os
import struct
import sys
import urllib.request

args = sys.argv[1:]
ev = []
if "--eval" in args:
    i = args.index("--eval")
    ev, args = args[i + 1:], args[:i]
cls_ids = []
if "--classes" in args:
    i = args.index("--classes")
    cj_cls = json.load(open(args[i + 1]))
    cls_ids = [t for name in args[i + 2].split(",") for t in cj_cls[name]]
    args = args[:i] + args[i + 3:]
IN, COUNTS, N = args[0], args[1], args[2]
OUT = args[3] if len(args) > 3 else None

f = open(IN, "rb")
SCAL = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def rd(fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def rstr():
    return f.read(rd("Q")).decode("utf-8", "replace")


def rval(t):
    if t in SCAL:
        return rd(SCAL[t])
    if t == 8:
        return rstr()
    et, n = rd("I"), rd("Q")
    if et == 8:
        return [rstr() for _ in range(n)]
    fmt = "<%d%s" % (n, SCAL[et])
    return list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))


def pstr(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


assert f.read(4) == b"GGUF", "not a GGUF file"
version, n_tensors, n_kv = rd("I"), rd("Q"), rd("Q")
kv_start = f.tell()
meta = {}
for _ in range(n_kv):
    k, t = rstr(), rd("I")
    meta[k] = rval(t)
kv_end = f.tell()
tensors = []
for _ in range(n_tensors):
    name = rstr()
    dims = [rd("Q") for _ in range(rd("I"))]
    tensors.append(dict(name=name, dims=dims, type=rd("I"), off=rd("Q")))
align = meta.get("general.alignment", 32)
data_start = -(-f.tell() // align) * align
file_size = os.path.getsize(IN)
by_off = sorted(tensors, key=lambda t: t["off"])
for a, b in zip(by_off, by_off[1:] + [None]):
    a["size"] = (b["off"] if b else file_size - data_start) - a["off"]

tokens = meta["tokenizer.ggml.tokens"]
ttype = meta["tokenizer.ggml.token_type"]
n_vocab = len(tokens)
out_t = next(t for t in tensors if t["name"] == "output.weight")
assert not any(t["name"] == "d2t" for t in tensors), "this file already has a shortlist"
assert out_t["type"] == 2 and out_t["dims"][1] == n_vocab and out_t["dims"][0] % 32 == 0, \
    f"output.weight must be Q4_0 [n_embd, n_vocab], got type {out_t['type']} {out_t['dims']}"
row_bytes = out_t["dims"][0] // 32 * 18
assert out_t["size"] >= n_vocab * row_bytes

# ---- the shortlist: control / user-defined tokens, then the most frequent ones
cj = json.load(open(COUNTS))
counts = {int(k): v for k, v in cj["counts"].items()}
always = sorted(set(i for i, t in enumerate(ttype) if t in (3, 4)) | set(cls_ids))
ranked = sorted(counts, key=lambda i: (-counts[i], i))
keep = set(always)
if N == "seen":
    keep |= set(counts)
elif N != "script":
    for i in ranked:
        if len(keep) >= int(N):
            break
        keep.add(i)
n_keep = -(-len(keep) // 64) * 64
for i in ranked:  # round up with the next most frequent tokens
    if len(keep) >= n_keep:
        break
    keep.add(i)
i = 0
while len(keep) < n_keep:  # corpus exhausted: fill with the lowest unused ids
    keep.add(i)
    i += 1
ids = sorted(keep)
covered = sum(c for i, c in counts.items() if i in keep)
print(f"{os.path.basename(IN)}: vocab {n_vocab}, output.weight {out_t['dims']} Q4_0, {row_bytes} bytes per row")
print(f"corpus: {cj['total']} tokens, {len(counts)} distinct; always kept: {len(always)} tokens (control, user-defined, --classes)")
print(f"shortlist: {len(ids)} tokens = {100 * len(ids) / n_vocab:.1f} % of the head, "
      f"covers {100 * covered / cj['total']:.3f} % of the corpus tokens")

if ev:
    url = ev[0]
    for path in ev[1:]:
        req = urllib.request.Request(url + "/tokenize", json.dumps({"content": open(path, encoding="utf-8").read()}).encode(),
                                     {"content-type": "application/json"})
        toks = json.load(urllib.request.urlopen(req, timeout=120))["tokens"]
        miss = [t for t in toks if t not in keep]
        shown = ", ".join(repr(tokens[t]) for t in dict.fromkeys(miss))[:300]
        print(f"eval {os.path.basename(path)}: {len(toks)} tokens, {len(miss)} outside the list "
              f"({100 * len(miss) / max(len(toks), 1):.2f} %){': ' + shown if miss else ''}")

if not OUT:
    print("dry run: no file written")
    sys.exit(0)
assert not os.path.exists(OUT), f"{OUT} exists"

# ---- write: header and metadata verbatim (+1 key), tensor infos with new offsets, data
new = [dict(t) for t in tensors]
for t in new:
    if t["name"] == "output.weight":
        t["dims"] = [t["dims"][0], len(ids)]
        t["size"] = len(ids) * row_bytes
new.append(dict(name="d2t", dims=[len(ids)], type=27, size=8 * len(ids), off=None))
cur = 0
for t in new:
    cur = -(-cur // align) * align
    t["new_off"] = cur
    cur += t["size"]

note = f"{len(ids)} of {n_vocab} tokens; frequencies from {cj['total']} corpus tokens"
with open(OUT, "wb") as o:
    o.write(b"GGUF" + struct.pack("<IQQ", version, len(new), n_kv + 1))
    f.seek(kv_start)
    o.write(f.read(kv_end - kv_start))
    o.write(pstr("qwen4exp.draft_shortlist") + struct.pack("<I", 8) + pstr(note))
    for t in new:
        o.write(pstr(t["name"]) + struct.pack("<I", len(t["dims"])) + struct.pack("<%dQ" % len(t["dims"]), *t["dims"])
                + struct.pack("<IQ", t["type"], t["new_off"]))
    o.write(b"\0" * (-o.tell() % align))
    base = o.tell()
    for t in new:
        o.write(b"\0" * (base + t["new_off"] - o.tell()))
        if t["name"] == "d2t":
            o.write(struct.pack("<%dq" % len(ids), *ids))
        elif t["name"] == "output.weight":
            for i in ids:
                f.seek(data_start + t["off"] + i * row_bytes)
                o.write(f.read(row_bytes))
        else:
            f.seek(data_start + t["off"])
            left = t["size"]
            while left:
                buf = f.read(min(left, 1 << 24))
                o.write(buf)
                left -= len(buf)
    o.write(b"\0" * (-o.tell() % align))
print(f"wrote {OUT}: {os.path.getsize(OUT) / 1e9:.2f} GB (input {file_size / 1e9:.2f} GB), {note}")
