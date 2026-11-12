#!/usr/bin/env python3
"""Classify the tokens of a Qwen GGUF by writing system, to size a draft shortlist.

Usage: shortlist-scripts.py MODEL.gguf [COUNTS.json]
Prints how many tokens are ASCII only, Latin with accents, symbols, Greek, or another
script, and (with COUNTS.json from shortlist-count.py) how many of each were seen."""
import json
import struct
import sys

f = open(sys.argv[1], "rb")
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


assert f.read(4) == b"GGUF"
_, _, n_kv = rd("I"), rd("Q"), rd("Q")
meta = {}
for _ in range(n_kv):
    k, t = rstr(), rd("I")
    meta[k] = rval(t)
tokens, ttype = meta["tokenizer.ggml.tokens"], meta["tokenizer.ggml.token_type"]

# GPT-2 style byte <-> character table used by the BPE vocabulary
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
cs, n = bs[:], 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
CH2B = {chr(c): b for b, c in zip(bs, cs)}


def token_class(tok, typ):
    if typ in (3, 4):
        return "control"
    if typ != 1:
        return "unused"
    try:
        raw = bytes(CH2B[c] for c in tok)
    except KeyError:
        return "other"
    cls = "ascii"
    for b in raw:
        if b < 0x80 or 0x80 <= b <= 0xBF:
            continue  # ASCII, or a continuation byte (judged by its lead byte)
        if b in (0xC2, 0xC3):
            c = "latin"       # U+0080-00FF: umlauts, accents, degree, micro, section
        elif b == 0xE2:
            c = "symbol"      # U+2000-2FFF: dashes, quotes, arrows, math, box drawing
        elif b in (0xCE, 0xCF):
            c = "greek"
        else:
            return "other"    # every other script, fullwidth forms, emoji
        if cls == "ascii" or (cls == "latin" and c != "latin"):
            cls = c
    if raw and all(0x80 <= b <= 0xBF for b in raw):
        return "fragment"     # continuation bytes only
    return cls


counts = {}
if len(sys.argv) > 2:
    counts = {int(k): v for k, v in json.load(open(sys.argv[2]))["counts"].items()}
tot = {}
for i, (tok, typ) in enumerate(zip(tokens, ttype)):
    c = token_class(tok, typ)
    a = tot.setdefault(c, [0, 0, 0])
    a[0] += 1
    if i in counts:
        a[1] += 1
        a[2] += counts[i]
print(f"{len(tokens)} tokens")
for c, (n_tok, seen, occ) in sorted(tot.items(), key=lambda x: -x[1][0]):
    print(f"  {c:<9} {n_tok:>7} tokens ({100 * n_tok / len(tokens):5.1f} %)   seen in corpus: {seen:>6} tokens, {occ:>8} occurrences")
if __name__ == "__main__" and len(sys.argv) > 3:
    json.dump({c: [i for i, (tok, typ) in enumerate(zip(tokens, ttype)) if token_class(tok, typ) == c] for c in tot},
              open(sys.argv[3], "w"))
    print("classes written to", sys.argv[3])
