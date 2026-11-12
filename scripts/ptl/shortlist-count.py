#!/usr/bin/env python3
"""Count token frequencies of a text corpus with a llama-server's tokenizer.

Usage: shortlist-count.py URL OUT.json SPEC [SPEC ...]
  SPEC = PATH[:MAX_MB[:GLOB[,GLOB...]]]   a file or a directory (walked), at most MAX_MB
         megabytes of text taken from it (default 5), only names matching a GLOB
         (default: every file that decodes as text)
Markup is stripped from .page/.xml/.html/.docbook files, roff requests from man pages
(.gz is unpacked). From gettext .mo catalogues the translated strings are taken. A word
list under /dict/ is expanded to " word Word" and "word" lines, so that the tokens of
whole words are seen with and without a leading blank.
OUT.json: {"total": n, "per_spec": {...}, "counts": {token_id: n}}. Existing counts in
OUT.json are kept and added to (run it again with more specs)."""
import fnmatch
import gzip
import json
import os
import re
import struct
import sys
import urllib.request

URL, OUT = sys.argv[1], sys.argv[2]
CHUNK = 48_000  # characters per /tokenize request
NUL = chr(0)
NL = chr(10)


def tokenize(text):
    req = urllib.request.Request(URL + "/tokenize", json.dumps({"content": text}).encode(),
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["tokens"]


def read_mo(raw):
    """Translated strings of a gettext catalogue."""
    if len(raw) < 20:
        return None
    magic = struct.unpack("<I", raw[:4])[0]
    end = "<" if magic == 0x950412DE else ">" if magic == 0xDE120495 else None
    if not end:
        return None
    _, n, _, off_t = struct.unpack(end + "4I", raw[4:20])
    out = []
    for i in range(n):
        ln, off = struct.unpack(end + "2I", raw[off_t + 8 * i: off_t + 8 * i + 8])
        out.extend(raw[off: off + ln].decode("utf-8", "ignore").split(NUL))
    return NL.join(out[1:])  # entry 0 is the catalogue header


def read_text(path):
    if path.endswith(".mo"):
        try:
            return read_mo(open(path, "rb").read())
        except (OSError, struct.error):
            return None
    if "/dict/" in path:
        words = [w for w in open(path, encoding="utf-8", errors="ignore").read().split() if w.isalpha()]
        return "".join(" " + w + " " + w.capitalize() + NL + w + NL for w in words)
    try:
        raw = gzip.open(path, "rb").read() if path.endswith(".gz") else open(path, "rb").read()
    except OSError:
        return None
    if bytes(1) in raw[:4096]:
        return None
    text = raw.decode("utf-8", "ignore")
    low = path.lower()
    if low.endswith((".page", ".xml", ".html", ".docbook", ".xhtml")):
        text = re.sub("<[^>]+>", " ", text)
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"')
        text = re.sub("[ ]+" + NL, NL, text)
        text = re.sub(NL + "[ " + NL + "]*" + NL, NL + NL, text)
    elif "/man" in low:
        text = NL.join(ln for ln in text.splitlines() if not ln.startswith((".", "'")))
    return text


def files_of(path, globs):
    if os.path.isfile(path):
        yield path
        return
    for root, dirs, names in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "build", "node_modules", "__pycache__"))
        for n in sorted(names):
            if not globs or any(fnmatch.fnmatch(n, g) for g in globs):
                yield os.path.join(root, n)


data = {"total": 0, "per_spec": {}, "counts": {}}
if os.path.exists(OUT):
    data = json.load(open(OUT))
counts = {int(k): v for k, v in data["counts"].items()}

for spec in sys.argv[3:]:
    parts = spec.split(":")
    path = parts[0]
    cap = float(parts[1]) * 1e6 if len(parts) > 1 and parts[1] else 5e6
    globs = parts[2].split(",") if len(parts) > 2 else None
    chars = toks = nfiles = 0
    buf = ""
    for f in files_of(path, globs):
        if chars >= cap:
            break
        text = read_text(f)
        if not text or len(text) < 200:
            continue
        text = text[: int(cap - chars)]
        chars += len(text)
        nfiles += 1
        buf += text + NL
        while len(buf) >= CHUNK:
            cut = buf.rfind(NL, 0, CHUNK) + 1 or CHUNK
            for t in tokenize(buf[:cut]):
                counts[t] = counts.get(t, 0) + 1
                toks += 1
            buf = buf[cut:]
    if buf.strip():
        for t in tokenize(buf):
            counts[t] = counts.get(t, 0) + 1
            toks += 1
    data["per_spec"][spec] = {"files": nfiles, "chars": chars, "tokens": toks}
    data["total"] += toks
    print(f"{spec}: {nfiles} files, {chars / 1e6:.2f} MB, {toks} tokens; distinct so far {len(counts)}", flush=True)
    data["counts"] = {str(k): v for k, v in counts.items()}
    json.dump(data, open(OUT, "w"))
print(f"total {data['total']} tokens, {len(counts)} distinct")
