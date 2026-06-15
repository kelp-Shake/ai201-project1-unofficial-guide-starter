"""
The Unofficial Guide - Milestone 3: Ingestion + Cleaning + Chunking

Two-track, structure-aware chunking (per planning.md):
  - Reddit threads: title + OP body + one chunk per comment. A comment stays
    whole unless it exceeds the size cap, in which case it is sub-split.
  - Official pages (.md / .txt): section-based on markdown headers, size-capped.

Size-based splitting/sub-splitting uses LangChain RecursiveCharacterTextSplitter.

Run:  python3 chunk_pipeline.py
Output: prints stats + 5 sample chunks, writes documents/chunks.json
"""
import os
import re
import json
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]

# ---- Config (starting values from planning.md; verify at the inspection step) ----
REDDIT_SIZE_CAP = 500     # a comment longer than this gets sub-split
REDDIT_SUBSPLIT_OVERLAP = 30
PAGE_SIZE_CAP = 500
PAGE_OVERLAP = 75         # ~15%
MIN_CHUNK_CHARS = 40      # drop fragments shorter than this

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")

# ---- Size splitter ----
def size_split(text, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]

SPLITTER = "langchain RecursiveCharacterTextSplitter"


# ---- Cleaning helpers ----
def strip_images(text):
    return re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

def strip_md_links(text):
    # [label](url) -> label  (keeps anchor text, drops the URL)
    return re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

def collapse_ws(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Boilerplate nav block shared across the official pages; everything from the
# "Career Resources" footer link list onward is repeated site chrome.
def trim_page_boilerplate(text):
    cut = len(text)
    for marker in ["## Career Resources", "## Contact Options"]:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


# ---- Reddit parsing ----
def parse_reddit(path, raw):
    lines = raw.splitlines()
    title = lines[0].lstrip("#").strip() if lines else ""
    title = strip_md_links(title)

    author_m = re.search(r"\*\*Author:\*\*\s*\[([^\]]+)\]", raw)
    op_author = author_m.group(1) if author_m else "Unknown"

    # OP body sits between the first '---' after the header block and the
    # 'Comments' marker. Not every thread has one (link/image posts don't).
    body = ""
    cm = re.search(r"##\s*Comments", raw)
    comments_start = cm.start() if cm else len(raw)
    header_end = raw.find("---")
    if header_end != -1:
        # skip the header's own --- and find the body region before Comments
        region = raw[header_end:comments_start]
        # body is the text between the LAST '---' in the header region and Comments
        segs = [s.strip() for s in region.split("---")]
        cand = [s for s in segs if s and not s.startswith("###")
                and "Subreddit:" not in s and "Author:" not in s
                and "Vote:" not in s and not s.startswith("![")]
        body = strip_images(strip_md_links("\n".join(cand))).strip()

    # Comments: each begins with "- by [author](#) **... NN**" then text on
    # following lines (often after <br/>), until the next "- by".
    comment_block = raw[comments_start:]
    raw_comments = re.split(r"\n-\s+by\s+", comment_block)[1:]
    comments = []
    for rc in raw_comments:
        # drop the vote header line, keep the body
        txt = re.sub(r"^\[[^\]]*\]\(#\)\s*\*\*[^*]*\*\*", "", rc)
        txt = txt.replace("<br/>", "\n")
        txt = strip_images(strip_md_links(txt))
        txt = collapse_ws(txt)
        if len(txt) >= MIN_CHUNK_CHARS:
            comments.append(txt)

    chunks = []
    src = os.path.basename(path)

    # title + OP body as one contextual chunk (so a query can match the topic)
    head = title + (("\n\n" + body) if body else "")
    for piece in size_split(head, PAGE_SIZE_CAP, REDDIT_SUBSPLIT_OVERLAP):
        chunks.append({"text": piece, "source": src, "doc_type": "reddit",
                       "section": "title+op", "author": op_author})

    for idx, c in enumerate(comments):
        if len(c) <= REDDIT_SIZE_CAP:
            pieces = [c]
        else:
            pieces = size_split(c, REDDIT_SIZE_CAP, REDDIT_SUBSPLIT_OVERLAP)
        for j, p in enumerate(pieces):
            chunks.append({"text": p, "source": src, "doc_type": "reddit",
                           "section": f"comment_{idx}" + (f"_{j}" if len(pieces) > 1 else ""),
                           "author": "unknown"})
    return chunks


# ---- Official page parsing (markdown headers) ----
def parse_page(path, raw):
    src = os.path.basename(path)
    text = trim_page_boilerplate(raw)
    text = strip_images(text)
    text = strip_md_links(text)

    # Split into sections on # / ## / ### headers, keeping the header as the
    # section name. blocks alternates: [pre, header, body, header, body, ...]
    sections = []
    blocks = re.split(r"(?m)^(#{1,3} .+)$", text)
    header = "intro"
    for seg in blocks:
        if re.match(r"^#{1,3} ", seg or ""):
            header = seg.lstrip("#").strip()
        else:
            body = collapse_ws(seg or "")
            if len(body) >= MIN_CHUNK_CHARS:
                sections.append((header, body))

    chunks = []
    for header, body in sections:
        labeled = f"{header}. {body}" if header != "intro" else body
        for piece in size_split(labeled, PAGE_SIZE_CAP, PAGE_OVERLAP):
            chunks.append({"text": piece, "source": src, "doc_type": "page",
                           "section": header})
    return chunks


# ---- LinkedIn .txt (=== SECTION === delimited) ----
def parse_linkedin(path, raw):
    src = os.path.basename(path)
    blocks = re.split(r"\n=== (.+?) ===\n", raw)
    chunks = []
    # blocks: [preamble, header, body, header, body, ...]
    pre = blocks[0]
    pairs = list(zip(blocks[1::2], blocks[2::2]))
    if pre.strip():
        pairs = [("HEADER", pre)] + pairs
    for header, body in pairs:
        body = collapse_ws(strip_md_links(body))
        if len(body) < MIN_CHUNK_CHARS:
            continue
        labeled = f"{header}. {body}"
        for piece in size_split(labeled, PAGE_SIZE_CAP, PAGE_OVERLAP):
            chunks.append({"text": piece, "source": src, "doc_type": "page",
                           "section": header})
    return chunks


def is_reddit(raw):
    return "**Subreddit:**" in raw or "## Comments" in raw

def main():
    files = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.md"))
                   + glob.glob(os.path.join(UPLOAD_DIR, "*.txt")))
    all_chunks = []
    per_file = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        if path.endswith("linkedin.txt"):
            ch = parse_linkedin(path, raw)
        elif is_reddit(raw):
            ch = parse_reddit(path, raw)
        else:
            ch = parse_page(path, raw)
        # Drop sub-threshold fragments that slip through after size-splitting
        # and label-prefixing (the body filters run before those steps).
        ch = [c for c in ch if len(c["text"]) >= MIN_CHUNK_CHARS]
        per_file[os.path.basename(path)] = len(ch)
        all_chunks.extend(ch)

    with open(os.path.join(UPLOAD_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Splitter: {SPLITTER}")
    print(f"Documents processed: {len(files)}")
    print(f"Total chunks: {len(all_chunks)}\n")
    print("Chunks per document:")
    for name, n in per_file.items():
        print(f"  {n:3d}  {name[:70]}")

    lengths = [len(c["text"]) for c in all_chunks]
    print(f"\nChunk length chars: min {min(lengths)}, "
          f"mean {sum(lengths)//len(lengths)}, max {max(lengths)}")

    print("\n" + "=" * 70)
    print("5 SAMPLE CHUNKS (inspect: is each self-contained?)")
    print("=" * 70)
    import random
    random.seed(7)
    for c in random.sample(all_chunks, 5):
        print(f"\n[{c['doc_type']} | {c['section']} | {c['source'][:40]}]")
        print(f"({len(c['text'])} chars) {c['text'][:400]}")

if __name__ == "__main__":
    main()
