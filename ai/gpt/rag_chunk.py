"""
faktor_rag_chunk_and_index.py

Purpose: Crawl Mohsensoft Faktor documentation, extract text, chunk it with overlap,
create OpenAI embeddings (text-embedding-3-large by default), and index vectors
with FAISS + metadata saved in SQLite/JSON. Also provides a `query` function
that returns the top-k matching chunks (metadata + text) which you can feed
into an LLM for RAG.

Usage steps:
1. pip install -r requirements.txt
   requirements.txt should include:
     requests
     beautifulsoup4
     openai
     faiss-cpu    # or faiss if your platform provides it
     numpy
     tiktoken     # optional but recommended for token-aware chunking

2. Export your OpenAI API key:
   export OPENAI_API_KEY="sk-..."   (Linux/macOS)
   setx OPENAI_API_KEY "sk-..."      (Windows - use PowerShell or cmd)

3. Run:
   python faktor_rag_chunk_and_index.py --crawl-only   # just crawl + chunk + save
   python faktor_rag_chunk_and_index.py --index        # crawl+chunk+embed+index
   python faktor_rag_chunk_and_index.py --query "چطور نصب کنم" --topk 5

Note: This script is a starting point. Adjust page filtering and HTML selectors
based on the actual site structure if needed.

"""

import os
import re
import json
import argparse
import time
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import numpy as np

# OpenAI client
import openai

# FAISS
import faiss

# Optional token-aware chunking (faster and more accurate if installed)
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except Exception:
    TOKENIZER_AVAILABLE = False

# --------------------------- CONFIG ---------------------------------
BASE_URL = "https://mohsensoft.com/docs/faktor/"
ALLOWED_NETLOC = urlparse(BASE_URL).netloc
OUTPUT_DIR = "faktor_rag_output"
CHUNKS_JSON = os.path.join(OUTPUT_DIR, "chunks.json")
METADATA_JSON = os.path.join(OUTPUT_DIR, "metadata.json")
FAISS_INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss.index")
VECTORS_NPY = os.path.join(OUTPUT_DIR, "vectors.npy")

# Chunk settings (tweakable)
MAX_WORDS = 500
OVERLAP_WORDS = 100
EMBED_MODEL = "text-embedding-3-large"  # default recommended
BATCH_SIZE = 16  # for embeddings batching

# --------------------------- UTIL -----------------------------------

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_url(url, timeout=15):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; faktor-rag/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def is_internal_link(href):
    if not href:
        return False
    parsed = urlparse(href)
    if parsed.netloc and parsed.netloc != ALLOWED_NETLOC:
        return False
    # keep only links under /docs/faktor/
    full = urljoin(BASE_URL, href)
    return urlparse(full).path.startswith(urlparse(BASE_URL).path)


def discover_pages(start_url, max_pages=200):
    print("Discovering pages under:", start_url)
    to_visit = [start_url]
    seen = set()
    pages = []
    while to_visit and len(pages) < max_pages:
        url = to_visit.pop(0)
        if url in seen:
            continue
        try:
            html = fetch_url(url)
        except Exception as e:
            print("Failed fetching", url, e)
            seen.add(url)
            continue
        seen.add(url)
        pages.append(url)
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get('href')
            if not href:
                continue
            full = urljoin(url, href)
            if full in seen:
                continue
            if is_internal_link(full) and full not in to_visit:
                to_visit.append(full)
    print(f"Discovered {len(pages)} pages")
    return pages


# Heuristic: extract main textual content. Try multiple selectors.
def extract_main_text(html):
    soup = BeautifulSoup(html, "html.parser")
    # remove scripts/styles
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # try common containers
    selectors = ["main", "article", "div.post", "div.content", "div#content", "div.docsearch-content"]
    content = None
    for sel in selectors:
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            content = node
            break
    if content is None:
        content = soup.body or soup

    # get title from h1 if present
    title_tag = content.find(["h1"]) if content else soup.find(["h1"]) 
    title = title_tag.get_text(strip=True) if title_tag else None

    # clean text lines
    text = content.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    cleaned = "\n".join(lines)
    return title, cleaned


# Simple whitespace tokenizer
_whitespace_re = re.compile(r"\S+")

def split_into_words(text):
    return _whitespace_re.findall(text)


def chunk_text_by_words(text, max_words=MAX_WORDS, overlap_words=OVERLAP_WORDS):
    words = split_into_words(text)
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        end = min(i + max_words, n)
        chunk_words = words[i:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        if end == n:
            break
        i = end - overlap_words
    return chunks


# token-aware chunking (if tiktoken available and you prefer tokens over words)
def chunk_text_by_tokens(text, max_tokens=500, overlap_tokens=100, model_name="gpt-4o-mini"):  # model_name only for tokenizer choice
    if not TOKENIZER_AVAILABLE:
        return chunk_text_by_words(text, max_words=max_tokens, overlap_words=overlap_tokens)
    enc = tiktoken.encoding_for_model(model_name)
    toks = enc.encode(text)
    chunks = []
    i = 0
    n = len(toks)
    while i < n:
        end = min(i + max_tokens, n)
        chunk_tokens = toks[i:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end == n:
            break
        i = end - overlap_tokens
    return chunks


# --------------------------- CHUNK CREATION --------------------------

def create_chunks_from_pages(pages, use_token_chunking=False):
    all_chunks = []
    cid = 0
    for url in pages:
        try:
            html = fetch_url(url)
        except Exception as e:
            print("fetch failed", url, e)
            continue
        title, cleaned = extract_main_text(html)
        if not cleaned or len(cleaned.split()) < 20:
            print("Skipping (too short):", url)
            continue
        if use_token_chunking:
            chunks = chunk_text_by_tokens(cleaned, max_tokens=MAX_WORDS, overlap_tokens=OVERLAP_WORDS)
        else:
            chunks = chunk_text_by_words(cleaned, max_words=MAX_WORDS, overlap_words=OVERLAP_WORDS)
        for i, c in enumerate(chunks):
            chunk_obj = {
                "id": f"chunk_{cid}",
                "source_url": url,
                "title": title,
                "chunk_index": i,
                "content": c
            }
            all_chunks.append(chunk_obj)
            cid += 1
    return all_chunks


# --------------------------- EMBEDDINGS ------------------------------

def get_openai_embeddings(texts, model=EMBED_MODEL, batch_size=BATCH_SIZE):
    """Batch texts and call OpenAI embeddings. Returns np.array of vectors."""
    if not isinstance(texts, list):
        texts = [texts]
    vectors = []
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # sleep a little to be polite
        time.sleep(0.1)
        resp = openai.Embeddings.create(model=model, input=batch)
        # resp['data'] is a list with embedding vectors
        for item in resp['data']:
            vectors.append(np.array(item['embedding'], dtype='float32'))
    return np.vstack(vectors)


# --------------------------- INDEXING --------------------------------

def build_faiss_index(vectors: np.ndarray):
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors -> cosine if normalized
    # normalize for cosine similarity
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


# --------------------------- SAVE / LOAD ------------------------------

def save_chunks_json(chunks, path=CHUNKS_JSON):
    ensure_output_dir()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Saved chunks to", path)


def save_metadata(chunks, path=METADATA_JSON):
    meta = [{"id": c['id'], "source_url": c['source_url'], "title": c.get('title'), "chunk_index": c.get('chunk_index')} for c in chunks]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved metadata to", path)


def save_vectors_npy(vectors, path=VECTORS_NPY):
    np.save(path, vectors)
    print("Saved vectors to", path)


def save_faiss(index, path=FAISS_INDEX_FILE):
    faiss.write_index(index, path)
    print("Saved FAISS index to", path)


# --------------------------- QUERY -----------------------------------

def query_topk(query_text, k=5, model=EMBED_MODEL, chunks_path=CHUNKS_JSON, vectors_path=VECTORS_NPY, index_path=FAISS_INDEX_FILE):
    # load metadata
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    vectors = np.load(vectors_path)
    # load index
    index = faiss.read_index(index_path)
    # embed query
    q_emb = get_openai_embeddings([query_text], model=model)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        # idx is integer index of vector
        results.append({
            "score": float(score),
            "chunk_id": chunks[idx]['id'],
            "source_url": chunks[idx]['source_url'],
            "title": chunks[idx].get('title'),
            "content": chunks[idx]['content']
        })
    return results


# --------------------------- MAIN ------------------------------------

def main(args):
    ensure_output_dir()
    if args.crawl:
        pages = discover_pages(BASE_URL)
        chunks = create_chunks_from_pages(pages, use_token_chunking=False)
        save_chunks_json(chunks)
        save_metadata(chunks)
        print(f"Created {len(chunks)} chunks")
        # If only crawl, exit
        if not args.index:
            return
    else:
        # if not crawling, try to load existing chunks
        with open(CHUNKS_JSON, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

    if args.index:
        texts = [c['content'] for c in chunks]
        print("Requesting embeddings for", len(texts), "chunks...")
        vectors = get_openai_embeddings(texts)
        print("Embeddings shape:", vectors.shape)
        save_vectors_npy(vectors)
        index = build_faiss_index(vectors.copy())
        save_faiss(index)
        save_metadata(chunks)
        print("Indexing complete.")

    if args.query:
        print("Querying for:", args.query)
        results = query_topk(args.query, k=args.topk)
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crawl', dest='crawl', action='store_true', help='crawl and chunk pages')
    parser.add_argument('--index', dest='index', action='store_true', help='create embeddings and index')
    parser.add_argument('--crawl-only', dest='crawl_only', action='store_true', help='only crawl and chunk (shorthand)')
    parser.add_argument('--query', type=str, default=None, help='run a sample query after indexing')
    parser.add_argument('--topk', type=int, default=5, help='top-k results')
    args = parser.parse_args()
    # convenience: if crawl-only -> set crawl True and index False
    if args.crawl_only:
        args.crawl = True
        args.index = False
    # default behavior: if no flags, do crawl+index
    if not any([args.crawl, args.index, args.query, args.crawl_only]):
        args.crawl = True
        args.index = True
    main(args)
