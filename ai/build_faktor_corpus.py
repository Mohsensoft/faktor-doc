import argparse
import re
import time
import json
from collections import deque
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
from readability import Document
from markdownify import markdownify as md
from tqdm import tqdm

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

# tiktoken for token-aware chunking
try:
    import tiktoken
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None  # fallback to char-length chunking if not installed


def normalize_url(url, base=None):
    if base:
        url = urljoin(base, url)
    url, _ = urldefrag(url)  # drop #fragments
    # drop querystrings for docs crawling (optional)
    parsed = urlparse(url)
    url = parsed._replace(query="").geturl()
    # normalize trailing slash for directories
    if parsed.path.endswith("/index.html"):
        url = url.replace("/index.html", "/")
    return url


def is_doc_url(url, allowed_prefix):
    if not url.startswith(allowed_prefix):
        return False
    # skip assets
    for ext in (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".zip", ".mp4", ".webm", ".css", ".js", ".ico"):
        if url.lower().endswith(ext):
            return False
    return True


def detect_encoding(content, default="utf-8"):
    """Detect encoding from content or headers"""
    # Ensure content is bytes for detection
    if isinstance(content, str):
        # If already decoded, assume it was correctly decoded as UTF-8
        return "utf-8"
    
    # First try to detect from HTML meta tags
    try:
        # Use a small sample to avoid parsing huge files
        sample = content[:8192] if len(content) > 8192 else content
        soup = BeautifulSoup(sample, "html.parser", from_encoding="utf-8")
        meta_charset = soup.find("meta", charset=True)
        if meta_charset:
            detected_enc = meta_charset.get("charset", "").lower()
            if detected_enc:
                return detected_enc
        
        meta_content = soup.find("meta", attrs={"http-equiv": re.compile(r"content-type", re.I)})
        if meta_content:
            content_type = meta_content.get("content", "")
            match = re.search(r'charset=([^;]+)', content_type, re.I)
            if match:
                return match.group(1).strip().lower()
    except Exception:
        pass  # Fall through to chardet
    
    # Fallback to chardet if available
    if HAS_CHARDET:
        try:
            detected = chardet.detect(content)
            if detected and detected.get("encoding") and detected.get("confidence", 0) > 0.5:
                return detected["encoding"].lower()
        except Exception:
            pass
    
    return default


def fetch(url, session, retries=3, timeout=25):
    headers = {"User-Agent": "Mohsensoft-Faktor-DocScraper/1.0 (+for RAG embedding)"}
    last_exc = None
    for _ in range(retries):
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                last_modified = resp.headers.get("Last-Modified")
                if last_modified:
                    try:
                        last_modified = parsedate_to_datetime(last_modified).isoformat()
                    except Exception:
                        last_modified = None
                
                # Detect encoding properly
                encoding = resp.encoding or detect_encoding(resp.content)
                # Handle common encoding variations
                if encoding.lower() in ("iso-8859-1", "latin-1") and HAS_CHARDET:
                    # Often mis-detected, re-check
                    detected = chardet.detect(resp.content)
                    if detected and detected.get("confidence", 0) > 0.7:
                        encoding = detected["encoding"]
                
                # Decode with detected/fallback encoding
                try:
                    text = resp.content.decode(encoding)
                except (UnicodeDecodeError, LookupError):
                    # Fallback chain: utf-8, windows-1256 (Arabic), iso-8859-1
                    for fallback_enc in ["utf-8", "windows-1256", "iso-8859-1"]:
                        try:
                            text = resp.content.decode(fallback_enc)
                            break
                        except (UnicodeDecodeError, LookupError):
                            continue
                    else:
                        # Last resort: decode with errors='replace'
                        text = resp.content.decode("utf-8", errors="replace")
                
                return text, last_modified
            elif resp.status_code in (301, 302, 303, 307, 308):
                # requests follows redirects by default
                pass
        except Exception as e:
            last_exc = e
        time.sleep(0.7)
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def make_links_absolute(soup, base_url):
    # anchors
    for a in soup.find_all("a", href=True):
        a["href"] = urljoin(base_url, a["href"])
    # images
    for img in soup.find_all("img", src=True):
        img["src"] = urljoin(base_url, img["src"])


def strip_boilerplate(soup):
    # remove obvious non-content areas
    for selector in ["nav", "header", "footer", "aside", "script", "style", "noscript"]:
        for tag in soup.select(selector):
            tag.decompose()
    # common doc-site classes to drop (best-effort)
    for cls in ["sidebar", "toc", "TableOfContents", "breadcrumbs", "pagination", "menu", "ad", "cookie", "newsletter"]:
        for tag in soup.find_all(True, class_=lambda c: c and cls in c if isinstance(c, str) else False):
            tag.decompose()


def promote_article_or_main(soup):
    # try to narrow to main content area if present
    for selector in ["article", "main", "div.markdown", "div.md-content", "section[role=main]"]:
        el = soup.select_one(selector)
        if el:
            return el
    return soup.body or soup


def extract_title(soup, fallback=None):
    # prefer h1 in content
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    # fallback to <title>
    t = soup.find("title")
    if t and t.get_text(strip=True):
        return t.get_text(" ", strip=True)
    return fallback or ""


def convert_pre_to_code_fences(soup):
    # Replace <pre><code>...</code></pre> or <pre>...</pre> with markdown fences
    pres = soup.find_all("pre")
    for pre in pres:
        code_el = pre.find("code")
        classes = []
        if pre.has_attr("class"):
            classes.extend(pre.get("class", []))
        if code_el and code_el.has_attr("class"):
            classes.extend(code_el.get("class", []))
        lang = ""
        for c in classes:
            m = re.match(r"(language|lang)-([A-Za-z0-9_+\-]+)", c)
            if m:
                lang = m.group(2)
                break
        code_text = code_el.get_text("\n", strip=False) if code_el else pre.get_text("\n", strip=False)
        fence = soup.new_string(f"\n\n```{lang}\n{code_text}\n```\n\n")
        pre.replace_with(fence)


def html_to_markdown(html, base_url):
    # Ensure HTML is a string (not bytes) and properly encoded
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    
    # Readability to isolate main content
    doc = Document(html)
    content_html = doc.summary(html_partial=True)
    # Parse with explicit UTF-8 to preserve Persian/Arabic characters
    content_soup = BeautifulSoup(content_html, "html.parser", from_encoding="utf-8")

    # Clean & prep
    strip_boilerplate(content_soup)
    make_links_absolute(content_soup, base_url)
    convert_pre_to_code_fences(content_soup)
    content_root = promote_article_or_main(content_soup)

    # Convert to markdown - ensure proper unicode handling
    content_str = str(content_root)
    # Double-check it's properly decoded
    if isinstance(content_str, bytes):
        content_str = content_str.decode("utf-8", errors="replace")
    
    markdown = md(content_str, heading_style="ATX", bullets="*")
    # Post-clean
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return markdown


def get_headings(markdown):
    # Extract headings for context/breadcrumbs
    heads = []
    for line in markdown.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line.strip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            heads.append({"level": level, "title": title})
    return heads


def tokenize_len(text):
    if ENCODING:
        return len(ENCODING.encode(text))
    return max(1, len(text) // 4)  # rough fallback


def chunk_markdown(text, url, title, max_tokens=900, overlap_tokens=100):
    # Paragraph-aware chunking with token budget
    paras = re.split(r"\n\s*\n", text)
    chunks = []
    buf = []
    buf_tokens = 0

    def flush():
        nonlocal buf, buf_tokens
        if not buf:
            return None
        chunk_text = "\n\n".join(buf).strip()
        chunks.append(chunk_text)
        buf = []
        buf_tokens = 0
        return chunk_text

    for p in paras:
        t = tokenize_len(p)
        if buf_tokens + t <= max_tokens:
            buf.append(p)
            buf_tokens += t
        else:
            # flush current
            last_chunk = flush()
            # overlap from end of last_chunk
            if last_chunk and overlap_tokens > 0 and ENCODING:
                tok = ENCODING.encode(last_chunk)
                overlap = ENCODING.decode(tok[-overlap_tokens:]) if tok else ""
                if overlap:
                    buf.append(overlap)
                    buf_tokens = tokenize_len(overlap)
                else:
                    buf_tokens = 0
            # add current paragraph
            buf.append(p)
            buf_tokens += t
    flush()

    # Wrap into records
    records = []
    for i, ct in enumerate(chunks):
        records.append({
            "id": f"{url}#chunk-{i+1}",
            "url": url,
            "title": title,
            "chunk_index": i + 1,
            "text": ct,
            "num_tokens": tokenize_len(ct),
        })
    return records


def crawl_and_build(base_url, start_url, out_dir, max_pages=1000):
    allowed_prefix = normalize_url(base_url)
    start = normalize_url(start_url, base=allowed_prefix)

    session = requests.Session()
    q = deque([start])
    seen = set()
    pages = []

    pbar = tqdm(total=max_pages, desc="Crawling")
    while q and len(pages) < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)
        try:
            html, last_modified = fetch(url, session)
        except Exception:
            continue

        # Parse with explicit UTF-8 encoding
        soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")
        # discover links
        for a in soup.find_all("a", href=True):
            u = normalize_url(a["href"], base=url)
            if is_doc_url(u, allowed_prefix) and u not in seen:
                q.append(u)

        # extract title and markdown
        try:
            title = extract_title(soup) or url
            markdown = html_to_markdown(html, url)
            if not markdown or len(markdown) < 50:
                # fallback: use whole body if too short
                body = soup.body or soup
                body_str = str(body)
                if isinstance(body_str, bytes):
                    body_str = body_str.decode("utf-8", errors="replace")
                markdown = md(body_str, heading_style="ATX", bullets="*")
        except Exception:
            continue

        headings = get_headings(markdown)
        pages.append({
            "url": url,
            "title": title,
            "last_modified": last_modified,
            "markdown": markdown,
            "headings": headings,
        })
        pbar.update(1)
        time.sleep(0.2)  # be gentle
    pbar.close()

    # Write a single Markdown file (human-friendly)
    md_path = f"{out_dir}/faktor-docs.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Faktor Documentation (Mohsensoft)\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Source base: {allowed_prefix}\n\n")
        for p in pages:
            f.write("\n\n---\n\n")
            f.write(f"## {p['title']}\n")
            #f.write(f"Source: {p['url']}\n")
            #if p["last_modified"]:
            #    f.write(f"Last-Modified: {p['last_modified']}\n")
            f.write("\n")
            f.write(p["markdown"])
            f.write("\n")

    # Build chunked JSONL (machine-friendly for embeddings)
    jsonl_path = f"{out_dir}/faktor-docs.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fw:
        for p in pages:
            chunks = chunk_markdown(
                f"# {p['title']}\n\n{p['markdown']}",
                p["url"], p["title"],
                max_tokens=900, overlap_tokens=120
            )
            for ch in chunks:
                rec = {
                    "id": ch["id"],
                    "url": ch["url"],
                    "title": ch["title"],
                    "chunk_index": ch["chunk_index"],
                    "text": ch["text"],
                    "num_tokens": ch["num_tokens"],
                    "last_modified": p["last_modified"],
                }
                fw.write(json.dumps(rec, ensure_ascii=False) + ",\n")

    # Also write a minimal sitemap of collected pages
    with open(f"{out_dir}/faktor-pages.json", "w", encoding="utf-8") as fp:
        json.dump(
            [{"url": p["url"], "title": p["title"], "last_modified": p["last_modified"]} for p in pages],
            fp, ensure_ascii=False, indent=2
        )

    return {
        "pages": len(pages),
        "md_path": md_path,
        "jsonl_path": jsonl_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="https://mohsensoft.com/docs/faktor/")
    parser.add_argument("--start", default="https://mohsensoft.com/docs/faktor/index.html")
    parser.add_argument("--out", default="out")
    parser.add_argument("--max_pages", type=int, default=1000)
    args = parser.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    stats = crawl_and_build(args.base, args.start, args.out, max_pages=args.max_pages)
    print(f"Done. Collected {stats['pages']} pages.")
    print(f"- Markdown: {stats['md_path']}")
    print(f"- JSONL:    {stats['jsonl_path']}")


if __name__ == "__main__":
    main()