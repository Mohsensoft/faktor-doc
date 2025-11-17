import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import re
import os
import html

BASE_URL = "https://mohsensoft.com/docs/faktor/"
OUTPUT_FILE = "chunks.jsonl"
HTML_FILE = "chunks.html"

# -------------------------------
# 1) Crawl all pages inside faktor/
# -------------------------------

def get_page_links(start_url):
    print("Crawling:", start_url)
    response = requests.get(start_url, timeout=10)
    response.raise_for_status()
    response.encoding = "utf-8"
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]

        # تبدیل به لینک کامل
        full_url = urljoin(start_url, href)

        # فقط لینک های داخل faktor/
        if BASE_URL in full_url:
            # حذف #hash
            clean = full_url.split("#")[0]
            links.add(clean)

    return list(links)


# -------------------------------
# 2) Extract clean text from HTML
# -------------------------------

def extract_clean_text(url):
    print("Extracting:", url)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response.encoding = "utf-8"
        html = response.text
    except:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # حذف اسکریپت/استایل
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text("\n")

    # تمیز کردن
    text = re.sub(r"\n\s*\n", "\n", text)  # حذف خطوط خالی زیاد
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# -------------------------------
# 3) Chunking function
# -------------------------------

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


# -------------------------------
# 4) Save JSONL chunks
# -------------------------------

def save_jsonl(chunks):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_html(chunks):
    rows = []
    for item in chunks:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['id'])}</td>"
            f"<td><a href=\"{html.escape(item['url'])}\">{html.escape(item['url'])}</a></td>"
            f"<td>{html.escape(item['content'])}</td>"
            "</tr>"
        )

    html_content = """<!DOCTYPE html>
<html lang="fa">
<head>
    <meta charset="utf-8">
    <title>فاکتور - خروجی چانک‌ها</title>
    <style>
        body { font-family: sans-serif; direction: rtl; margin: 24px; }
        table { width: 100%%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
        th { background-color: #f5f5f5; }
        tr:nth-child(even) { background-color: #fafafa; }
        a { color: #0062cc; text-decoration: none; }
    </style>
</head>
<body>
    <h1>جدول چانک‌ها</h1>
    <table>
        <thead>
            <tr>
                <th>شناسه</th>
                <th>لینک</th>
                <th>محتوا</th>
            </tr>
        </thead>
        <tbody>
            %s
        </tbody>
    </table>
</body>
</html>""" % "\n".join(rows)

    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)


# -------------------------------
# Main
# -------------------------------

def main():
    print("Fetching list of pages...")
    pages = get_page_links(BASE_URL)
    pages = sorted(set(pages))

    all_chunks = []
    
    for url in pages:
        text = extract_clean_text(url)
        if not text or len(text) < 50:
            continue

        page_slug = url.rstrip("/").split("/")[-1]
        chunks = chunk_text(text)

        for i, ch in enumerate(chunks):
            all_chunks.append({
                "id": f"{page_slug}_{i+1:04d}",
                "url": url,
                "content": ch
            })

    save_jsonl(all_chunks)
    save_html(all_chunks)

    print(f"\nDone! Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
