import argparse
import collections
import dataclasses
import json
import re
import time
from html import unescape
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_USER_AGENT = "DocCrawler/1.0 (+for-embedding; compatible)"


def build_http_session(user_agent: str) -> requests.Session:
	session = requests.Session()
	session.headers.update({"User-Agent": user_agent})
	retries = Retry(
		total=5,
		connect=5,
		read=5,
		backoff_factor=0.5,
		status_forcelist=[429, 500, 502, 503, 504],
		allowed_methods=["HEAD", "GET", "OPTIONS"],
		raise_on_status=False,
	)
	adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
	session.mount("http://", adapter)
	session.mount("https://", adapter)
	return session


def _parse_header_charset(content_type: Optional[str]) -> Optional[str]:
	if not content_type:
		return None
	# Example: text/html; charset=UTF-8
	m = re.search(r"charset=([\w\-\d:._]+)", content_type, flags=re.IGNORECASE)
	if m:
		return m.group(1).strip().strip("\"'")
	return None


def _detect_meta_charset(html_head_bytes: bytes) -> Optional[str]:
	# Search only in the first N bytes for performance
	sample = html_head_bytes.decode("ascii", errors="ignore")
	# Look for <meta charset="..."> or <meta http-equiv="Content-Type" content="text/html; charset=...">
	patterns = [
		r"<meta\s+charset=['\"]?([\w\-\d:._]+)['\"]?",
		r"<meta[^>]+http-equiv=['\"]?Content-Type['\"]?[^>]+charset=([\w\-\d:._]+)",
	]
	for pat in patterns:
		m = re.search(pat, sample, flags=re.IGNORECASE)
		if m:
			return m.group(1).strip().strip("\"'")
	return None


def choose_best_encoding(html_bytes: bytes, content_type_hdr: Optional[str], resp_encoding: Optional[str], apparent_encoding: Optional[str]) -> str:
	# 1) BOM
	if html_bytes.startswith(b"\xef\xbb\xbf"):
		return "utf-8-sig"
	# 2) Meta charset in head
	meta = _detect_meta_charset(html_bytes[:8192])
	if meta:
		return meta
	# 3) If utf-8 decodes cleanly, strongly prefer utf-8 (common when server lies with ISO-8859-1)
	try:
		html_bytes.decode("utf-8")
		return "utf-8"
	except Exception:
		pass
	# 4) Fallback order: apparent -> header charset -> resp.encoding -> utf-8
	header_charset = _parse_header_charset(content_type_hdr or "")
	for enc in (apparent_encoding, header_charset, resp_encoding):
		if enc:
			return enc
	return "utf-8"


def normalize_url(base_url: str, href: str) -> Optional[str]:
	if not href:
		return None
	if href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
		return None
	absolute = urljoin(base_url, href)
	absolute, _frag = urldefrag(absolute)
	parsed = urlparse(absolute)
	if not parsed.scheme.startswith("http"):
		return None
	return absolute


def _normalize_root_prefix(root: str) -> str:
	parsed = urlparse(root)
	# Ensure path ends with a slash to avoid partial prefix matches
	path = parsed.path if parsed.path else "/"
	if not path.endswith("/"):
		path = path + "/"
	return f"{parsed.scheme}://{parsed.netloc}{path}"


def has_url_prefix(candidate: str, root_prefix: str) -> bool:
	"""
	Returns True only if candidate starts with the exact scheme+host and the path prefix of root.
	Example:
	  root_prefix=https://docs.example.com/guide/
	  OK:     https://docs.example.com/guide/install
	  NOT OK: https://docs.example.com/tutorial/...
	  NOT OK: https://blog.example.com/guide/...
	"""
	cu = urlparse(candidate)
	rp = urlparse(root_prefix)
	if (cu.scheme, cu.netloc) != (rp.scheme, rp.netloc):
		return False
	# Ensure rp.path ends with '/'; _normalize_root_prefix ensures it when creating root_prefix
	root_path = rp.path if rp.path else "/"
	return (cu.path or "/").startswith(root_path)


def load_robots(root_url: str, user_agent: str) -> RobotFileParser:
	parsed = urlparse(root_url)
	robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
	rp = RobotFileParser()
	try:
		rp.set_url(robots_url)
		rp.read()
	except Exception:
		# Be permissive if robots not reachable; sites without robots.txt default to allowed.
		rp.parse([])
	rp.useragent = user_agent
	return rp


def is_allowed_by_robots(rp: RobotFileParser, url: str, user_agent: str) -> bool:
	# RobotFileParser.can_fetch accepts a UA string
	try:
		return rp.can_fetch(user_agent, url)
	except Exception:
		return True


def clean_text(text: str) -> str:
	text = unescape(text)
	text = re.sub(r"\s+", " ", text, flags=re.MULTILINE).strip()
	return text


def extract_title(soup: BeautifulSoup) -> str:
	if soup.title and soup.title.string:
		return clean_text(soup.title.string)
	# Fallback to first h1
	h1 = soup.find(["h1"])
	if h1:
		return clean_text(h1.get_text())
	return ""


@dataclasses.dataclass
class ParagraphRecord:
	url: str
	page_title: str
	heading_path: List[str]  # e.g., ["Getting Started", "Install"]
	paragraph_index: int
	text: str


def iter_text_blocks(node: Tag) -> Iterable[Tuple[str, str]]:
	"""
	Yields tuples of (block_type, text) where block_type in {"p", "li", "pre", "code", "blockquote"}
	Skips nav, header, footer, script, style, noscript.
	"""
	skips = {"script", "style", "noscript", "header", "footer", "nav", "form", "svg"}
	block_tags = {"p", "li", "pre", "code", "blockquote"}
	for el in node.find_all(True):
		if el.name in skips:
			continue
		if el.name in block_tags:
			# Avoid injecting artificial spaces between inline tags like <span>
			txt = clean_text(el.get_text())
			if txt:
				yield el.name, txt
		# Handle standalone text nodes under sectioning elements
		if el.name in {"section", "article"}:
			for child in el.children:
				if isinstance(child, NavigableString):
					txt = clean_text(str(child))
					if txt:
						yield "p", txt


def build_heading_path(el: Tag) -> List[str]:
	"""
	Given a heading element, walk backwards through previous siblings/parents
	to reconstruct a reasonable heading path.
	"""
	levels: Dict[str, int] = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
	path: List[Tuple[int, str]] = []

	def nearest_previous_headings(start: Tag) -> List[Tuple[int, str]]:
		result: List[Tuple[int, str]] = []
		cur: Optional[Tag] = start
		while cur is not None:
			prev = cur.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
			if prev is None:
				break
			level = levels.get(prev.name, 7)
			title = clean_text(prev.get_text())
			if title:
				result.append((level, title))
			cur = prev
		result.reverse()
		return result

	collected = nearest_previous_headings(el)
	for level, title in collected:
		path.append((level, title))
	return [t for _lvl, t in path]


def extract_paragraphs(url: str, html: "bytes|str", encoding_hint: Optional[str] = None) -> List[ParagraphRecord]:
	# Prefer parsing from bytes so BeautifulSoup can auto-detect or use encoding hints/meta
	if isinstance(html, (bytes, bytearray)):
		soup = BeautifulSoup(html, "html.parser", from_encoding=encoding_hint)
	else:
		soup = BeautifulSoup(html, "html.parser")
	page_title = extract_title(soup)
	# Prefer only the main article body if present
	content_root: Tag = soup.select_one('div[itemprop="articleBody"]') or soup.body or soup

	records: List[ParagraphRecord] = []
	para_idx = 0

	# Strategy:
	# - Traverse only h2 headings; for each h2 section, collect subsequent block texts until the next h2

	def siblings_until_next_heading(start: Tag) -> Iterable[Tag]:
		for sib in start.next_siblings:
			if isinstance(sib, NavigableString):
				if clean_text(str(sib)):
					yield sib
				continue
			if not isinstance(sib, Tag):
				continue
			# Stop only at the next h2 section
			if sib.name and sib.name.lower() in {"h2"}:
				break
			yield sib

	# Only consider h2 as section headers
	headings = content_root.find_all(["h2"])
	seen_blocks: Set[int] = set()

	for h in headings:
		# Use the h2 text as the section title
		section_heading_path = [clean_text(h.get_text())]
		for part in siblings_until_next_heading(h):
			if isinstance(part, NavigableString):
				txt = clean_text(str(part))
				if not txt:
					continue
				if id(part) in seen_blocks:
					continue
				seen_blocks.add(id(part))
				records.append(ParagraphRecord(url=url, page_title=page_title, heading_path=section_heading_path, paragraph_index=para_idx, text=txt))
				para_idx += 1
				continue
			if not isinstance(part, Tag):
				continue
			if id(part) in seen_blocks:
				continue
			seen_blocks.add(id(part))
			for _kind, txt in iter_text_blocks(part):
				records.append(ParagraphRecord(url=url, page_title=page_title, heading_path=section_heading_path, paragraph_index=para_idx, text=txt))
				para_idx += 1

	# Do not fallback to whole body; only extract content between h2 sections

	return records


def combine_by_char_limit(records: List[ParagraphRecord], max_chars: int) -> List[ParagraphRecord]:
	if max_chars <= 0:
		return records
	combined: List[ParagraphRecord] = []
	buffer: List[ParagraphRecord] = []
	buf_len = 0

	def flush():
		nonlocal buffer, buf_len
		if not buffer:
			return
		first = buffer[0]
		text = "\n\n".join(r.text for r in buffer)
		combined.append(ParagraphRecord(url=first.url, page_title=first.page_title, heading_path=first.heading_path, paragraph_index=first.paragraph_index, text=text))
		buffer = []
		buf_len = 0

	for rec in records:
		if not buffer:
			buffer.append(rec)
			buf_len = len(rec.text)
			continue
		same_context = (rec.url == buffer[-1].url) and (rec.heading_path == buffer[-1].heading_path)
		next_len = buf_len + 2 + len(rec.text)
		if same_context and next_len <= max_chars:
			buffer.append(rec)
			buf_len = next_len
		else:
			flush()
			buffer.append(rec)
			buf_len = len(rec.text)
	flush()
	return combined


def crawl(start_url: str, max_pages: int, delay: float, session: requests.Session, user_agent: str) -> Iterable[Tuple[str, bytes, Optional[str]]]:
	visited: Set[str] = set()
	queue: Deque[str] = collections.deque([start_url])
	robots = load_robots(start_url, user_agent)
	root_prefix = _normalize_root_prefix(start_url)

	while queue and len(visited) < max_pages:
		url = queue.popleft()
		if url in visited:
			continue
		if not has_url_prefix(url, root_prefix):
			continue
		if not is_allowed_by_robots(robots, url, user_agent):
			continue
		try:
			resp = session.get(url, timeout=20)
		except Exception:
			continue
		if resp.status_code >= 400 or "text/html" not in resp.headers.get("Content-Type", ""):
			continue

		# Capture bytes and robust encoding hint
		apparent = getattr(resp, "apparent_encoding", None)
		encoding_hint = choose_best_encoding(resp.content, resp.headers.get("Content-Type"), resp.encoding, apparent)
		html_bytes = resp.content
		visited.add(url)
		yield url, html_bytes, encoding_hint

		# discover links
		try:
			soup = BeautifulSoup(html_bytes, "html.parser", from_encoding=encoding_hint)
			for a in soup.find_all("a", href=True):
				norm = normalize_url(url, a["href"])
				if not norm:
					continue
				if not has_url_prefix(norm, root_prefix):
					continue
				if norm not in visited:
					queue.append(norm)
		except Exception:
			pass

		if delay > 0:
			time.sleep(delay)


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
	with open(path, "w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
	parser = argparse.ArgumentParser(description="Crawl a docs site and extract paragraphs for embeddings (JSONL).")
	parser.add_argument("--start-url", required=True, help="Root URL of the docs (e.g., https://docs.example.com/guide/)")
	parser.add_argument("--output", required=True, help="Output JSONL file for paragraph-level chunks")
	parser.add_argument("--combined-output", help="Optional JSONL file for combined chunks (by char limit)")
	parser.add_argument("--max-pages", type=int, default=500, help="Maximum pages to crawl")
	parser.add_argument("--delay", type=float, default=0.25, help="Delay (seconds) between requests")
	parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="Custom User-Agent")
	parser.add_argument("--combine-max-chars", type=int, default=0, help="Combine consecutive paragraphs within same heading up to this char size")
	args = parser.parse_args()

	session = build_http_session(args.user_agent)

	all_records: List[ParagraphRecord] = []
	for url, html_bytes, encoding_hint in crawl(args.start_url, args.max_pages, args.delay, session, args.user_agent):
		records = extract_paragraphs(url, html_bytes, encoding_hint)
		all_records.extend(records)

	# Paragraph-level JSONL
	def rec_to_row(r: ParagraphRecord) -> Dict:
		return {
			"url": r.url,
			"page_title": r.page_title,
			"heading_path": r.heading_path,
			"paragraph_index": r.paragraph_index,
			"text": r.text,
			"title_with_path": " > ".join([p for p in ([r.page_title] if r.page_title else []) + r.heading_path if p]),
		}

	write_jsonl(args.output, (rec_to_row(r) for r in all_records))

	# Combined JSONL (optional)
	if args.combined_output:
		combined = combine_by_char_limit(all_records, args.combine_max_chars)
		write_jsonl(args.combined_output, (rec_to_row(r) for r in combined))


if __name__ == "__main__":
	main()


