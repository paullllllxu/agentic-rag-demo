from __future__ import annotations

import ipaddress
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

from .transport import APIError

FALLBACK = "无法确认，建议咨询官方客服"
HUMAN = "此问题需转人工处理，请通过游戏内官方客服入口联系人工客服。"


def public_url(url: str, domains: tuple[str, ...]) -> bool:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if len(url) > 8192 or parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
            return False
        if any(ord(c) < 32 for c in url) or parsed.port not in {None, 80, 443}:
            return False
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")) or "." not in host:
            return False
        try:
            if not ipaddress.ip_address(host).is_global:
                return False
        except ValueError:
            pass
        return not domains or any(host == d or host.endswith("." + d) for d in domains)
    except (ValueError, TypeError):
        return False


class BodyText(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "form", "iframe", "noscript", "svg", "button"}
    BREAK = {"p", "div", "section", "article", "main", "br", "li", "h1", "h2", "h3", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip.append(tag)
        elif tag in self.BREAK and not self.skip:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.skip:
            if tag == self.skip[-1]:
                self.skip.pop()
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def clean_body(raw: str, limit: int) -> str:
    raw = re.sub(r"<!--[\s\S]*?-->", "", raw)
    # Fetch already returns Markdown. Strip executable HTML and navigation if present.
    if re.search(r"</?(?:html|body|div|article|script|style|nav|p|iframe|br|span|font|b|strong|ul|ol)\b", raw, re.I):
        parser = BodyText()
        parser.feed(raw)
        raw = "".join(parser.parts)
    raw = re.sub(r"!\[[^\]]*\]\([^\n]*?\)", "", raw)
    raw = re.sub(r"\[([^\]]+)\]\([^\n]*?\)", r"\1", raw)
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    if len(raw) > limit:
        raw = raw[:limit].rsplit("\n", 1)[0] if "\n" in raw[:limit] else raw[:limit]
    return raw.strip()


def normalize_quote(text: str) -> str:
    return re.sub(r"\s+", "", text)


def validate_claims(payload: dict, sources: list[dict], *, max_claims: int = 6) -> list[dict]:
    claims = payload.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= max_claims:
        raise APIError("answer: 缺少可验证结论")
    by_id = {s["id"]: s for s in sources}
    checked = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise APIError("answer: 结论结构非法")
        text, ids, quotes = claim.get("text"), claim.get("source_ids"), claim.get("quotes")
        if not isinstance(text, str) or not text.strip() or len(text) > 500 or re.search(r"[\r\n]|\[\d+\]|https?://|参考来源", text):
            raise APIError("answer: 结论格式非法")
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i not in by_id for i in ids) or len(set(ids)) != len(ids):
            raise APIError("answer: 引用了不存在的来源")
        if not isinstance(quotes, list) or not quotes:
            raise APIError("answer: 缺少原文证据")
        matched = set()
        for quote in quotes:
            if not isinstance(quote, dict) or type(quote.get("source_id")) is not int or quote["source_id"] not in ids:
                raise APIError("answer: 证据来源非法")
            excerpt = quote.get("quote")
            if not isinstance(excerpt, str) or len(normalize_quote(excerpt)) < 4 or normalize_quote(excerpt) not in normalize_quote(by_id[quote["source_id"]]["content"]):
                raise APIError("answer: 证据不是来源正文原文")
            matched.add(quote["source_id"])
        if matched != set(ids):
            raise APIError("answer: 每条引用都必须有原文证据")
        checked.append({"text": text.strip(), "source_ids": ids, "quotes": quotes})
    return checked


def safe_title(title: str) -> str:
    return re.sub(r"[\r\n\[\]<>]", " ", title).strip()[:180]


def render(claims: list[dict], sources: list[dict]) -> tuple[str, list[dict]]:
    used_ids = list(dict.fromkeys(i for c in claims for i in c["source_ids"]))
    numbers = {source_id: i + 1 for i, source_id in enumerate(used_ids)}
    by_id = {s["id"]: s for s in sources}
    lines = [c["text"].rstrip() + " " + "".join(f"[{numbers[i]}]" for i in c["source_ids"]) for c in claims]
    references = []
    for source_id in used_ids:
        source = by_id[source_id]
        title = safe_title(source["title"])
        label = f"《{title}》· 知识库" if source["kind"] == "kb" else f"{title} · {source['url']}"
        references.append(f"[{numbers[source_id]}] {label}")
    cited = [{**by_id[i], "citation_number": numbers[i]} for i in used_ids]
    return "\n".join(lines) + "\n\n参考来源：\n" + "\n".join(references), cited


def policy_answer(human: bool = False, *, message: str | None = None) -> str:
    text = message if message is not None else HUMAN if human else FALLBACK
    return text + " [1]\n\n参考来源：\n[1] 《客服转人工与兜底规则》· 本地策略"
