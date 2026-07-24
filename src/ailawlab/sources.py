"""Online legal databases: search remote repositories and pull results into the corpus.

Every provider here is a *staging* step, not a live oracle. A search returns candidate
hits; nothing enters the corpus until someone selects it and it is fetched, chunked and
embedded like any uploaded PDF. That ordering is deliberate for a research setting: an
experiment must be reproducible against a fixed corpus snapshot, which is impossible if
the underlying authority can silently change between runs. Ingested documents record
their canonical URL and retrieval date in `metadata`, so a result can always be traced
back to what the source said on the day it was read.

Providers are grouped by topic and degrade independently. Two need no credential at all;
the two that do report themselves unavailable with a reason rather than raising, so a
missing key costs you one panel instead of the page.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import unescape
from typing import Any, ClassVar

import httpx

from .config import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- topics


@dataclass(frozen=True)
class Topic:
    id: str
    name: str
    blurb: str


TOPICS: tuple[Topic, ...] = (
    Topic("case_law", "Case law",
          "Judicial opinions — the holdings and reasoning that bind lower courts."),
    Topic("regulations", "Regulations & rulemaking",
          "What agencies have enacted, and what they are proposing to change."),
    Topic("statutes", "Statutes & legislation",
          "Enacted law and the bills moving through Congress."),
    Topic("securities", "Securities filings",
          "What public companies actually disclosed, in their own words."),
)


# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class SourceHit:
    """One candidate result. `ref` is whatever the provider needs to fetch full text."""
    provider: str
    ref: str
    title: str
    snippet: str = ""
    url: str = ""
    date: str = ""
    badge: str = ""          # court, agency, or form type — the "what kind" at a glance
    full_text: bool = True   # False when only the snippet can be ingested

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "ref": self.ref, "title": self.title,
            "snippet": self.snippet, "url": self.url, "date": self.date,
            "badge": self.badge, "full_text": self.full_text,
        }


@dataclass
class FetchedDoc:
    title: str
    text: str
    source_uri: str
    doc_type: str = "other"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- text helpers


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


def strip_markup(raw: str) -> str:
    """Flatten HTML/XML to text, preserving paragraph breaks.

    chunk_text() splits on blank lines, so block-level tags must become newlines rather
    than disappearing -- otherwise an entire filing collapses into one unsplittable
    paragraph and every chunk boundary lands mid-sentence.
    """
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<(br|/p|/div|/h[1-6]|/li|/tr|/head|/HEAD)\s*/?>", "\n\n", raw)
    text = unescape(_TAG.sub(" ", raw))
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages).strip()
    except Exception as e:                       # scanned images, malformed xref tables
        log.warning("pdf extraction failed: %s", e)
        return ""


def _client(headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    # follow_redirects because govinfo and SEC both hand out 302s to their CDN.
    return httpx.AsyncClient(
        timeout=settings.source_timeout_s,
        follow_redirects=True,
        headers={"User-Agent": settings.sec_user_agent, **(headers or {})},
    )


# ---------------------------------------------------------------- provider base


class Provider:
    """A remote legal repository that can be searched and selectively ingested."""

    id: str = ""
    name: str = ""
    topic: str = ""
    blurb: str = ""
    homepage: str = ""
    corpus: str = "default"
    query_hint: str = ""

    def status(self) -> tuple[bool, str]:
        """(usable, reason). Reason is shown to the user when not usable."""
        return True, ""

    async def search(self, query: str, limit: int) -> list[SourceHit]:
        raise NotImplementedError

    async def fetch(self, ref: str, hit: dict[str, Any] | None = None) -> FetchedDoc:
        """Retrieve full text for one hit.

        `hit` is the serialized SourceHit the user selected, passed back so a provider can
        reuse metadata the search already returned instead of paying a second round-trip
        for it -- and, where full text is gated behind a credential, fall back to the
        snippet rather than failing the ingest outright.
        """
        raise NotImplementedError

    def as_dict(self) -> dict[str, Any]:
        ok, reason = self.status()
        return {"id": self.id, "name": self.name, "topic": self.topic,
                "blurb": self.blurb, "homepage": self.homepage, "corpus": self.corpus,
                "query_hint": self.query_hint, "available": ok, "reason": reason}


# ---------------------------------------------------------------- CourtListener


class CourtListener(Provider):
    id = "courtlistener"
    name = "CourtListener"
    topic = "case_law"
    blurb = ("Free Law Project's opinion database — roughly 9 million opinions from every "
             "US federal court and all state appellate courts, back to the 1700s.")
    homepage = "https://www.courtlistener.com/"
    corpus = "case_law"
    query_hint = 'Try: qualified immunity, or "res ipsa loquitur" court_id:wyo'

    API = "https://www.courtlistener.com/api/rest/v4"

    def _headers(self) -> dict[str, str]:
        tok = settings.courtlistener_token.strip()
        return {"Authorization": f"Token {tok}"} if tok else {}

    def status(self) -> tuple[bool, str]:
        if not settings.courtlistener_token.strip():
            return True, ("No API token set — search works, but opinions are ingested from "
                          "their search snippet rather than full text. Add "
                          "AILAWLAB_COURTLISTENER_TOKEN to .env for complete opinions.")
        return True, ""

    async def search(self, query: str, limit: int) -> list[SourceHit]:
        has_token = bool(settings.courtlistener_token.strip())
        async with _client(self._headers()) as c:
            r = await c.get(f"{self.API}/search/",
                            params={"q": query, "type": "o", "page_size": limit})
            r.raise_for_status()
            data = r.json()

        hits: list[SourceHit] = []
        for res in data.get("results", [])[:limit]:
            ops = res.get("opinions") or [{}]
            # Cluster-level results carry several sibling opinions (majority, dissents);
            # the first is the one the search matched on.
            op_id = ops[0].get("id")
            if op_id is None:
                continue
            hits.append(SourceHit(
                provider=self.id,
                ref=str(op_id),
                title=res.get("caseName") or res.get("caseNameFull") or f"Opinion {op_id}",
                snippet=strip_markup(ops[0].get("snippet") or "")[:400],
                url=f"https://www.courtlistener.com{res.get('absolute_url', '')}",
                date=res.get("dateFiled") or "",
                badge=res.get("court_citation_string") or res.get("court") or "",
                full_text=has_token,
            ))
        return hits

    def _from_snippet(self, ref: str, hit: dict[str, Any]) -> FetchedDoc:
        """Ingest the search snippet when no token is configured.

        A snippet is a poor substitute for an opinion, so it is labelled as one in both
        doc_type and metadata: a retrieval that surfaces it should be visibly weaker
        evidence than one that surfaces a full opinion, not silently equivalent.
        """
        snippet = (hit.get("snippet") or "").strip()
        if not snippet:
            raise PermissionError(
                "CourtListener needs an API token for opinion text, and this hit has no "
                "snippet to fall back on. Set AILAWLAB_COURTLISTENER_TOKEN in .env.")
        title = hit.get("title") or f"CourtListener opinion {ref}"
        return FetchedDoc(
            title=title,
            text=f"{title}\n{hit.get('badge', '')} {hit.get('date', '')}\n\n{snippet}",
            source_uri=hit.get("url") or f"{self.API}/opinions/{ref}/",
            doc_type="opinion_snippet",
            metadata={"provider": self.id, "opinion_id": ref, "snippet_only": True,
                      "court": hit.get("badge", ""), "date_filed": hit.get("date", ""),
                      "retrieved_at": datetime.now(UTC).isoformat()},
        )

    async def fetch(self, ref: str, hit: dict[str, Any] | None = None) -> FetchedDoc:
        if not settings.courtlistener_token.strip():
            return self._from_snippet(ref, hit or {})

        async with _client(self._headers()) as c:
            r = await c.get(f"{self.API}/opinions/{ref}/")
            if r.status_code in (401, 403):
                # A token that is present but rejected is a configuration error worth
                # surfacing, not something to paper over with a snippet.
                raise PermissionError(
                    f"CourtListener rejected the configured API token (HTTP "
                    f"{r.status_code}). Check AILAWLAB_COURTLISTENER_TOKEN in .env.")
            r.raise_for_status()
            op = r.json()

            # Preference order runs cleanest-text first: the plain_text column is OCR'd
            # or publisher-supplied, while the HTML variants carry markup we would strip
            # anyway. Falling through all of them means the opinion simply has no text.
            text = ""
            for key in ("plain_text", "html_with_citations", "html", "html_lawbox",
                        "html_columbia", "xml_harvard"):
                if op.get(key):
                    text = op[key] if key == "plain_text" else strip_markup(op[key])
                    break

            cluster_name, filed = "", ""
            if op.get("cluster"):
                cr = await c.get(op["cluster"])
                if cr.status_code == 200:
                    cl = cr.json()
                    cluster_name = cl.get("case_name") or ""
                    filed = cl.get("date_filed") or ""

        title = cluster_name or f"CourtListener opinion {ref}"
        return FetchedDoc(
            title=title,
            text=text,
            source_uri=op.get("absolute_url") and
            f"https://www.courtlistener.com{op['absolute_url']}" or f"{self.API}/opinions/{ref}/",
            doc_type="opinion",
            metadata={"provider": self.id, "opinion_id": ref, "date_filed": filed,
                      "retrieved_at": datetime.now(UTC).isoformat()},
        )


# ---------------------------------------------------------------- Federal Register


class FederalRegister(Provider):
    id = "federal_register"
    name = "Federal Register"
    topic = "regulations"
    blurb = ("The daily journal of the US government — proposed rules, final rules, "
             "notices and presidential documents, published since 1994.")
    homepage = "https://www.federalregister.gov/"
    corpus = "regulations"
    query_hint = "Try: data breach notification, or PFAS drinking water"

    API = "https://www.federalregister.gov/api/v1"
    FIELDS = ("title", "document_number", "publication_date", "raw_text_url",
              "html_url", "abstract", "type", "agencies")

    async def search(self, query: str, limit: int) -> list[SourceHit]:
        params: list[tuple[str, str]] = [
            ("conditions[term]", query), ("per_page", str(limit)),
            ("order", "relevance"),
        ]
        params += [("fields[]", f) for f in self.FIELDS]
        async with _client() as c:
            r = await c.get(f"{self.API}/documents.json", params=params)
            r.raise_for_status()
            data = r.json()

        out = []
        for d in data.get("results", [])[:limit]:
            agencies = ", ".join(a.get("name", "") for a in (d.get("agencies") or [])[:2])
            out.append(SourceHit(
                provider=self.id,
                ref=d["document_number"],
                title=d.get("title") or d["document_number"],
                snippet=(d.get("abstract") or "")[:400],
                url=d.get("html_url", ""),
                date=d.get("publication_date", ""),
                badge=d.get("type") or agencies,
            ))
        return out

    async def fetch(self, ref: str, hit: dict[str, Any] | None = None) -> FetchedDoc:
        async with _client() as c:
            r = await c.get(f"{self.API}/documents/{ref}.json",
                            params=[("fields[]", f) for f in self.FIELDS])
            r.raise_for_status()
            doc = r.json()

            text = ""
            if doc.get("raw_text_url"):
                tr = await c.get(doc["raw_text_url"])
                if tr.status_code == 200:
                    # "raw text" is served wrapped in <html><body><pre>. The <pre> body is
                    # already plain text whose line breaks carry the column layout, so it
                    # is extracted directly rather than run through the markup stripper.
                    m = re.search(r"(?is)<pre>(.*?)</pre>", tr.text)
                    text = unescape(m.group(1)) if m else strip_markup(tr.text)
            # Documents older than the full-text archive expose only an abstract; that is
            # still worth indexing, just flagged so nobody mistakes it for the whole rule.
            if not text.strip():
                text = doc.get("abstract") or ""

        return FetchedDoc(
            title=doc.get("title") or ref,
            text=text,
            source_uri=doc.get("html_url") or f"{self.API}/documents/{ref}.json",
            doc_type="regulation",
            metadata={"provider": self.id, "document_number": ref,
                      "publication_date": doc.get("publication_date", ""),
                      "type": doc.get("type", ""),
                      "abstract_only": not doc.get("raw_text_url"),
                      "retrieved_at": datetime.now(UTC).isoformat()},
        )


# ---------------------------------------------------------------- eCFR


class ECFR(Provider):
    id = "ecfr"
    name = "e-CFR"
    topic = "regulations"
    blurb = ("The Code of Federal Regulations as currently in force — the consolidated "
             "text agencies actually enforce, updated daily.")
    homepage = "https://www.ecfr.gov/"
    corpus = "regulations"
    query_hint = "Try: hazardous waste manifest, or recordkeeping requirements"

    API = "https://www.ecfr.gov/api"

    async def search(self, query: str, limit: int) -> list[SourceHit]:
        async with _client() as c:
            r = await c.get(f"{self.API}/search/v1/results",
                            params={"query": query, "per_page": limit})
            r.raise_for_status()
            data = r.json()

        out = []
        for res in data.get("results", [])[:limit]:
            h = res.get("hierarchy") or {}
            hh = res.get("hierarchy_headings") or {}
            # The versioner needs every level that is present, so the ref carries the
            # whole hierarchy rather than just the section number.
            ref = json.dumps({k: v for k, v in h.items() if v})
            cite = f"{h.get('title', '?')} CFR " + (
                f"§ {h['section']}" if h.get("section") else f"Part {h.get('part', '?')}")
            # Search headings arrive with <strong> around the matched terms.
            heading = strip_markup((res.get("headings") or {}).get("section")
                                   or hh.get("section") or "")
            out.append(SourceHit(
                provider=self.id,
                ref=ref,
                title=f"{cite} {heading}".strip(),
                snippet=strip_markup(res.get("full_text_excerpt") or "")[:400],
                url=self._web_url(h),
                date=res.get("starts_on", ""),
                badge=(hh.get("title") or "").strip(),
            ))
        return out

    @staticmethod
    def _web_url(h: dict) -> str:
        if h.get("section"):
            return f"https://www.ecfr.gov/current/title-{h['title']}/section-{h['section']}"
        parts = "/".join(f"{k}-{v}" for k, v in h.items() if v and k != "title")
        return f"https://www.ecfr.gov/current/title-{h.get('title', '')}/{parts}"

    # Each title is republished on its own schedule, so there is no single "current"
    # date that works across all fifty. Cached because ingesting a dozen sections
    # otherwise re-fetches the same 50 KB index a dozen times.
    _issue_dates: ClassVar[dict[str, str]] = {}

    async def _latest_issue(self, c: httpx.AsyncClient, title_no: str) -> str:
        if not self._issue_dates:
            r = await c.get(f"{self.API}/versioner/v1/titles.json")
            r.raise_for_status()
            for t in r.json().get("titles", []):
                if t.get("latest_issue_date"):
                    ECFR._issue_dates[str(t["number"])] = t["latest_issue_date"]
        date = self._issue_dates.get(str(title_no))
        if not date:
            raise LookupError(f"eCFR does not publish a title {title_no}")
        return date

    async def fetch(self, ref: str, hit: dict[str, Any] | None = None) -> FetchedDoc:
        h = json.loads(ref)
        title_no = h.pop("title", None)
        if not title_no:
            raise ValueError("eCFR reference is missing a title number")
        async with _client() as c:
            date = await self._latest_issue(c, title_no)
            r = await c.get(f"{self.API}/versioner/v1/full/{date}/title-{title_no}.xml",
                            params=h)
            if r.status_code == 404:
                raise LookupError(
                    f"eCFR has no {' '.join(f'{k} {v}' for k, v in h.items())} "
                    f"in title {title_no} as of {date}")
            r.raise_for_status()
            xml = r.text

        heading = ""
        m = re.search(r"<HEAD>(.*?)</HEAD>", xml, re.DOTALL)
        if m:
            heading = strip_markup(m.group(1))
        cite = f"{title_no} CFR " + (f"§ {h['section']}" if h.get("section")
                                     else f"Part {h.get('part', '')}")
        return FetchedDoc(
            title=heading or cite,
            text=strip_markup(xml),
            source_uri=self._web_url({"title": title_no, **h}),
            doc_type="regulation",
            metadata={"provider": self.id, "citation": cite, "hierarchy": h,
                      "as_of": date, "retrieved_at": datetime.now(UTC).isoformat()},
        )


# ---------------------------------------------------------------- govinfo


class GovInfo(Provider):
    id = "govinfo"
    name = "govinfo (GPO)"
    topic = "statutes"
    blurb = ("The Government Publishing Office's authenticated repository — the US Code, "
             "Public Laws, Congressional bills, and committee reports.")
    homepage = "https://www.govinfo.gov/"
    corpus = "statutes"
    query_hint = 'Try: collection:USCODE artificial intelligence, or collection:BILLS privacy'

    API = "https://api.govinfo.gov"

    def status(self) -> tuple[bool, str]:
        if not settings.govinfo_api_key.strip():
            return False, ("Needs a free API key from api.data.gov/signup. "
                           "Set AILAWLAB_GOVINFO_API_KEY in .env to enable this source.")
        return True, ""

    async def search(self, query: str, limit: int) -> list[SourceHit]:
        ok, reason = self.status()
        if not ok:
            raise PermissionError(reason)
        async with _client({"X-Api-Key": settings.govinfo_api_key.strip()}) as c:
            r = await c.post(f"{self.API}/search", json={
                "query": query, "pageSize": min(limit, 100), "offsetMark": "*",
                "sorts": [{"field": "score", "sortOrder": "DESC"}],
            })
            r.raise_for_status()
            data = r.json()

        out = []
        for res in data.get("results", [])[:limit]:
            pkg = res.get("packageId") or ""
            gran = res.get("granuleId") or ""
            out.append(SourceHit(
                provider=self.id,
                ref=f"{pkg}/{gran}" if gran else pkg,
                title=res.get("title") or pkg,
                snippet=strip_markup(res.get("summary") or "")[:400],
                url=f"https://www.govinfo.gov/app/details/{pkg}"
                    + (f"/{gran}" if gran else ""),
                date=(res.get("dateIssued") or "")[:10],
                badge=res.get("collectionCode") or "",
            ))
        return out

    async def fetch(self, ref: str, hit: dict[str, Any] | None = None) -> FetchedDoc:
        ok, reason = self.status()
        if not ok:
            raise PermissionError(reason)
        pkg, _, gran = ref.partition("/")
        base = (f"{self.API}/packages/{pkg}/granules/{gran}" if gran
                else f"{self.API}/packages/{pkg}")
        async with _client({"X-Api-Key": settings.govinfo_api_key.strip()}) as c:
            s = await c.get(f"{base}/summary")
            s.raise_for_status()
            summary = s.json()

            text = ""
            # htm is the only format guaranteed across collections; pdf is the fallback
            # for older scanned material where GPO never produced a text rendition.
            for fmt, conv in (("htm", strip_markup), ("txt", lambda x: x)):
                fr = await c.get(f"{base}/{fmt}")
                if fr.status_code == 200 and fr.text.strip():
                    text = conv(fr.text)
                    break
            if not text:
                pr = await c.get(f"{base}/pdf")
                if pr.status_code == 200:
                    text = pdf_to_text(pr.content)

        return FetchedDoc(
            title=summary.get("title") or ref,
            text=text,
            source_uri=f"https://www.govinfo.gov/app/details/{pkg}"
                       + (f"/{gran}" if gran else ""),
            doc_type="statute",
            metadata={"provider": self.id, "package_id": pkg, "granule_id": gran,
                      "collection": summary.get("collectionCode", ""),
                      "date_issued": summary.get("dateIssued", ""),
                      "retrieved_at": datetime.now(UTC).isoformat()},
        )


# ---------------------------------------------------------------- SEC EDGAR


class SecEdgar(Provider):
    id = "sec_edgar"
    name = "SEC EDGAR"
    topic = "securities"
    blurb = ("Full-text search across every public company filing since 2001 — 10-K, "
             "8-K, S-1, proxy statements and their exhibits.")
    homepage = "https://www.sec.gov/edgar/search/"
    corpus = "securities"
    query_hint = 'Try: "material weakness", or "climate risk" (quote exact phrases)'

    SEARCH = "https://efts.sec.gov/LATEST/search-index"
    ARCHIVE = "https://www.sec.gov/Archives/edgar/data"

    async def search(self, query: str, limit: int) -> list[SourceHit]:
        async with _client() as c:
            r = await c.get(self.SEARCH, params={"q": query})
            r.raise_for_status()
            data = r.json()

        out = []
        for h in data.get("hits", {}).get("hits", [])[:limit]:
            src = h.get("_source", {})
            names = src.get("display_names") or []
            company = names[0].split("  (")[0] if names else "Unknown filer"
            form = src.get("form") or src.get("file_type") or ""
            out.append(SourceHit(
                provider=self.id,
                ref=h["_id"],                       # "accession:filename"
                title=f"{company} — {form}",
                snippet=src.get("file_description") or "",
                url=self._doc_url(h["_id"], (src.get("ciks") or ["0"])[0]),
                date=src.get("file_date", ""),
                badge=form,
            ))
        return out

    def _doc_url(self, ref: str, cik: str) -> str:
        acc, _, fname = ref.partition(":")
        return f"{self.ARCHIVE}/{cik.lstrip('0')}/{acc.replace('-', '')}/{fname}"

    async def fetch(self, ref: str, hit: dict[str, Any] | None = None) -> FetchedDoc:
        acc, _, fname = ref.partition(":")
        async with _client() as c:
            # The search index does not return the CIK alongside a bare ref, so recover
            # it by asking the index about this exact accession number.
            r = await c.get(self.SEARCH, params={"q": f'"{acc}"'})
            cik, company, form, filed = "0", "", "", ""
            if r.status_code == 200:
                for h in r.json().get("hits", {}).get("hits", []):
                    if h.get("_id") == ref:
                        s = h["_source"]
                        cik = (s.get("ciks") or ["0"])[0]
                        names = s.get("display_names") or []
                        company = names[0].split("  (")[0] if names else ""
                        form, filed = s.get("form", ""), s.get("file_date", "")
                        break

            url = self._doc_url(ref, cik)
            dr = await c.get(url)
            dr.raise_for_status()
            text = (pdf_to_text(dr.content) if fname.lower().endswith(".pdf")
                    else strip_markup(dr.text))

        return FetchedDoc(
            title=f"{company or 'SEC filing'} {form} ({filed})".strip(),
            text=text,
            source_uri=url,
            doc_type="filing",
            metadata={"provider": self.id, "accession": acc, "cik": cik, "form": form,
                      "filed": filed, "retrieved_at": datetime.now(UTC).isoformat()},
        )


# ---------------------------------------------------------------- registry


_PROVIDERS: tuple[Provider, ...] = (
    CourtListener(), FederalRegister(), ECFR(), GovInfo(), SecEdgar(),
)
PROVIDERS: dict[str, Provider] = {p.id: p for p in _PROVIDERS}


def get_provider(pid: str) -> Provider:
    p = PROVIDERS.get(pid)
    if p is None:
        raise KeyError(f"no such source provider: {pid!r}")
    return p


def by_topic() -> list[dict[str, Any]]:
    """Providers grouped into the topic sections the Legal Sources page renders."""
    return [
        {"id": t.id, "name": t.name, "blurb": t.blurb,
         "providers": [p.as_dict() for p in _PROVIDERS if p.topic == t.id]}
        for t in TOPICS
    ]


async def search(pid: str, query: str, limit: int | None = None) -> list[SourceHit]:
    provider = get_provider(pid)
    limit = min(limit or settings.source_search_limit, 50)
    try:
        return await provider.search(query, limit)
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"{provider.name} returned HTTP {e.response.status_code}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"{provider.name} is unreachable: {e}") from e


async def ingest(pid: str, hits: list[dict[str, Any]], corpus_name: str,
                 router) -> dict[str, Any]:
    """Fetch the selected hits and add them to `corpus_name`. Returns a per-hit report.

    Takes whole hits rather than bare refs so providers can reuse search metadata (see
    Provider.fetch). Documents are fetched concurrently but embedded one at a time by
    Corpus.add_document, which is the right split: fetching is IO-bound against four
    different hosts, while embedding contends for one finite pool of cluster slots.
    """
    from .rag import Corpus

    provider = get_provider(pid)
    hits = hits[:settings.source_max_ingest]
    store = Corpus(router, name=corpus_name or provider.corpus)

    async def one(hit: dict[str, Any]) -> dict[str, Any]:
        ref = hit.get("ref", "")
        try:
            return {"ref": ref, "doc": await provider.fetch(ref, hit)}
        except Exception as e:
            log.warning("fetch %s/%s failed: %s", pid, ref, e)
            return {"ref": ref, "error": f"{type(e).__name__}: {e}"}

    fetched = await asyncio.gather(*(one(h) for h in hits))

    report: list[dict[str, Any]] = []
    for item in fetched:
        if "error" in item:
            report.append({"ref": item["ref"], "status": "error", "detail": item["error"]})
            continue
        doc: FetchedDoc = item["doc"]
        if not doc.text.strip():
            report.append({"ref": item["ref"], "status": "empty",
                           "detail": "source returned no extractable text",
                           "title": doc.title})
            continue
        try:
            doc_id = await store.add_document(
                title=doc.title, text=doc.text, source_uri=doc.source_uri,
                doc_type=doc.doc_type, metadata=doc.metadata)
        except Exception as e:
            log.exception("ingest %s/%s failed", pid, item["ref"])
            report.append({"ref": item["ref"], "status": "error",
                           "detail": f"{type(e).__name__}: {e}", "title": doc.title})
            continue
        report.append({
            "ref": item["ref"], "title": doc.title,
            "status": "ingested" if doc_id else "duplicate",
            "detail": "" if doc_id else "already in the corpus (same sha256)",
        })

    return {"corpus": store.name, "results": report,
            "ingested": sum(1 for r in report if r["status"] == "ingested")}
