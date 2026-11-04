"""Whether a source read online may be kept: a full copy saved into a library.

Reading and keeping are separate. A research agent may read any page its site lets AI agents
read (site_rules) and may cite what it read. Keeping a copy is a further step: the agent asks
for it when it judges the source reliable and useful, and this module decides whether the
source's rights allow it. The rule is fixed in code, and anything it cannot confirm is not
kept: a page is kept only when one of these holds.

- It came from a public legal database whose documents carry no copyright or are public
  records: court opinions (CourtListener), the Federal Register, the eCFR, govinfo, and
  filings on SEC EDGAR.
- Its site is on the list of US federal sites (settings.keep_sites), whose content is
  generally a work of the US government (17 U.S.C. § 105). State and local government sites
  are not listed: their works can be copyrighted.
- The page marks an open license or a public-domain dedication in a machine-readable way: a
  rel="license" link, or a JSON-LD "license", pointing to a Creative Commons license or
  public-domain tool.

Pure: no network, no database.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

# The legal databases' documents, by provider id (sources.PROVIDERS), and why each may be kept.
PUBLIC_PROVIDERS = {
    "courtlistener": "a court opinion (judicial opinions carry no copyright)",
    "federal_register": "a Federal Register document (a US government work)",
    "ecfr": "a regulation from the eCFR (a US government work)",
    "govinfo": "a US government publication from govinfo",
    "sec_edgar": "a public filing on SEC EDGAR",
}

# US federal sites whose content is generally a US government work. Admins can change the list
# with AILAWLAB_KEEP_SITES; a site also covers its subdomains.
DEFAULT_KEEP_SITES = [
    "supremecourt.gov", "uscourts.gov", "congress.gov", "govinfo.gov", "federalregister.gov",
    "ecfr.gov", "uscode.house.gov", "justice.gov",
]


@dataclass(frozen=True)
class KeepCheck:
    """The rights check's result: whether a copy may be kept, and why (shown to the agent and
    on the run page). `license_url` is the license a page marked, when that is the reason."""
    allowed: bool
    reason: str
    license_url: str = ""


NOT_CONFIRMED = ("its rights could not be confirmed (it is not from a public legal database "
                 "or a listed federal site, and it marks no open license)")


def check(provider: str, url: str, body: bytes | str = b"", content_type: str = "",
          sites: list[str] | None = None) -> KeepCheck:
    """May a copy of this source be kept? `provider` is the legal database it came from, or
    "web" for a page read from the open web; `body` and `content_type` are the page as it was
    read, for its license marking."""
    if provider in PUBLIC_PROVIDERS:
        return KeepCheck(True, PUBLIC_PROVIDERS[provider])
    site = listed_site(url, DEFAULT_KEEP_SITES if sites is None else sites)
    if site:
        return KeepCheck(True, f"a page on {site}, a listed US federal site")
    license_url = open_license(body, content_type)
    if license_url:
        return KeepCheck(True, f"the page marks the open license {license_name(license_url)}",
                         license_url)
    return KeepCheck(False, NOT_CONFIRMED)


def listed_site(url: str, sites: list[str]) -> str:
    """The listed site `url` is on (its host or a parent domain), or ""."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    for site in sites:
        site = site.strip().lower().lstrip(".")
        if site and (host == site or host.endswith("." + site)):
            return site
    return ""


_CC = r"https?://(?:www\.)?creativecommons\.org/(?:licenses|publicdomain)/[^\s\"'<>]+"
_TAG = re.compile(r"<(?:a|link)\b[^>]*>", re.IGNORECASE)
_REL_LICENSE = re.compile(r"""\brel\s*=\s*["']?[^"'>]*\blicense\b""", re.IGNORECASE)
_HREF = re.compile(r"""\bhref\s*=\s*["']?(""" + _CC + ")", re.IGNORECASE)
_JSON_LD = re.compile(r'"license"\s*:\s*"(' + _CC + ')"', re.IGNORECASE)


def open_license(body: bytes | str, content_type: str = "") -> str:
    """The Creative Commons license or public-domain tool an HTML page marks for itself, or
    "". Only machine-readable markings count (rel="license", JSON-LD "license"): a page that
    merely mentions a license in its text is not enough, and nothing else is guessed."""
    if not body:
        return ""
    if content_type and "html" not in content_type.lower():
        return ""
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
    for tag in _TAG.findall(text):
        if _REL_LICENSE.search(tag) and (m := _HREF.search(tag)):
            return m.group(1)
    if m := _JSON_LD.search(text):
        return m.group(1)
    return ""


def license_name(url: str) -> str:
    """"CC BY 4.0", "CC0 1.0" or "Public Domain Mark 1.0" for a Creative Commons address."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "licenses":
        version = parts[2] if len(parts) > 2 else ""
        return f"CC {parts[1].upper()}" + (f" {version}" if version else "")
    if len(parts) >= 2 and parts[0] == "publicdomain":
        version = parts[2] if len(parts) > 2 else ""
        name = {"zero": "CC0", "mark": "Public Domain Mark"}.get(parts[1], "a public-domain tool")
        return name + (f" {version}" if version else "")
    return url
