"""Whether a site lets AI agents read a page: robots.txt, "noai", TDM reservations. Pure."""
from __future__ import annotations

import asyncio

import httpx

from ailawlab import network_tools, site_rules, source_material
from ailawlab.site_rules import RobotsTxt, page_refusal, parse_robots, tdm_reservation, token

ROBOTS = """
User-agent: *
Disallow: /private/
Allow: /private/open.html
Disallow: /*.pdf$
Crawl-delay: 5

User-agent: GPTBot
User-agent: ClaudeBot
Disallow: /articles/

User-agent: AILawLab-agent
Disallow: /no-lab/
"""


def test_the_agent_is_known_to_robots_txt_by_its_product_token():
    assert token("AILawLab-agent/1.0 (+https://datahive.uwyo.edu/ai_law_lab/)") == "ailawlab-agent"
    assert token("Mozilla/5.0") == "mozilla"


def test_wildcards_and_the_most_specific_rule_decide():
    r = RobotsTxt(ROBOTS)
    assert not r.allows("somebot", "https://e.org/private/x.html")
    assert r.allows("somebot", "https://e.org/private/open.html")        # longer Allow wins
    assert not r.allows("somebot", "https://e.org/lib/opinion.pdf")       # "*" and "$" understood
    assert r.allows("somebot", "https://e.org/lib/opinion.pdf?download=1")
    assert r.allows("somebot", "https://e.org/robots.txt")
    assert RobotsTxt("User-agent: *\nDisallow: /a\nAllow: /a\n").allows("x", "https://e.org/a")  # tie: Allow
    assert RobotsTxt("User-agent: *\nDisallow:\n").allows("x", "https://e.org/anything")
    # A group written for an agent replaces the "*" group for it.
    assert r.allows("ailawlab-agent", "https://e.org/private/x.html")
    assert not r.allows("ailawlab-agent", "https://e.org/no-lab/x")
    assert r.crawl_delay("somebot") == 5 and r.crawl_delay("gptbot") is None


def test_a_page_closed_to_ai_agents_is_closed_to_the_lab():
    robots = parse_robots(ROBOTS, "ailawlab-agent")
    assert robots.allows("https://e.org/about.html")
    refusal = robots.refusal("https://e.org/articles/story.html")
    assert "turns AI agents away" in refusal and "GPTBot" in refusal and "ClaudeBot" in refusal
    assert "does not allow this lab" in robots.refusal("https://e.org/no-lab/x")
    # Agents falling back to "*" are not counted as AI-specific refusals.
    assert parse_robots("User-agent: *\nDisallow: /x/\n", "ailawlab-agent").allows("https://e.org/y")


def test_tdm_reservations_in_tdmrep_json_apply_by_location():
    entries = [{"location": "/free/*", "tdm-reservation": 0},
               {"location": "/*", "tdm-reservation": 1, "tdm-policy": "https://e.org/policy.json"}]
    assert tdm_reservation(entries, "https://e.org/free/a.html") == ""
    reserved = tdm_reservation(entries, "https://e.org/news/a.html")
    assert "reserves its text-and-data-mining rights" in reserved and "https://e.org/policy.json" in reserved
    assert tdm_reservation("not a list", "https://e.org/a") == ""
    robots = parse_robots("", "ailawlab-agent")
    robots.tdm = entries
    assert not robots.allows("https://e.org/news/a.html")


def test_noai_and_tdm_reservations_on_the_page_itself():
    agent = "ailawlab-agent"
    html = b"<html><head><meta name='robots' content='noindex, noai'></head><body>x</body></html>"
    assert "noai" in page_refusal({"content-type": "text/html"}, html, agent)
    assert "noai" in page_refusal({"x-robots-tag": "noai"}, b"%PDF-1.7", agent)
    assert page_refusal({"x-robots-tag": "otherbot: noai"}, b"%PDF", agent) == ""   # for another crawler
    assert "noai" in page_refusal({"x-robots-tag": "ailawlab-agent: noai"}, b"%PDF", agent)
    assert "tdm-reservation" in page_refusal({"tdm-reservation": "1"}, b"%PDF", agent)
    meta = b'<html><head><meta name="tdm-reservation" content="1"></head></html>'
    assert "tdm-reservation" in page_refusal({"content-type": "text/html"}, meta, agent)
    plain = b"<html><head><meta name='robots' content='noindex'></head><body>x</body></html>"
    assert page_refusal({"content-type": "text/html"}, plain, agent) == ""
    # A tag in the body is the page's content, not its instructions.
    body_only = b"<html><head></head><body>&lt;meta name=robots content=noai&gt; <meta name='robots' content='noai'></body></html>"
    assert page_refusal({"content-type": "text/html"}, body_only, agent) == ""


def _site(routes: dict[str, tuple[int, str]]):
    def handle(req: httpx.Request) -> httpx.Response:
        status, body = routes.get(req.url.path, (404, ""))
        return httpx.Response(status, text=body)
    return httpx.MockTransport(handle)


def test_reading_a_site_gets_robots_and_tdmrep_and_a_failing_site_reads_nothing(monkeypatch):
    async def public(url):
        return None
    monkeypatch.setattr(source_material, "check_public_url", public)
    transport = _site({"/robots.txt": (200, ROBOTS),
                       "/.well-known/tdmrep.json": (200, '[{"location": "/reserved/*", "tdm-reservation": 1}]')})
    robots = asyncio.run(site_rules.read_robots("https://e.org/a", user_agent="AILawLab-agent/1.0",
                                                transport=transport))
    assert robots.allows("https://e.org/a") and not robots.allows("https://e.org/reserved/b")
    down = asyncio.run(site_rules.read_robots("https://e.org/a", transport=_site({"/robots.txt": (503, "")})))
    assert not down.allows("https://e.org/a")
    none = asyncio.run(site_rules.read_robots("https://e.org/a", transport=_site({})))
    assert none.status == "none" and none.allows("https://e.org/a")


def test_an_agent_is_told_a_page_was_not_read_and_the_run_keeps_it(monkeypatch):
    from ailawlab.grounding import SourceLedger
    from ailawlab.rag import Libraries

    class Notes:
        def __init__(self):
            self.messages = []

        async def note(self, message, **kw):
            self.messages.append(message)

    async def public(url):
        return None

    async def rules(url, user_agent):
        return parse_robots(ROBOTS, token(user_agent))

    async def never(*a, **k):
        raise AssertionError("a refused page must not be fetched")

    monkeypatch.setattr(source_material, "check_public_url", public)
    monkeypatch.setattr(site_rules, "for_page", rules)
    monkeypatch.setattr(source_material, "fetch_page", never)
    reader = network_tools.OnlineReader(None, Libraries(None, []), SourceLedger(), Notes(), "run-1",
                                        "Fetched: Test", web=True)
    out = asyncio.run(reader.read("https://e.org/articles/story.html"))
    assert out.startswith("ERROR: https://e.org/articles/story.html was not read")
    assert "Find another source" in out
    assert reader.refused == [{"url": "https://e.org/articles/story.html", "agent": None,
                               "reason": reader.refused[0]["reason"]}]
    assert "GPTBot" in reader.refused[0]["reason"]
    assert reader.tracer.messages and reader.tracer.messages[0].startswith("did not read https://e.org/articles/")

    # A page that arrives marked noai is not kept either.
    async def noai(url, **k):
        return url, httpx.Headers({"content-type": "text/html", "x-robots-tag": "noai"}), b"<html>x</html>"
    monkeypatch.setattr(source_material, "fetch_page", noai)
    out = asyncio.run(reader.read("https://e.org/about.html"))
    assert "was not read" in out and "noai" in out and len(reader.refused) == 2
