"""CourtListener: which of a case's opinions is ingested, and what it is called. Offline:
the network is a mock transport."""
from __future__ import annotations

import httpx

from ailawlab import sources
from ailawlab.config import settings
from ailawlab.sources import CourtListener, opinion_rank, pick_lead

API = CourtListener.API
CLUSTER = f"{API}/clusters/111204/"

# Nix v. Williams as CourtListener serves it: four typed siblings plus an untyped combined
# opinion, with the dissent listed first.
SEARCH_OPINIONS = [
    {"id": 9429650, "type": "dissent", "ordering_key": 4, "snippet": "Justice Brennan, dissenting"},
    {"id": 9429649, "type": "concurrence-opinion", "ordering_key": 3, "snippet": "Justice Stevens"},
    {"id": 9429647, "type": "lead-opinion", "ordering_key": 1, "snippet": "Chief Justice Burger"},
    {"id": 111204, "type": "combined-opinion", "ordering_key": None, "snippet": "467 U.S. 431"},
]
OPINIONS = {
    9429650: {"id": 9429650, "type": "040dissent", "cluster": CLUSTER,
              "plain_text": "Justice Brennan, with whom Justice Marshall joins, dissenting.",
              "absolute_url": "/opinion/111204/nix-v-williams/"},
    9429647: {"id": 9429647, "type": "020lead", "cluster": CLUSTER,
              "plain_text": "Chief Justice Burger delivered the opinion of the Court.",
              "absolute_url": "/opinion/111204/nix-v-williams/"},
    9429648: {"id": 9429648, "type": "030concurrence", "cluster": CLUSTER,
              "plain_text": "Justice White, concurring."},
}


def test_rank_puts_the_court_first_and_dissents_last():
    assert opinion_rank("lead-opinion") == opinion_rank("020lead") == 0
    assert opinion_rank("015unamimous") < opinion_rank("030concurrence")   # sic
    assert opinion_rank("combined-opinion") < opinion_rank("concurrence-opinion")
    assert opinion_rank("040dissent") == opinion_rank("dissent") == max(
        opinion_rank(t) for t in ("lead-opinion", "combined-opinion", "concurrence-opinion",
                                  "dissent", None))


def test_pick_lead_skips_a_dissent_listed_first():
    assert pick_lead(SEARCH_OPINIONS)["id"] == 9429647
    assert pick_lead([]) is None
    assert pick_lead([{"type": "lead-opinion"}]) is None          # no id: unusable


def _install(monkeypatch, **kw) -> list[str]:
    """Point the provider at a mock CourtListener; returns the URLs it requests, in order."""
    calls: list[str] = []
    monkeypatch.setattr(settings, "courtlistener_token", "test-token")
    transport = httpx.MockTransport(_handler(calls, **kw))
    monkeypatch.setattr(sources, "_client",
                        lambda headers=None: httpx.AsyncClient(transport=transport))
    monkeypatch.setattr(sources.asyncio, "sleep", _no_sleep)
    return calls


def _handler(calls, cluster_status=200, rate_limit_first=False):
    seen = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        seen["n"] += 1
        if rate_limit_first and seen["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        if url.startswith(f"{API}/search/"):
            return httpx.Response(200, json={"results": [{
                "caseName": "Nix v. Williams", "absolute_url": "/opinion/111204/nix-v-williams/",
                "dateFiled": "1984-06-11", "court_citation_string": "SCOTUS",
                "opinions": SEARCH_OPINIONS}]})
        if url.startswith(CLUSTER):
            if cluster_status != 200:
                return httpx.Response(cluster_status)
            return httpx.Response(200, json={"case_name": "Nix v. Williams",
                                             "date_filed": "1984-06-11"})
        if request.url.path.endswith("/opinions/") and request.url.params.get("cluster"):
            assert request.url.params["cluster"] == "111204"
            return httpx.Response(200, json={"results": [
                {"id": 9429650, "type": "040dissent"}, {"id": 9429648, "type": "030concurrence"},
                {"id": 9429647, "type": "020lead"}]})
        oid = int(request.url.path.rstrip("/").rsplit("/", 1)[-1])
        return httpx.Response(200, json=OPINIONS[oid])
    return handle


async def test_search_offers_the_opinion_of_the_court(monkeypatch):
    _install(monkeypatch)
    hits = await CourtListener().search("Nix v. Williams", 5)
    assert [h.ref for h in hits] == ["9429647"]
    assert "Burger" in hits[0].snippet


async def test_fetching_a_dissent_ingests_the_lead_opinion(monkeypatch):
    _install(monkeypatch)
    doc = await CourtListener().fetch("9429650", {"title": "Nix v. Williams"})
    assert doc.text.startswith("Chief Justice Burger")
    assert doc.title == "Nix v. Williams"
    assert doc.metadata["opinion_id"] == "9429647"
    assert doc.metadata["requested_opinion_id"] == "9429650"
    assert doc.metadata["opinion_type"] == "020lead"


async def test_a_lead_opinion_is_fetched_without_looking_at_siblings(monkeypatch):
    calls = _install(monkeypatch)
    doc = await CourtListener().fetch("9429647", {})
    assert doc.metadata["opinion_id"] == "9429647"
    assert "requested_opinion_id" not in doc.metadata
    assert not any("cluster=" in u for u in calls)


async def test_unreadable_cluster_falls_back_to_the_search_title(monkeypatch):
    _install(monkeypatch, cluster_status=429)
    doc = await CourtListener().fetch("9429647", {"title": "Nix v. Williams", "date": "1984-06-11"})
    assert doc.title == "Nix v. Williams"
    assert doc.metadata["date_filed"] == "1984-06-11"


async def test_a_429_is_retried_once(monkeypatch):
    calls = _install(monkeypatch, rate_limit_first=True)
    doc = await CourtListener().fetch("9429647", {})
    assert doc.text.startswith("Chief Justice Burger")
    assert calls[0] == calls[1]


async def _no_sleep(_s):
    return None
