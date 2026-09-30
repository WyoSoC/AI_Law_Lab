"""Citation lists for libraries. Pure."""
from __future__ import annotations

from datetime import UTC, datetime

from ailawlab.citations import as_text, citation, citation_list


def test_each_source_gets_its_own_form():
    assert citation({"title": "§ 261.11 Criteria for listing hazardous waste.",
                     "source_uri": "https://www.ecfr.gov/x", "metadata": {
                         "provider": "ecfr", "citation": "40 CFR § 261.11", "as_of": "2026-07-21",
                         "retrieved_at": "2026-07-24T16:52:12+00:00"}})["text"] == (
        "40 CFR § 261.11 — Criteria for listing hazardous waste. "
        "eCFR, current as of July 21, 2026, retrieved July 24, 2026.")
    op = citation({"title": "In re X", "source_uri": "https://www.courtlistener.com/o/1/",
                   "metadata": {"provider": "courtlistener", "court": "Wyo.", "date_filed": "1994-11-18",
                                "opinion_id": "1", "snippet_only": True}})
    assert op["text"] == "In re X (Wyo., November 18, 1994). CourtListener, opinion no. 1, search snippet only."
    assert op["url"] == "https://www.courtlistener.com/o/1/"
    fr = citation({"title": "Water Rule", "metadata": {"provider": "federal_register", "document_number": "2026-1",
                                                      "type": "Rule", "publication_date": "2026-01-05"}})
    assert fr["text"] == "Water Rule. Federal Register document 2026-1 (Rule, published January 5, 2026)."
    web = citation({"title": "Guidance", "doc_type": "web page", "source_uri": "https://deq.wyo.gov/g",
                    "metadata": {"link": "https://deq.wyo.gov/g", "site": "Wyoming DEQ"}})
    assert web["text"] == "Guidance. Wyoming DEQ." and web["url"] == "https://deq.wyo.gov/g"
    up = citation({"title": "Lease", "source_uri": "lease.pdf", "metadata": {},
                   "created_at": datetime(2026, 9, 30, tzinfo=UTC)})
    assert up["text"] == "Lease. Uploaded file lease.pdf, retrieved September 30, 2026." and up["url"] == ""


def test_a_list_is_in_title_order_and_numbered_as_text():
    entries = citation_list([{"id": 2, "title": "b", "metadata": {}}, {"id": 1, "title": "A", "metadata": {}}])
    assert [e["id"] for e in entries] == [1, 2]
    assert as_text(entries, "Library x") == "Library x\n\n1. A.\n2. b.\n"


def test_version_notes_say_what_changed():
    from ailawlab.rag import describe_change

    before = {1: "Old statute", 2: "Guidance", 3: "Opinion"}
    after = {2: "Guidance", 4: "New rule", 5: "Guidance page"}
    # 1 was re-read into 5 (a changed web page), 3 was removed, 4 was added.
    note = describe_change(before, after, replaced={1: 5})
    assert note == "Added 1: “New rule”. Updated 1: “Guidance page”. Removed 1: “Opinion”."
    assert describe_change(before, before) == "No change."

    assert describe_change({1: "Lease"}, {}, moved={1: "Contracts"}) == "Moved 1 to “Contracts”: “Lease”."
