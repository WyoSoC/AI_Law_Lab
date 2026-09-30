"""Who gets in, and the checks in front of every request. Pure."""
from __future__ import annotations

import base64
import hashlib

from ailawlab.accounts import email_is_proven, initial_access
from ailawlab.web.auth import may, origin_allowed, pkce_pair, safe_next

ADMINS, DOMAINS = "gojian@uwyo.edu, other@uwyo.edu", "uwyo.edu"


def test_first_sign_in_decides_status_and_role():
    assert initial_access("GoJian@UWYO.edu", True, "uwyo", ADMINS, DOMAINS) == ("active", "admin")
    assert initial_access("student@uwyo.edu", True, "uwyo", ADMINS, DOMAINS) == ("active", "researcher")
    assert initial_access("student@uwyo.edu", True, "google", ADMINS, DOMAINS) == ("active", "researcher")
    assert initial_access("student@uwyo.edu", True, "", ADMINS, DOMAINS) == ("active", "researcher")
    # Unverified addresses, other domains, and Microsoft-reported addresses wait for an admin.
    assert initial_access("student@uwyo.edu", False, "uwyo", ADMINS, DOMAINS) == ("pending", "researcher")
    assert initial_access("lawyer@firm.com", True, "google", ADMINS, DOMAINS) == ("pending", "researcher")
    assert initial_access("student@uwyo.edu", True, "microsoft", ADMINS, DOMAINS) == ("pending", "researcher")
    assert initial_access("gojian@uwyo.edu", True, "microsoft", ADMINS, DOMAINS) == ("pending", "researcher")
    assert initial_access("x@evil-uwyo.edu", True, "google", ADMINS, DOMAINS) == ("pending", "researcher")
    assert not email_is_proven("", True, "google")


def test_after_sign_in_only_this_site_is_a_destination():
    assert safe_next("/ai_law_lab/runs/1?x=2", "/ai_law_lab") == "/ai_law_lab/runs/1?x=2"
    for bad in (None, "", "https://evil.example/", "//evil.example/x", "/\\evil", "/other_app/",
                "javascript:alert(1)"):
        assert safe_next(bad, "/ai_law_lab") == "/ai_law_lab/dashboard"


def test_state_changes_from_other_sites_are_refused():
    site = "https://datahive.uwyo.edu/ai_law_lab"
    assert origin_allowed("https://datahive.uwyo.edu", site)
    assert origin_allowed(None, site)
    for bad in ("https://evil.example", "http://datahive.uwyo.edu", "https://datahive.uwyo.edu.evil.example",
                "null", ""):
        assert not origin_allowed(bad, site)


def test_roles():
    viewer = {"status": "active", "role": "viewer"}
    researcher = {"status": "active", "role": "researcher"}
    admin = {"status": "active", "role": "admin"}
    assert may(viewer, "read") and not may(viewer, "write") and not may(viewer, "admin")
    assert may(researcher, "write") and not may(researcher, "admin")
    assert may(admin, "admin") and may(admin, "write")
    assert not may({"status": "pending", "role": "admin"}, "read") and not may(None, "read")


def test_pkce_challenge_is_the_s256_of_the_verifier():
    verifier, challenge = pkce_pair()
    assert 43 <= len(verifier) <= 128
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected
