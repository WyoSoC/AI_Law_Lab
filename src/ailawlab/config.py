"""Runtime configuration, loaded from environment / .env."""
from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AILAWLAB_", extra="ignore")

    # --- inference tier -------------------------------------------------
    # Tailscale hostnames. The 10.99.252.x lab addresses are NOT routable from
    # the orchestrator, so they must never appear here.
    spark_hosts: list[str] = Field(default=["test-spark3", "test-spark4"])
    ollama_port: int = 11434

    chat_model: str = "gemma4:latest"
    embed_model: str = "nomic-embed-text:latest"

    # Measured 2026-07-23: aggregate throughput on a Spark plateaus at ~16 concurrent
    # requests (~160 tok/s). Beyond that, latency grows linearly while throughput stays
    # flat -- the extra requests simply queue inside Ollama. Admitting more than this
    # per host buys nothing and destroys the latency signal in experiment results.
    slots_per_host: int = 16

    request_timeout_s: float = 300.0
    health_check_interval_s: float = 30.0

    # --- storage --------------------------------------------------------
    database_url: str = "postgresql://ailawlab:ailawlab@127.0.0.1:5433/ailawlab"

    # --- memory ---------------------------------------------------------
    short_term_token_budget: int = 6000
    compact_fraction: float = 0.5
    retrieval_top_k: int = 4
    retrieval_min_similarity: float = 0.55

    # --- rag ------------------------------------------------------------
    rag_top_k: int = 6
    rag_min_similarity: float = 0.35
    chunk_chars: int = 2400
    chunk_overlap: int = 300

    # --- online legal sources -------------------------------------------
    # Credentials are optional by design: two of the four providers need none, and the
    # other two degrade to a disabled panel rather than raising, so a fresh checkout
    # still gets a working Legal Sources page without anyone signing up for anything.
    courtlistener_token: str = ""   # free: courtlistener.com/profile/api. Without it
                                    # search works but full opinion text 401s, so hits
                                    # are ingested from their search snippet instead.
    govinfo_api_key: str = ""       # free: api.data.gov/signup
    # SEC blocks generic user agents outright; their fair-access policy wants a real
    # contact address. Sending a fake one gets the whole institution rate-limited.
    sec_user_agent: str = "AI Law Lab (University of Wyoming) gojian@uwyo.edu"

    source_timeout_s: float = 45.0
    source_search_limit: int = 20
    # Each ingested document occupies embedding slots on the cluster for as long as it
    # takes to chunk and embed it, so a single click is capped rather than unbounded.
    source_max_ingest: int = 10

    # --- source material for AI-drafted casts ----------------------------
    # The whole source goes into one drafting prompt. ~12k words is ~17k tokens: a long news
    # feature or an opinion's syllabus and majority, with room left for gemma4 to write the
    # cast. A 40k-word Supreme Court PDF is cut to its first 12k words, with a visible note.
    source_max_words: int = 12_000
    source_max_bytes: int = 8_000_000

    # --- web links kept in a corpus ---------------------------------------
    # A corpus link is chunked, not put in one prompt, so it keeps far more than a drafting
    # source: 100k words covers a long opinion or a statute chapter page. How many links one
    # request adds is capped, because each one occupies embedding slots on the cluster.
    web_link_max_words: int = 100_000
    web_link_max_per_request: int = 20

    # --- crawling a page's links into a library ---------------------------
    # "Add everything linked from this page": the rules in crawler.py. Pages are fetched one
    # at a time at least crawl_delay_s apart (longer if robots.txt asks, up to
    # crawl_max_delay_s); one crawl takes at most crawl_max_pages.
    crawl_delay_s: float = 2.0
    crawl_max_delay_s: float = 30.0
    crawl_max_pages: int = 100
    crawl_user_agent: str = ("AILawLab-crawler/1.0 (+https://datahive.uwyo.edu/ai_law_lab/; "
                             "University of Wyoming legal research; gojian@uwyo.edu)")

    # --- agentic workflow --------------------------------------------------
    # An agent decides for itself when it has enough to answer; this is only a safety net.
    # Its last step (or the step after its context fills) gets no tools, so it must answer.
    agent_max_steps: int = 400
    agent_context_tokens: int = 131072
    agent_wrap_up_share: float = 0.85    # past this share of the context, the next step answers

    # --- network tools for agents -----------------------------------------
    # With "Allow network tools" an agent may search the online databases and read a result
    # or a public web page. Each read is saved into a library and embedded, so it occupies
    # cluster slots like an ingest: a run may read at most this many documents.
    network_max_reads: int = 8          # the budget when an experiment sets no number of sources
    network_sources_default: int = 5    # sources an agent is asked to read, unless the experiment says
    network_sources_max: int = 20
    network_hits_per_source: int = 4

    # --- roleplay --------------------------------------------------------
    # A negotiation needs room to actually move: 12 turns is roughly three exchanges per
    # side, which tends to end with positions restated rather than shifted. The moderator
    # still ends a scene early on agreement or a clear impasse, so the default is a ceiling.
    default_max_turns: int = 100
    max_turns_limit: int = 100
    # Measured 2026-09-14 on a live run: a 1000-word turn (reply, private notes, moderator)
    # takes roughly 26-30 s, rising as prompts grow, so a full 100-turn scene takes about an
    # hour; at the default of 2000 words the pages estimate about 1 h 40 min.
    # graphs/roleplay_policy.ESTIMATE holds the figures the pages show.
    default_word_limit: int = 2000
    word_limit_max: int = 2000

    # --- web ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8088

    # Mount point when served behind a reverse proxy at a subpath, e.g. "/ai_law_lab".
    # Empty means the app owns the root. Every link, form action, redirect and fetch()
    # in the UI is built from this, so it must match the proxy's ProxyPass path exactly.
    url_prefix: str = ""

    # --- accounts (see docs/keycloak.md) ---------------------------------
    # Sign-in is required everywhere but the About page and static files. Turning it off
    # is for local development only: every request then acts as a built-in admin.
    auth_required: bool = True
    # Where browsers reach the app, used to build the sign-in callback address.
    public_url: str = "https://datahive.uwyo.edu/ai_law_lab"
    # Keycloak's realm as browsers see it (it is also the tokens' issuer), and as this
    # server reaches it: datahive cannot reach its own public address, so token exchange
    # and key lookups go over loopback.
    oidc_issuer: str = "https://datahive.uwyo.edu/sso/realms/ailawlab"
    oidc_internal_url: str = "http://127.0.0.1:8180/sso/realms/ailawlab"
    oidc_client_id: str = "ai-law-lab"
    oidc_client_secret: str = ""
    # Signs the session cookie. Generated into .env; changing it signs everyone out.
    session_secret: str = ""
    session_hours: int = 12
    # Verified addresses made admins on sign-in (comma-separated), and domains whose
    # verified addresses are approved as researchers without waiting for an admin.
    admin_emails: str = ""
    auto_approve_domains: str = "uwyo.edu"

    @field_validator("url_prefix")
    @classmethod
    def _normalise_prefix(cls, v: str) -> str:
        """Accept 'ai_law_lab', '/ai_law_lab' or '/ai_law_lab/' -- store '/ai_law_lab'.

        Concatenation sites all assume no trailing slash, so normalising here keeps a
        stray slash in .env from producing '//experiments' links.
        """
        v = v.strip().strip("/")
        return f"/{v}" if v else ""

    @property
    def embed_dim(self) -> int:
        return 768  # nomic-embed-text

    def base_urls(self) -> list[str]:
        return [f"http://{h}:{self.ollama_port}" for h in self.spark_hosts]


settings = Settings()
