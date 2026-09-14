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

    # --- roleplay --------------------------------------------------------
    # A negotiation needs room to actually move: 12 turns is roughly three exchanges per
    # side, which tends to end with positions restated rather than shifted. The moderator
    # still ends a scene early on agreement or a clear impasse, so the default is a ceiling.
    default_max_turns: int = 100
    max_turns_limit: int = 100
    # Measured 2026-09-13: a ~1000-word gemma4 reply costs ~1.4k output tokens with thinking
    # on (~45 s on one Spark); with the moderator and private-notes calls a turn is ~70 s,
    # so a full 100-turn scene takes about two hours.
    default_word_limit: int = 1000
    word_limit_max: int = 2000

    # --- web ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8088

    # Mount point when served behind a reverse proxy at a subpath, e.g. "/ai_law_lab".
    # Empty means the app owns the root. Every link, form action, redirect and fetch()
    # in the UI is built from this, so it must match the proxy's ProxyPass path exactly.
    url_prefix: str = ""

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
