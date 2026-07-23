"""Runtime configuration, loaded from environment / .env."""
from __future__ import annotations

from pydantic import Field
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

    # --- web ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8088

    @property
    def embed_dim(self) -> int:
        return 768  # nomic-embed-text

    def base_urls(self) -> list[str]:
        return [f"http://{h}:{self.ollama_port}" for h in self.spark_hosts]


settings = Settings()
