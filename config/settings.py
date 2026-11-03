
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str | None = Field(default=None)
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_chat_model: str = Field(default="gpt-4o")

    # LangSmith
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="clinical_trial_intelligence")
    langsmith_tracing_v2: bool = Field(default=False)

    # Google Cloud
    gcp_project_id: str = Field(...)
    gcp_region: str = Field(default="us-central1")
    gcs_bucket_name: str = Field(...)

    # Database
    db_host: str = Field(...)
    db_port: int = Field(default=5432)
    db_name: str = Field(default="clinical_trial_db")
    db_user: str = Field(...)
    db_password: str = Field(...)

    # ClinicalTrials.gov
    clinical_trials_base_url: str = Field(
        default="https://clinicaltrials.gov/api/v2"
    )
    clinical_trials_page_size: int = Field(default=100)

    # PubMed
    pubmed_base_url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    )

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_env: str = Field(default="development")

    @property
    def database_url(self) -> str:
        """
        Supports both:
        1. Local PostgreSQL connections using hostname/IP.
        2. Google Cloud SQL connections using Unix sockets.
        """

        if self.db_host.startswith("/"):
            return URL.create(
                drivername="postgresql+asyncpg",
                username=self.db_user,
                password=self.db_password,
                database=self.db_name,
                query={
                    "host": self.db_host,
                    "port": str(self.db_port),
                },
            ).render_as_string(hide_password=False)

        return URL.create(
            drivername="postgresql+asyncpg",
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        ).render_as_string(hide_password=False)

    @property
    def is_production(self) -> bool:
        return self.api_env.lower() == "production"


settings = Settings()
"""Global settings object."""