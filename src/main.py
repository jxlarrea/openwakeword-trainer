"""ASGI entry point for `uvicorn src.main:app`."""
from __future__ import annotations

from src.logging_config import configure_logging
from src.settings import get_settings
from src.webui.app import create_app

settings = get_settings()
configure_logging(settings.log_level)

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
