"""HuggingFace Hub pipeline registry client.

Search, install, and browse Tramontane pipelines published as
HF datasets tagged ``tramontane-pipeline``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import BaseModel
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

_CYAN = "#00D4EE"
_FROST = "#DCE9F5"
_STORM = "#4A6480"
_WARN = "#FFB020"
_RIM = "#1C2E42"
_console = Console()

_TAG = "tramontane-pipeline"


class HubPipeline(BaseModel):
    """Metadata for a pipeline published on HuggingFace Hub."""

    name: str
    author: str
    description: str
    tags: list[str] = []
    models_used: list[str] = []
    version: str = "0.1.2"
    downloads: int = 0
    likes: int = 0
    hf_url: str = ""


class HubClient:
    """Browse and install pipelines from HuggingFace Hub."""

    def __init__(self, hf_token: str | None = None) -> None:
        self._token = hf_token or os.environ.get("HF_TOKEN")

    def search(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[HubPipeline]:
        """Search HF datasets tagged ``tramontane-pipeline``."""
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self._token)
            search_tags = [_TAG, *(tags or [])]
            datasets = api.list_datasets(
                search=query,
                filter=search_tags,
                limit=limit,
                sort="likes",
            )
            results: list[HubPipeline] = []
            for ds in datasets:
                results.append(HubPipeline(
                    name=ds.id,
                    author=ds.author or "",
                    description=getattr(ds, "description", "") or "",
                    tags=list(ds.tags) if ds.tags else [],
                    downloads=ds.downloads or 0,
                    likes=ds.likes or 0,
                    hf_url=f"https://hf.co/datasets/{ds.id}",
                ))
            return results

        except ImportError:
            logger.warning("huggingface_hub not installed — hub search unavailable")
            return []
        except Exception:
            logger.warning("Hub search failed", exc_info=True)
            return []

    def install(
        self,
        name: str,
        target_dir: str = "./pipelines",
    ) -> str:
        """Download a pipeline YAML from HF Hub."""
        try:
            from huggingface_hub import hf_hub_download

            Path(target_dir).mkdir(parents=True, exist_ok=True)
            local_path = hf_hub_download(
                repo_id=name,
                filename="pipeline.yaml",
                repo_type="dataset",
                token=self._token,
                local_dir=target_dir,
            )
            _console.print(
                f"  [{_CYAN}]Installed[/] [{_FROST}]{name}[/] "
                f"[{_STORM}]-> {local_path}[/]"
            )
            return str(local_path)

        except ImportError:
            logger.warning("huggingface_hub not installed")
            return ""
        except Exception:
            logger.warning("Failed to install %s", name, exc_info=True)
            return ""

    def get_info(self, name: str) -> HubPipeline | None:
        """Fetch metadata for a specific pipeline."""
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self._token)
            ds = api.dataset_info(name)
            return HubPipeline(
                name=ds.id,
                author=ds.author or "",
                description=getattr(ds, "description", "") or "",
                tags=list(ds.tags) if ds.tags else [],
                downloads=ds.downloads or 0,
                likes=ds.likes or 0,
                hf_url=f"https://hf.co/datasets/{ds.id}",
            )
        except Exception:
            return None

    def display_search_results(self, results: list[HubPipeline]) -> None:
        """Display search results — EU Premium Rich table."""
        if not results:
            _console.print(Panel(
                f"[{_STORM}]No pipelines found \u00b7 "
                f"Be the first to publish one[/]",
                border_style=f"dim {_RIM}",
            ))
            return

        table = Table(
            title="Tramontane Hub",
            title_style=f"bold {_CYAN}",
            box=box.MINIMAL_HEAVY_HEAD,
            header_style=f"bold {_CYAN}",
            border_style=f"dim {_RIM}",
        )
        table.add_column("Name", style=f"bold {_CYAN}")
        table.add_column("Description", style=_STORM, max_width=40)
        table.add_column("Models", style=f"italic {_CYAN}")
        table.add_column("Downloads", justify="right", style=_FROST)
        table.add_column("Likes", justify="right", style=f"bold {_WARN}")

        for p in results:
            table.add_row(
                p.name,
                p.description[:40],
                ", ".join(p.models_used[:3]) or "-",
                str(p.downloads),
                str(p.likes),
            )
        _console.print(table)
