"""Minimal extraction pipeline for the Kanastra Data Service API.

The code focuses on reading data in an idempotent, functional manner.  Each
function accepts explicit parameters and returns plain Python iterables, making
the pipeline easy to test and reason about.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import pandas as pd
from typing import Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, TYPE_CHECKING
from zipfile import ZipFile

import requests

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import pandas as pd


@dataclass(frozen=True)
class APIConfig:
    """Holds immutable configuration required to talk to the API."""

    base_url: str
    client_id: str
    client_secret: str
    timeout: int = 30


def load_config_from_env() -> APIConfig:
    """Load configuration from environment variables."""

    base_url = os.environ["KANASTRA_BASE_URL"].rstrip("/")
    client_id = os.environ["KANASTRA_CLIENT_ID"]
    client_secret = os.environ["KANASTRA_CLIENT_SECRET"]
    return APIConfig(base_url=base_url, client_id=client_id, client_secret=client_secret)


def _auth_payload(config: APIConfig) -> Mapping[str, str]:
    return {"client_id": config.client_id, "client_secret": config.client_secret}


def _build_session(config: APIConfig) -> requests.Session:
    session = requests.Session()
    token = _fetch_access_token(session, config)
    session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept"] = "application/json"
    session.headers["Content-Type"] = "application/json"
    return session


def _fetch_access_token(session: requests.Session, config: APIConfig) -> str:
    url = f"{config.base_url}/v2/auth"
    response = session.post(url, headers={"Content-Type": "application/json"}, json=_auth_payload(config))
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token") or payload.get("token")
    print(f"Token: {token}")
    if not token:
        raise RuntimeError("Auth response did not contain an access token")
    return token


def _request_json(
    session: requests.Session,
    config: APIConfig,
    path: str,
    params: Optional[Mapping[str, object]] = None,
) -> Mapping[str, object]:
    url = f"{config.base_url}{path}"
    response = session.get(url, params=params, timeout=config.timeout)
    response.raise_for_status()
    return response.json()


def _stream_paginated(
    session: requests.Session,
    config: APIConfig,
    path: str,
    base_params: Optional[Mapping[str, object]] = None,
    page_size: int = 500,
    items_key: str = "items",
) -> Iterator[Mapping[str, object]]:
    """Yield items from endpoints that follow the standard paginated response."""

    params: MutableMapping[str, object] = dict(base_params or {})
    params.setdefault("size", page_size)

    page = 1
    while True:
        params["page"] = page
        payload = _request_json(session, config, path, params)

        if isinstance(payload, list):
            # Some endpoints, such as `/indexes`, return a plain list.
            items = payload
        else:
            items = payload.get(items_key) or []

        if not items:
            return

        for item in items:
            yield item

        # Idempotence: stop once we have consumed all items available for the
        # supplied query window. This avoids duplicate processing.
        if isinstance(payload, Mapping):
            total_pages = payload.get("pages") or payload.get("total_pages")
            if total_pages and page >= int(total_pages):
                return

        # Fallback termination when total page count is absent.
        if len(items) < params["size"]:
            return

        page += 1


def fetch_portfolios(
    config: APIConfig,
    *,
    start_reference_date: str,
    end_reference_date: str,
    slug: str,
    page_size: int = 500,
) -> Iterable[Mapping[str, object]]:
    with _build_session(config) as session:
        params = {
            "start_reference_date": start_reference_date,
            "end_reference_date": end_reference_date,
            "slug": slug,
        }
        return list(
            _stream_paginated(
                session,
                config,
                "/portfolio",
                base_params=params,
                page_size=page_size,
            )
        )


def fetch_acquisitions(
    config: APIConfig,
    *,
    start_reference_date: str,
    end_reference_date: str,
    slug: str,
    page_size: int = 500,
) -> Iterable[Mapping[str, object]]:
    with _build_session(config) as session:
        params = {
            "start_reference_date": start_reference_date,
            "end_reference_date": end_reference_date,
            "slug": slug,
        }
        return list(
            _stream_paginated(
                session,
                config,
                "/acquisitions",
                base_params=params,
                page_size=page_size,
            )
        )


def fetch_classes(
    config: APIConfig,
    *,
    start_reference_date: str,
    end_reference_date: str,
    slug: str,
    page_size: int = 500,
) -> Iterable[Mapping[str, object]]:
    with _build_session(config) as session:
        params = {
            "start_reference_date": start_reference_date,
            "end_reference_date": end_reference_date,
            "slug": slug,
        }
        return list(
            _stream_paginated(
                session,
                config,
                "/classes",
                base_params=params,
                page_size=page_size,
            )
        )


def fetch_liquidations(
    config: APIConfig,
    *,
    start_reference_date: str,
    end_reference_date: str,
    slug: str,
    page_size: int = 500,
) -> Iterable[Mapping[str, object]]:
    with _build_session(config) as session:
        params = {
            "start_reference_date": start_reference_date,
            "end_reference_date": end_reference_date,
            "slug": slug,
        }
        return list(
            _stream_paginated(
                session,
                config,
                "/liquidations",
                base_params=params,
                page_size=page_size,
            )
        )


def fetch_investor_positions(
    config: APIConfig,
    *,
    start_reference_date: str,
    end_reference_date: str,
    slug: str,
    page_size: int = 500,
) -> Iterable[Mapping[str, object]]:
    with _build_session(config) as session:
        params = {
            "start_reference_date": start_reference_date,
            "end_reference_date": end_reference_date,
            "slug": slug,
        }
        return list(
            _stream_paginated(
                session,
                config,
                "/investor-positions",
                base_params=params,
                page_size=page_size,
            )
        )


def fetch_indexes(config: APIConfig) -> Iterable[Mapping[str, object]]:
    with _build_session(config) as session:
        payload = _request_json(session, config, "/indexes")
        if isinstance(payload, list):
            return payload
        return payload.get("items", [])


def _download_holdings_archive(
    url: str,
    fmt: str,
    *,
    download_dir: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> "pd.DataFrame":
    import pandas as pd

    response = requests.get(url, timeout=timeout or 120)
    response.raise_for_status()

    content = response.content

    if download_dir is not None:
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split("?")[0].rstrip("/").split("/")[-1] or "holdings.zip"
        (download_dir / filename).write_bytes(content)

    with ZipFile(BytesIO(content)) as archive:
        frames = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            with archive.open(info) as file_obj:
                frames.append(_read_holdings_member(file_obj, info.filename, fmt))

    if not frames:
        raise RuntimeError("Holdings archive did not contain any data files")

    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _read_holdings_member(file_obj, filename: str, fmt: str) -> "pd.DataFrame":
    import pandas as pd

    name = filename.lower()
    if name.endswith(".csv") or fmt.upper() == "CSV":
        return pd.read_csv(file_obj)
    if name.endswith(".jsonl"):
        return pd.read_json(file_obj, lines=True)
    if name.endswith(".json") or fmt.upper() == "JSON":
        return pd.read_json(file_obj)
    if name.endswith(".parquet") or fmt.upper() == "PARQUET":
        return pd.read_parquet(file_obj)
    if name.endswith(".avro") or fmt.upper() == "AVRO":
        raise NotImplementedError("Parsing AVRO holdings is not implemented; request CSV or JSON instead")

    raise ValueError(f"Unsupported holdings file format for '{filename}'")


def fetch_holdings_v3(
    config: APIConfig,
    *,
    reference_date: str,
    slug: str,
    fmt: str = "CSV",
    poll_interval: int = 10,
    timeout: int = 600,
    download_dir: Optional[Path] = None,
    download_timeout: Optional[int] = None,
) -> "pd.DataFrame":
    """Trigger and download holdings export using the v3 job API."""

    fmt_normalized = fmt.upper()
    if fmt_normalized not in {"CSV", "JSON", "PARQUET", "AVRO"}:
        raise ValueError("fmt must be one of {'CSV', 'JSON', 'PARQUET', 'AVRO'}")

    params: MutableMapping[str, object] = {
        "reference_date": reference_date,
        "slug": slug,
        "format": fmt_normalized,
    }

    with _build_session(config) as session:
        trigger = session.post(f"{config.base_url}/v3/holdings", params=params, timeout=config.timeout)

        if trigger.status_code in (200,202, 409):
            start = time.monotonic()
            count = 0
            while count < 10:
                status_resp = session.get(f"{config.base_url}/v3/holdings", params=params, timeout=config.timeout)
                status_resp.raise_for_status()
                status_payload = status_resp.json() or {}

                status = status_payload.get("status").lower()
                if status == "done":
                    storage_path = status_payload.get("storage_path")
                    if not storage_path:
                        raise RuntimeError("Holdings job completed without providing a storage_path")
                    return _download_holdings_archive(
                        storage_path,
                        fmt_normalized,
                        download_dir=download_dir,
                        timeout=download_timeout,
                    )

                if status == "error":
                    raise RuntimeError(status_payload.get("message") or "Holdings job failed")
                
                if status == "empty":
                    return pd.DataFrame()

                if time.monotonic() - start > timeout:
                    raise TimeoutError("Holdings job did not complete within the allotted timeout")

                time.sleep(poll_interval)
        else:
            raise RuntimeError("Unexpected response status from holdings export trigger")

def extract_dataset(
    dataset: str,
    config: Optional[APIConfig] = None,
    **params: object,
) -> object:
    """Lookup table that maps dataset names to the appropriate extractor.

    The return type depends on the selected dataset. Traditional REST endpoints
    yield iterables of dictionaries, while asynchronous jobs (such as
    ``holdings_v3``) may return pandas ``DataFrame`` instances.
    """

    config = config or load_config_from_env()

    registry: Dict[str, Callable[..., object]] = {
        "portfolio": fetch_portfolios,
        "acquisitions": fetch_acquisitions,
        "classes": fetch_classes,
        "liquidations": fetch_liquidations,
        "investor_positions": fetch_investor_positions,
        "indexes": fetch_indexes,
        "holdings_v3": fetch_holdings_v3,
    }

    extractor = registry.get(dataset)
    if extractor is None:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return extractor(config, **params)


if __name__ == "__main__":
    # Example usage keeping the entry point idempotent and side-effect free.
    config = load_config_from_env()

    portfolios = extract_dataset(
        "portfolio",
        config=config,
        start_reference_date="2025-01-01",
        end_reference_date="2025-01-31",
        slug="fidc-kanastra",
    )

    print(f"Fetched {len(portfolios)} portfolio records")