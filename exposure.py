from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPOSURE_FILE = Path(__file__).with_name("exposure_log.csv")
EXPOSURE_COLUMNS = [
    "item_key", "media_type", "title", "release_year",
    "shown_count", "last_shown_at", "selected_count", "last_selected_at",
]


def load_exposure() -> dict[str, dict]:
    """Load exposure log from CSV. Returns {item_key: row_dict}."""
    if not EXPOSURE_FILE.exists():
        return {}
    try:
        df = pd.read_csv(EXPOSURE_FILE, encoding="utf-8")
        result: dict[str, dict] = {}
        for _, row in df.iterrows():
            key = str(row.get("item_key", "") or "").strip()
            if not key:
                continue
            last_shown = str(row.get("last_shown_at", "") or "").strip() or None
            last_selected = str(row.get("last_selected_at", "") or "").strip() or None
            result[key] = {
                "item_key": key,
                "media_type": str(row.get("media_type", "") or ""),
                "title": str(row.get("title", "") or ""),
                "release_year": str(row.get("release_year", "") or ""),
                "shown_count": int(row.get("shown_count", 0) or 0),
                "last_shown_at": last_shown,
                "selected_count": int(row.get("selected_count", 0) or 0),
                "last_selected_at": last_selected,
            }
        return result
    except Exception as e:
        logger.warning("Failed to load exposure log: %s", e)
        return {}


def save_exposure(log: dict) -> bool:
    """Save exposure log to CSV. Returns True on success."""
    try:
        if not log:
            pd.DataFrame(columns=EXPOSURE_COLUMNS).to_csv(EXPOSURE_FILE, index=False)
            return True
        rows = list(log.values())
        df = pd.DataFrame(rows, columns=EXPOSURE_COLUMNS)
        df.to_csv(EXPOSURE_FILE, index=False)
        return True
    except (IOError, OSError) as e:
        logger.warning("Failed to save exposure log: %s", e)
        return False


def record_shown(
    log: dict,
    item_key: str,
    media_type: str,
    title: str,
    release_year: str,
) -> None:
    """Upsert: increment shown_count and update last_shown_at."""
    if not item_key:
        return
    now = datetime.now().isoformat()
    if item_key in log:
        log[item_key]["shown_count"] = log[item_key].get("shown_count", 0) + 1
        log[item_key]["last_shown_at"] = now
    else:
        log[item_key] = {
            "item_key": item_key,
            "media_type": media_type,
            "title": title,
            "release_year": release_year,
            "shown_count": 1,
            "last_shown_at": now,
            "selected_count": 0,
            "last_selected_at": "",
        }


def record_selected(log: dict, item_key: str) -> None:
    """Increment selected_count and update last_selected_at."""
    if not item_key:
        return
    now = datetime.now().isoformat()
    if item_key in log:
        log[item_key]["selected_count"] = log[item_key].get("selected_count", 0) + 1
        log[item_key]["last_selected_at"] = now
    else:
        log[item_key] = {
            "item_key": item_key,
            "media_type": "",
            "title": "",
            "release_year": "",
            "shown_count": 0,
            "last_shown_at": "",
            "selected_count": 1,
            "last_selected_at": now,
        }


def get_exposure(log: dict, item_key: str) -> dict:
    """Return row dict for item_key, or defaults if not found."""
    if not item_key or item_key not in log:
        return {
            "shown_count": 0,
            "last_shown_at": None,
            "selected_count": 0,
            "last_selected_at": None,
        }
    return log[item_key]


def load_seen_keys(exposure_log: dict) -> set[str]:
    """Derive seen canonical keys from items marked selected at least once."""
    return {k for k, v in exposure_log.items() if v.get("selected_count", 0) > 0}
