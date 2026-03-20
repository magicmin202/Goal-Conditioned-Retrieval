"""Date utility functions."""
from __future__ import annotations
from datetime import date, datetime


def today_str() -> str:
    return date.today().isoformat()


def parse_date(date_str: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def date_range_str(start: str, end: str) -> str:
    return f"{start} ~ {end}"
