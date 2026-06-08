#!/usr/bin/env python3
"""goal/log 임베딩 실행 및 캐시 저장.

Usage:
    .venv/bin/python scripts/run_embed.py           # mock 임베딩 (빠름)
    .venv/bin/python scripts/run_embed.py --real    # Gemini API 실제 임베딩
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.pipeline.embed import run_embed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Gemini API 실제 임베딩 사용")
    args = parser.parse_args()
    run_embed(real=args.real)
