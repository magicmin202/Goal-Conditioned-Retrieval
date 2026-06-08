#!/usr/bin/env python3
"""합성 데이터셋 v2 빌드.

설계 원칙:
  1. 100명 × avg 4 goals ≈ 400 goals
     각 goal에 noise_ratio ∈ {0.0, 0.1, ..., 1.0} 균등 배분 (A안)
       - noise_ratio=0.0 → 100% goal 관련 로그 (engaged)
       - noise_ratio=0.5 → 50% goal 로그 / 50% noise 로그
       - noise_ratio=1.0 → 전체 noise 로그 (disengaged)
  2. 도메인 30개 × 목표 13개 = 390 목표 (쿼터 기반 균등 배분)
  3. 페르소나: nvidia/Nemotron-Personas-Korea 기반
  4. Label: relevant / irrelevant 이진만 사용

References:
  - Chan et al. (2024). Scaling Synthetic Data Creation with 1,000,000,000 Personas.
  - nvidia/Nemotron-Personas-Korea (HuggingFace)
  - Simulating Authentic User Personas to Generate Analytics for
    Evaluating Digital Products.

Usage:
    cd /home/user/retrieval
    .venv/bin/python scripts/build_dataset_v2.py
    .venv/bin/python scripts/build_dataset_v2.py --output data/synthetic_v2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

from app.data_generation.v2.goal_generator_v2 import generate_goals_v2
from app.data_generation.v2.label_generator_v2 import generate_labels_v2
from app.data_generation.v2.log_generator_v2 import generate_logs_v2
from app.data_generation.v2.persona_loader import load_personas
from app.schemas import ResearchUser

START_DATE = date(2026, 3, 1)
END_DATE   = date(2026, 3, 31)
SEED       = 42
N_USERS    = 100

# 30개 도메인 × 13 = 390 목표 (쿼터 기반 균등 배분)
ALL_DOMAINS = [
    "romance", "marriage", "family", "appearance", "health",
    "social", "friendship", "receiving", "boundaries", "prosocial",
    "teaching", "leadership", "religion", "awareness", "ethics",
    "freedom", "aesthetics", "creativity", "openness", "entertainment",
    "wellbeing", "stability", "meaning", "growth", "order",
    "achievement", "autonomy", "career", "education", "finance",
]
TARGET_DOMAIN_COUNTS: dict[str, int] = {d: 13 for d in ALL_DOMAINS}

# noise_ratio 쿼터 (A안): 0.0~1.0, 11 레벨, 각 ~37 goals
NOISE_LEVELS: list[float] = [round(i / 10, 1) for i in range(11)]
NOISE_TARGET = 37  # ceil(400 / 11) ≈ 37
NOISE_TARGETS: dict[float, int] = {n: NOISE_TARGET for n in NOISE_LEVELS}


def build_v2(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("페르소나 로드 중 (nvidia/Nemotron-Personas-Korea)...")
    personas = load_personas(n=N_USERS, seed=SEED)
    log.info("  %d개 페르소나 로드 완료", len(personas))

    all_users, all_goals, all_logs, all_labels = [], [], [], []
    domain_counter: dict[str, int] = {d: 0 for d in ALL_DOMAINS}
    noise_counter: dict[float, int] = {n: 0 for n in NOISE_LEVELS}

    total_pairs, total_rels = 0, 0

    for i, persona in enumerate(personas):
        user_idx = i + 1
        user_id  = f"U{user_idx:04d}"

        user = ResearchUser(
            user_id=user_id,
            profile={
                "name":             persona.name,
                "age":              persona.age,
                "sex":              persona.sex,
                "occupation":       persona.occupation,
                "district":         persona.district,
                "province":         persona.province,
                "education_level":  persona.education_level,
                "marital_status":   persona.marital_status,
                "family_type":      persona.family_type,
                "hobbies":          persona.hobbies,
                "skills":           persona.skills,
                "career_goals":     persona.career_goals,
                "persona_text":     persona.persona_text,
                "inferred_domains": persona.inferred_domains,
                "persona_uuid":     persona.uuid,
            },
            dataset_version="v2",
            created_at="2026-01-01",
        )

        goals = generate_goals_v2(
            user_id=user_id,
            persona=persona,
            min_goals=3,
            max_goals=5,
            seed=SEED + user_idx * 97,
            domain_counter=domain_counter,
            domain_targets=TARGET_DOMAIN_COUNTS,
            noise_counter=noise_counter,
            noise_targets=NOISE_TARGETS,
        )

        logs = generate_logs_v2(
            user_id=user_id,
            user_index=user_idx,
            goals=goals,
            start_date=START_DATE,
            end_date=END_DATE,
            min_logs=61,
            max_logs=81,
            seed=SEED + user_idx * 131,
            persona=persona,
        )

        labels = generate_labels_v2(
            goals=goals,
            logs=logs,
            user_id=user_id,
        )

        rel_count = sum(1 for lbl in labels if lbl.label == "relevant")
        total_rels  += rel_count
        total_pairs += len(goals) * len(logs)

        all_users.append(user)
        all_goals.extend(goals)
        all_logs.extend(logs)
        all_labels.extend(labels)

        noise_vals = ",".join(f"{g.noise_ratio:.1f}" for g in goals)
        log.info(
            "  [%s] goals=%d  logs=%d  labels=%d  rel=%d  noise=[%s]",
            user_id, len(goals), len(logs), len(labels), rel_count, noise_vals,
        )

    # ── 저장 ────────────────────────────────────────────────────────────────
    def _save(name: str, items: list) -> None:
        path = output_dir / f"{name}.json"
        data = [asdict(x) if hasattr(x, "__dataclass_fields__") else x for x in items]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        log.info("  저장: %s  (%d건)", path, len(items))

    log.info("\n[저장 중] %s", output_dir)
    _save("users",  all_users)
    _save("goals",  all_goals)
    _save("logs",   all_logs)
    _save("labels", all_labels)

    # ── 통계 ────────────────────────────────────────────────────────────────
    label_dist = Counter(lbl.label for lbl in all_labels)
    precision_ub = round(total_rels / (total_pairs + 1e-9), 3)

    goal_dicts = [asdict(g) for g in all_goals]
    goal_noise_counter: Counter[float] = Counter(
        round(g["noise_ratio"], 1) for g in goal_dicts
    )
    noise_dist_stats = [
        {
            "noise_ratio": n,
            "n_goals":     goal_noise_counter.get(n, 0),
            "target":      NOISE_TARGET,
        }
        for n in NOISE_LEVELS
    ]

    meta = {
        "version": "2.1",
        "description": "Persona-grounded per-goal noise-stratified synthetic dataset (A안)",
        "references": [
            "Chan et al. (2024). Scaling Synthetic Data Creation with 1,000,000,000 Personas.",
            "nvidia/Nemotron-Personas-Korea (HuggingFace)",
            "Simulating Authentic User Personas to Generate Analytics for Evaluating Digital Products.",
        ],
        "n_users":  len(all_users),
        "n_goals":  len(all_goals),
        "n_logs":   len(all_logs),
        "n_labels": len(all_labels),
        "label_distribution": dict(label_dist),
        "label_types": ["relevant", "irrelevant"],
        "precision_upper_bound": precision_ub,
        "noise_design": "per_goal_uniform (A안)",
        "noise_levels": NOISE_LEVELS,
        "noise_target_per_level": NOISE_TARGET,
        "noise_distribution": noise_dist_stats,
        "date_range": {"start": str(START_DATE), "end": str(END_DATE)},
        "logs_per_user": {"min": 61, "max": 81},
        "goals_per_user": {"min": 3, "max": 5},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )

    # ── 요약 출력 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  Dataset v2 빌드 완료  (A안: per-goal noise)")
    print("=" * 68)
    print(f"  users:  {len(all_users)}")
    print(f"  goals:  {len(all_goals)}")
    print(f"  logs:   {len(all_logs)}")
    print(f"  labels: {len(all_labels)}")
    print(f"\n  label 분포:")
    for k, v in label_dist.most_common():
        print(f"    {k:12s}: {v:6d}  ({v/len(all_labels)*100:.1f}%)")
    print(f"\n  precision upper bound (전체): {precision_ub:.3f}")

    print(f"\n  noise_ratio 분포 (target={NOISE_TARGET} each):")
    print(f"  {'noise':>6}  {'goals':>6}  {'diff':>5}")
    print("  " + "─" * 25)
    for row in noise_dist_stats:
        diff = row["n_goals"] - row["target"]
        diff_str = f"+{diff}" if diff >= 0 else str(diff)
        print(f"  {row['noise_ratio']:>6.1f}  {row['n_goals']:>6}  {diff_str:>5}")

    print(f"\n  도메인 분포 (target={TARGET_DOMAIN_COUNTS[ALL_DOMAINS[0]]} each):")
    for domain in ALL_DOMAINS:
        count = domain_counter.get(domain, 0)
        bar = "█" * count
        target = TARGET_DOMAIN_COUNTS[domain]
        diff = count - target
        diff_str = f"+{diff}" if diff >= 0 else str(diff)
        print(f"    {domain:15s}: {count:3d}  ({diff_str:>3})  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_v2",
                        help="출력 디렉토리 (default: data/synthetic_v2)")
    args = parser.parse_args()
    build_v2(ROOT / args.output)


if __name__ == "__main__":
    main()
