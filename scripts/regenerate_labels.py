#!/usr/bin/env python3
"""레이블만 재생성 (로그 content는 유지).

_CATEGORY_KEYWORDS 수정 후 스켈레톤을 동일 시드로 재생성하여
레이블만 업데이트한다. logs.json은 변경하지 않는다.

Usage:
    cd /home/user/retrieval
    .venv/bin/python scripts/regenerate_labels.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_generation.export_utils import load_dataset_from_json
from app.data_generation.label_generator import generate_goal_log_labels
from app.data_generation.log_skeleton_generator import generate_log_skeletons_for_user
from app.schemas import GoalLogLabel

DATA_DIR = Path("data/synthetic")
START_DATE = date(2026, 3, 1)
END_DATE   = date(2026, 3, 31)


def main() -> None:
    print("[기존 데이터 로드]")
    users, goals, logs, old_labels = load_dataset_from_json(str(DATA_DIR))
    print(f"  users={len(users)}  goals={len(goals)}  logs={len(logs)}  labels={len(old_labels)}")

    old_dist: dict[str, int] = defaultdict(int)
    for lb in old_labels:
        old_dist[lb.label] += 1
    print(f"  기존 분포: {dict(old_dist)}")

    # 유저별 goal 그룹화
    user_goals: dict[str, list] = defaultdict(list)
    for g in goals:
        user_goals[g.user_id].append(g)

    # 유저 index 매핑 (seed 계산에 사용)
    user_list = [u.user_id for u in users]
    user_index = {uid: i for i, uid in enumerate(user_list)}

    print("\n[스켈레톤 재생성 + 레이블 재생성]")
    new_labels: list[GoalLogLabel] = []

    for user in users:
        uid    = user.user_id
        u_goals = user_goals[uid]
        u_logs  = [l for l in logs if l.user_id == uid]
        if not u_goals or not u_logs:
            continue

        uidx = user_index[uid]
        skeletons = generate_log_skeletons_for_user(
            user_id=uid,
            user_index=uidx,
            goals=u_goals,
            start_date=START_DATE,
            end_date=END_DATE,
            seed=42,
        )

        # 스켈레톤 log_id는 날짜 정렬 순서 기반 → 기존 logs.json과 동일
        u_labels = generate_goal_log_labels(u_goals, skeletons, seed=42 + uidx)
        new_labels.extend(u_labels)

    new_dist: dict[str, int] = defaultdict(int)
    for lb in new_labels:
        new_dist[lb.label] += 1
    print(f"  새 레이블: {len(new_labels)}개")
    print(f"  새 분포: {dict(new_dist)}")

    # 변화량
    print("\n[변화 분석]")
    old_map = {(lb.goal_id, lb.log_id): lb.label for lb in old_labels}
    new_map = {(lb.goal_id, lb.log_id): lb.label for lb in new_labels}

    changed = {k: (old_map[k], new_map[k]) for k in old_map
               if k in new_map and old_map[k] != new_map[k]}
    print(f"  레이블 변경: {len(changed)}개")

    change_types: dict[str, int] = defaultdict(int)
    for (gid, lid), (old_l, new_l) in changed.items():
        change_types[f"{old_l}→{new_l}"] += 1
    for k, v in sorted(change_types.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}개")

    # 저장
    out = DATA_DIR / "labels.json"
    labels_dicts = []
    for lb in new_labels:
        labels_dicts.append({
            "label_id":       lb.label_id,
            "user_id":        lb.user_id,
            "goal_id":        lb.goal_id,
            "log_id":         lb.log_id,
            "label":          lb.label,
            "relevance_score": lb.relevance_score,
            "label_source":   lb.label_source,
        })

    with open(out, "w", encoding="utf-8") as f:
        json.dump(labels_dicts, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out}  ({len(new_labels)}개)")


if __name__ == "__main__":
    main()
