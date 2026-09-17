# eval/compare.py
import json
from pathlib import Path

def compare(baseline_path, improved_path):
    b = json.load(open(baseline_path))
    i = json.load(open(improved_path))

    print(f"Suite: {b['suite_name']}")
    print(f"  Pass rate:  {b['pass_rate']:.1%} → {i['pass_rate']:.1%}")
    print(f"  Avg score:  {b['avg_score']:.3f} → {i['avg_score']:.3f}")
    print(f"  Avg steps:  {b['avg_steps']:.1f} → {i['avg_steps']:.1f}")

    # 找出退化/改进的 task
    b_by_id = {r['task_id']: r for r in b['results']}
    i_by_id = {r['task_id']: r for r in i['results']}

    regressed = []
    improved = []
    for tid in b_by_id:
        if tid in i_by_id:
            delta = i_by_id[tid]['score'] - b_by_id[tid]['score']
            if delta < -0.1:
                regressed.append((tid, delta))
            elif delta > 0.1:
                improved.append((tid, delta))

    if regressed:
        print("\n  ⚠ Regressed:")
        for tid, d in regressed:
            print(f"    {tid}: {d:+.2f}")
    if improved:
        print("\n  ✅ Improved:")
        for tid, d in improved:
            print(f"    {tid}: {d:+.2f}")