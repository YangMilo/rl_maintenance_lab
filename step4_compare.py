"""실습 4 — 공정한 비교와 정답(DP)
① 모든 방법을 같은 시나리오 번호(seed)로 평가 ② 튜닝한 베이스라인 ③ 강화학습을 여러 번 따로 학습
④ 작은 문제라 구할 수 있는 정답(DP)과 비교 — 논문의 비교 실험과 같은 틀이다.
실행: python step4_compare.py                (시나리오 50개, 30초~1분)
      python step4_compare.py --n-test 200   (발표자료와 같은 설정, 1~2분)
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from maintenance_lab import (N_COND, evaluate, ljust_kr, load_json, mean_se, periodic, q_learning, reactive,
                             rjust_kr, save_figure, save_json, start_states, table_policy, test_seeds, threshold,
                             use_korean_font, value_iteration)

ap = argparse.ArgumentParser()
ap.add_argument("--n-test", type=int, default=50, help="비교 시나리오 수 (발표자료: 200)")
ap.add_argument("--runs", type=int, default=5, help="강화학습을 따로 학습하는 횟수 (시드 0, 1, 2, ...)")
args = ap.parse_args()
seeds = test_seeds(args.n_test)
b = load_json("step2_baselines.json")
best_n, best_k = (b["best_n"], b["best_k"]) if b else (8, 7)
t0 = time.time()
print(f"시나리오 {args.n_test}개 × 5,000일, 강화학습 {args.runs}회 학습. 주기 {best_n}일, 기준 상태 {best_k}.")

vals = {"사후 정비": evaluate(reactive, seeds),
        f"주기 정비 ({best_n}일)": evaluate(periodic(best_n), seeds),
        f"상태 기준 ({best_k} 이상)": evaluate(threshold(best_k), seeds)}
dp_table = np.argmax(value_iteration(), axis=2)
dp_vals = evaluate(table_policy(dp_table), seeds)

cache, runs = {}, []
for s in range(args.runs):
    Q, _ = q_learning(s)
    table = np.argmax(Q, axis=2)
    key = table[:N_COND - 1].tobytes()
    if key not in cache:
        cache[key] = evaluate(table_policy(table), seeds)
    runs.append({"seed": s, "values": cache[key], "start": start_states(table)})
    lo, hi = start_states(table)
    print(f"  학습 {s + 1}/{args.runs}: 하루 평균 이익 {cache[key].mean():.2f}, 정비 시작 상태 비수기 {lo} · 성수기 {hi}")
rl_avg = np.mean([r["values"] for r in runs], axis=0)       # 시나리오별 5회 평균
vals[f"강화학습 ({args.runs}회 평균)"] = rl_avg
vals["정답(DP)"] = dp_vals

dp = dp_vals.mean()
print("\n" + ljust_kr("방법", 20) + rjust_kr("하루 평균 이익", 16) + rjust_kr("표준오차", 10) + rjust_kr("정답 대비", 11))
for name, v in vals.items():
    m, se = mean_se(v)
    print(ljust_kr(name, 20) + f"{m:16.2f}{se:10.2f}{m / dp * 100:10.0f}%")
scores = [r["values"].mean() for r in runs]
print(f"강화학습 {args.runs}회: 최소 {min(scores):.2f}, 최대 {max(scores):.2f}")


def paired(a, c):
    d = a - c
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


thr_v = vals[f"상태 기준 ({best_k} 이상)"]
print("\n짝지은 비교 (같은 시나리오끼리 차이를 낸 뒤 평균):")
for name, a in [(f"강화학습({args.runs}회 평균) − 상태 기준", rl_avg), ("정답(DP) − 상태 기준", dp_vals)]:
    m, se = paired(a, thr_v)
    verdict = "차이가 뚜렷하다" if abs(m) > 2 * se else "통계적으로 구분하기 어렵다"
    print(f"  {name}: {m:+.3f} (표준오차 {se:.3f}) → {verdict}")
print(f"\n걸린 시간 {time.time() - t0:.0f}초")
save_json({"n_test": args.n_test, "best_n": best_n, "best_k": best_k,
           "results": {k: mean_se(v) for k, v in vals.items()},
           "rl_runs": [{"seed": r["seed"], "mean": float(r["values"].mean()), "start": r["start"]} for r in runs]},
          "step4_compare.json")

use_korean_font()
names = list(vals)
means = [vals[k].mean() for k in names]
colors = ["#D9573B", "#8A97A6", "#2F4A63", "#0B7A6F", "#E8A13A"]
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(names, means, color=colors[:len(names)])
for i, m in enumerate(means):
    ax.text(m + (0.08 if m >= 0 else -0.08), i, f"{m:.2f}", va="center", ha="left" if m >= 0 else "right", fontsize=10)
ax.axvline(0, color="#4A5868", lw=0.8)
ax.set_xlim(min(-1.5, min(means) - 1), max(means) * 1.15)
ax.set_xlabel(f"하루 평균 이익 (시나리오 {args.n_test}개 × 5,000일, 모든 방법이 같은 시나리오 번호)")
ax.set_title("공정한 비교: 강화학습은 정답(DP)에 가깝고, 튜닝한 상태 기준 규칙과 비슷하다", loc="left", fontsize=11)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
save_figure(fig, "step4_compare.png")
