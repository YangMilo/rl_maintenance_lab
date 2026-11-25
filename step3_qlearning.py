"""실습 3 — Q-learning으로 정비 시점 배우기
에이전트는 열화 확률을 모른다. 1,000일짜리 연습을 1,000번 하며 점수표 Q를 채우고,
20번마다 그 시점의 정책을 같은 시나리오로 평가해 학습 곡선을 그린다.
실행: python step3_qlearning.py               (10~30초)
      python step3_qlearning.py --seed 3      (다른 학습 시드)
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from maintenance_lab import (N_COND, OUT, evaluate, load_json, mean_se, periodic, print_policy,
                             q_learning, reactive, save_figure, save_json, table_policy, test_seeds,
                             threshold, use_korean_font, value_iteration)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0, help="학습 시드 (0~4: 발표자료와 같음)")
ap.add_argument("--episodes", type=int, default=1000, help="연습 횟수 (1회 = 1,000일)")
ap.add_argument("--n-test", type=int, default=50, help="평가 시나리오 수 (발표자료: 200)")
args = ap.parse_args()

b = load_json("step2_baselines.json")
best_n, best_k = (b["best_n"], b["best_k"]) if b else (8, 7)
if not b:
    print("(실습 2 결과가 없어 발표자료의 값 n = 8, k = 7을 씁니다)")

t0 = time.time()
Q, snaps = q_learning(args.seed, episodes=args.episodes)
print(f"학습 끝: 에피소드 {args.episodes}회 = {args.episodes * 1000:,}일의 경험, {time.time() - t0:.0f}초")
table = np.argmax(Q, axis=2)
os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, f"q_table_seed{args.seed}.npy"), Q)

seeds = test_seeds(args.n_test)
cache = {}


def score_table(t):
    key = t[:N_COND - 1].tobytes()            # 상태 9(고장)는 행동과 상관없이 수리
    if key not in cache:
        cache[key] = mean_se(evaluate(table_policy(t), seeds))[0]
    return cache[key]


print("학습 곡선 평가 중", end="", flush=True)
curve = []
for ep, t in snaps:
    curve.append((ep, score_table(t)))
    print(".", end="", flush=True)
dp_table = np.argmax(value_iteration(), axis=2)
ref = {"사후 정비": mean_se(evaluate(reactive, seeds))[0],
       f"주기 정비({best_n}일)": mean_se(evaluate(periodic(best_n), seeds))[0],
       f"상태 기준({best_k} 이상)": mean_se(evaluate(threshold(best_k), seeds))[0],
       "정답(DP)": score_table(dp_table)}
print()

print_policy(table, f"강화학습이 배운 정책 (학습 시드 {args.seed})")
print_policy(dp_table, "정답(DP) 정책")
final = curve[-1][1]
dp = ref["정답(DP)"]
first = lambda v: next((ep for ep, s in curve if s >= v), None)
print(f"\n마지막 정책의 하루 평균 이익 {final:.2f} = 정답의 {final / dp * 100:.0f}%")
print(f"주기 정비를 처음 넘은 때: {first(ref[f'주기 정비({best_n}일)'])}회째,  정답의 95%: {first(0.95 * dp)}회째")
save_json({"seed": args.seed, "curve": curve, "reference": ref, "policy": table.tolist(),
           "dp_policy": dp_table.tolist()}, f"step3_seed{args.seed}.json")

use_korean_font()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot([e for e, _ in curve], [s for _, s in curve], color="#0B7A6F", lw=2, label="강화학습")
colors = {"사후 정비": "#D9573B", "정답(DP)": "#E8A13A"}
label_y, gap = {}, 0.32                       # 값이 비슷한 기준선 이름이 겹치지 않게 위아래로 벌린다
for name, v in sorted(ref.items(), key=lambda kv: kv[1]):
    label_y[name] = max([v] + [y + gap for y in label_y.values()])
for name, v in ref.items():
    ax.axhline(v, color=colors.get(name, "#8A97A6"), ls="--", lw=1)
    ax.text(args.episodes * 1.01, label_y[name], f"{name} {v:.2f}", va="center", fontsize=9, color=colors.get(name, "#4A5868"))
ax.set_xlim(0, args.episodes)
ax.set_xlabel("에피소드 (1회 = 1,000일)")
ax.set_ylabel("하루 평균 이익")
ax.set_title(f"학습 곡선 — 시나리오 {args.n_test}개로 평가, 학습 시드 {args.seed}", loc="left")
ax.grid(alpha=0.3)
fig.tight_layout()
save_figure(fig, f"step3_curve_seed{args.seed}.png")
