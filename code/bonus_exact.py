"""보너스 — 정책의 장기 평균 이익을 '정확히' 계산하기
정책이 (설비 상태, 수요)만 보고 행동을 정하면 하루하루의 변화는 마르코프 연쇄가 된다.
연쇄의 정상분포 π(오래 돌렸을 때 각 상태에 머무는 비율)를 구하면
장기 하루 평균 이익 = Σ π(상태) × 그 상태의 하루 보상 으로, 시뮬레이션 잡음 없이 정확히 나온다.
(주기 정비는 '마지막 정비 뒤 지난 날'도 봐야 해서 여기서는 뺀다)
실행: python bonus_exact.py   (몇 초. 실습 3을 먼저 돌렸으면 그때 배운 정책도 계산한다)
"""
import os

import numpy as np

from maintenance_lab import (N_COND, OUT, PM_COST, PM_DOWN, PROFIT, RP_COST, RP_DOWN, STAY, deg_probs,
                             ljust_kr, start_states, value_iteration)

STATES = [(c, d) for c in range(N_COND) for d in range(2)]
INDEX = {s: i for i, s in enumerate(STATES)}


def transition(table):
    """정책 표 → 전이 행렬 P와 하루 보상 r (MaintenanceEnv.step과 같은 규칙)"""
    n = len(STATES)
    P, r = np.zeros((n, n)), np.zeros(n)
    for (c, d), i in INDEX.items():
        if c == N_COND - 1:                              # 고장 → 수리
            r[i] = -(RP_COST + RP_DOWN * PROFIT[d])
            nxt = [(1.0, 0)]
        elif int(table[c][d]) == 1:                      # 계획 정비
            r[i] = -(PM_COST + PM_DOWN * PROFIT[d])
            nxt = [(1.0, 0)]
        else:                                            # 생산 → 0/1/2칸 나빠짐
            r[i] = PROFIT[d]
            nxt = [(p, min(N_COND - 1, c + k)) for k, p in enumerate(deg_probs(c))]
        for pc, c2 in nxt:
            for pd, d2 in ((STAY, d), (1 - STAY, 1 - d)):  # 수요는 STAY 확률로 그대로
                P[i, INDEX[(c2, d2)]] += pc * pd
    return P, r


def long_run_average(table):
    """정상분포 π (π P = π, π의 합 = 1)를 풀어 장기 하루 평균 이익을 돌려준다."""
    P, r = transition(table)
    n = len(STATES)
    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(pi @ r)


def rule(lo, hi):
    """비수기엔 상태 lo 이상, 성수기엔 상태 hi 이상이면 정비하는 정책 표"""
    return np.array([[int(c >= lo), int(c >= hi)] for c in range(N_COND)])


def show(x):
    return "없음" if x is None else x


dp_table = np.argmax(value_iteration(), axis=2)
best = long_run_average(dp_table)
rows = [("사후 정비", rule(N_COND, N_COND))] + [(f"상태 기준 ({k} 이상)", rule(k, k)) for k in range(5, 9)]
q_path = os.path.join(OUT, "q_table_seed0.npy")
if os.path.exists(q_path):
    rows.append(("강화학습 (실습 3, 시드 0)", np.argmax(np.load(q_path), axis=2)))
rows.append(("정답(DP)", dp_table))

print("장기 하루 평균 이익 — 정확한 값 (시뮬레이션 잡음 없음)\n")
for name, t in rows:
    v = long_run_average(t)
    lo, hi = start_states(t)
    print(f"  {ljust_kr(name, 26)} {v:7.3f}   정답의 {v / best * 100:5.1f}%   정비 시작: 비수기 {show(lo)}, 성수기 {show(hi)}")

# 비수기·성수기 기준을 따로 정하는 규칙(1~9 × 1~9 = 81가지)을 모두 계산하면?
v, lo, hi = max((long_run_average(rule(lo, hi)), lo, hi) for lo in range(1, N_COND) for hi in range(1, N_COND))
same = np.array_equal(rule(lo, hi)[:N_COND - 1], dp_table[:N_COND - 1])
print(f"\n수요별 기준 규칙 81가지 중 최고: 비수기 {lo}, 성수기 {hi} → {v:.3f}"
      + ("  (정답(DP)과 같은 정책)" if same else ""))
