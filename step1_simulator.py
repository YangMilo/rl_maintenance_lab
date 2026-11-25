"""실습 1 — 시뮬레이터 살펴보기
(1) '생산만 하는' 방식으로 하루씩 돌리며 상태·수요·보상이 어떻게 바뀌는지 본다.
(2) 세 가지 기존 정비 방식을 같은 시나리오 번호(seed 7)로 150일 돌려 그림으로 비교한다.
    seed가 같아도 행동이 다르면 그 뒤 난수가 쓰이는 순서가 달라져, 날짜별 상황(수요 등)은 조금씩 다르다.
실행: python step1_simulator.py   (몇 초)
"""
import matplotlib.pyplot as plt

from maintenance_lab import (DEMAND_NAME, MaintenanceEnv, draw_trace, ljust_kr, periodic, reactive,
                             rjust_kr, run_policy, save_figure, threshold, use_korean_font)

EVENT = {"produce": "생산", "maintain": "계획 정비", "failure": "고장 수리"}

# (1) 하루씩 직접 돌려 보기 ------------------------------------------------------
env = MaintenanceEnv(horizon=40)
(c, d), _ = env.reset(seed=7)                 # seed = 시나리오 번호. 같은 번호·같은 행동이면 똑같이 재현된다
print("관측 공간:", env.observation_space, "  행동 공간:", env.action_space)
print("\n" + " | ".join([rjust_kr("날", 4), rjust_kr("상태", 4), ljust_kr("수요", 6), ljust_kr("행동", 4),
                         ljust_kr("일어난 일", 9), rjust_kr("보상", 7)]))
print("-+-".join("-" * w for w in (4, 4, 6, 4, 9, 7)))
for day in range(1, 41):
    action = 0                                # 0 = 생산, 1 = 계획 정비 (지금은 생산만)
    (c2, d2), reward, terminated, truncated, info = env.step(action)
    print(" | ".join([f"{day:4d}", f"{c:4d}", ljust_kr(DEMAND_NAME[d], 6), ljust_kr("정비" if action else "생산", 4),
                      ljust_kr(EVENT[info["event"]], 9), f"{reward:7.1f}"]))
    c, d = c2, d2
    if truncated:
        break

# (2) 세 가지 정비 방식, 같은 시나리오 번호로 150일 ------------------------------
policies = [("사후 정비 (고장 나면 고친다)", reactive, None),
            ("주기 정비 (8일마다)", periodic(8), None),
            ("상태 기준 정비 (상태 7 이상이면)", threshold(7), 7)]
use_korean_font()
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
print("\n시나리오 seed 7, 150일 동안:")
for ax, (name, pol, thr) in zip(axes, policies):
    avg, log = run_policy(pol, seed=7, days=150, trace=True)
    n_fail = sum(e[3] == "failure" for e in log)
    n_pm = sum(e[3] == "maintain" for e in log)
    print(f"  {ljust_kr(name, 34)} 고장 {n_fail:2d}회, 계획 정비 {n_pm:2d}회")
    draw_trace(ax, log, name, thr)
axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9, frameon=False)   # 그래프 위쪽 바깥
axes[-1].set_xlabel("날 (노란 띠 = 성수기)")
axes[1].set_ylabel("설비 상태 (9 = 고장)")
fig.suptitle("세 가지 정비 방식, 같은 시나리오(seed 7)로 150일", x=0.01, ha="left")
fig.tight_layout()
save_figure(fig, "step1_traces.png")
