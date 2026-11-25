"""실습 2 — 기존 정비 방식과 튜닝
사후 정비, 주기 정비(n일마다), 상태 기준 정비(상태 k 이상)를 같은 시나리오로 평가하고
가장 좋은 n과 k를 찾는다. 베이스라인도 튜닝해야 강화학습과 공정하게 비교할 수 있다.
실행: python step2_baselines.py              (시나리오 50개, 10~30초)
      python step2_baselines.py --n-test 200 (발표자료와 같은 설정, 1~2분)
"""
import argparse
import time

import matplotlib.pyplot as plt

from maintenance_lab import (evaluate, ljust_kr, mean_se, periodic, rates, reactive, rjust_kr, save_figure,
                             save_json, test_seeds, threshold, use_korean_font)

ap = argparse.ArgumentParser()
ap.add_argument("--n-test", type=int, default=50, help="비교에 쓸 시나리오 수 (발표자료: 200)")
ap.add_argument("--days", type=int, default=5000, help="시나리오 하나의 길이(일)")
args = ap.parse_args()
seeds = test_seeds(args.n_test)
t0 = time.time()
print(f"시나리오 {args.n_test}개 × {args.days:,}일로 평가합니다.")

reac = mean_se(evaluate(reactive, seeds, args.days))
print("주기 정비 n = 2~30일 시험 중", end="", flush=True)
per = {}
for n in range(2, 31):
    per[n] = mean_se(evaluate(periodic(n), seeds, args.days))
    print(".", end="", flush=True)
print("\n상태 기준 k = 1~8 시험 중", end="", flush=True)
thr = {}
for k in range(1, 9):
    thr[k] = mean_se(evaluate(threshold(k), seeds, args.days))
    print(".", end="", flush=True)
best_n = max(per, key=lambda n: per[n][0])
best_k = max(thr, key=lambda k: thr[k][0])

print("\n\n" + ljust_kr("방법", 20) + rjust_kr("하루 평균 이익", 16) + rjust_kr("표준오차", 10)
      + rjust_kr("1,000일당 고장", 16) + rjust_kr("1,000일당 정비", 16))
for name, (m, se), pol in [("사후 정비", reac, reactive),
                           (f"주기 정비 ({best_n}일)", per[best_n], periodic(best_n)),
                           (f"상태 기준 ({best_k} 이상)", thr[best_k], threshold(best_k))]:
    f, pm = rates(pol, seeds[:20], args.days)
    print(ljust_kr(name, 20) + f"{m:16.2f}{se:10.2f}{f:16.1f}{pm:16.1f}")
print(f"\n가장 좋은 주기 n = {best_n}일, 가장 좋은 기준 k = {best_k}  (걸린 시간 {time.time() - t0:.0f}초)")
path = save_json({"n_test": args.n_test, "days": args.days, "best_n": best_n, "best_k": best_k,
                  "reactive": reac, "periodic": per, "threshold": thr}, "step2_baselines.json")
print(f"결과 저장: {path}  (실습 3·4가 이 파일의 n, k를 씁니다)")

use_korean_font()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
ns = list(per)
a1.errorbar(ns, [per[n][0] for n in ns], yerr=[per[n][1] for n in ns], color="#8A97A6", marker="o", ms=3, lw=1)
a1.scatter([best_n], [per[best_n][0]], color="#0B7A6F", s=60, zorder=3)
a1.annotate(f"최고: {best_n}일 ({per[best_n][0]:.2f})", (best_n, per[best_n][0]), xytext=(10, 10), textcoords="offset points")
a1.set_title("주기 정비: 며칠마다 정비할까", loc="left")
a1.set_xlabel("정비 주기 n (일)")
a1.set_ylabel("하루 평균 이익")
ks = list(thr)
a2.errorbar(ks, [thr[k][0] for k in ks], yerr=[thr[k][1] for k in ks], color="#8A97A6", marker="o", ms=4, lw=1)
a2.scatter([best_k], [thr[best_k][0]], color="#0B7A6F", s=60, zorder=3)
a2.annotate(f"최고: 상태 {best_k} ({thr[best_k][0]:.2f})", (best_k, thr[best_k][0]), xytext=(-110, -25), textcoords="offset points")
a2.set_title("상태 기준 정비: 상태 몇부터 정비할까", loc="left")
a2.set_xlabel("정비 기준 상태 k")
for a in (a1, a2):
    a.axhline(reac[0], color="#D9573B", ls="--", lw=0.8)
    a.grid(alpha=0.3)
a2.text(8, reac[0] + 0.25, "사후 정비", color="#D9573B", ha="right")
fig.tight_layout()
save_figure(fig, "step2_tuning.png")
