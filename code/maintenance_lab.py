"""
설비 정비 시점 문제 — 강화학습 실습 공통 코드 (2026.11.28 특강 PART 4)
=========================================================================

step1~step4 스크립트가 이 파일을 불러 씁니다. 이 파일은 직접 실행하지 않습니다.

상황  : 설비 한 대가 날마다 생산하면서 조금씩 나빠진다. 상태 0(새것) ~ 9(고장).
        상태가 5 이상이면 열화가 빨라진다. 수요는 비수기/성수기가 번갈아 온다.
결정  : 매일 아침 '생산(0)' 또는 '계획 정비(1)'를 고른다.
보상  : 생산하면 그날 이익(비수기 2, 성수기 20)
        계획 정비: 고정비 10 + 3일 생산 중단(그 시기 하루 이익 x 3)
        고장(상태 9): 강제 수리, 고정비 100 + 6일 생산 중단(그 시기 하루 이익 x 6)
목표  : 하루 평균 이익을 최대로.
"""
from __future__ import annotations

import json
import os
import unicodedata

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")            # 그림과 결과 파일이 저장되는 폴더

# ---------------------------------------------------------------------------
# 1) 문제 설정 — 도전 과제에서 이 숫자들을 바꿔 봅니다
# ---------------------------------------------------------------------------
N_COND = 10                     # 설비 상태 0..9, 9 = 고장
PROFIT = (2.0, 20.0)            # 하루 이익: (비수기, 성수기)
PM_COST, PM_DOWN = 10.0, 3      # 계획 정비: 고정비, 생산 중단 일수
RP_COST, RP_DOWN = 100.0, 6     # 고장 수리: 고정비, 생산 중단 일수
STAY = 0.95                     # 수요가 다음 날에도 그대로일 확률 (평균 20일씩 지속)
DEMAND_NAME = ("비수기", "성수기")


def deg_probs(c: int):
    """하루 생산한 뒤 상태가 0/1/2칸 나빠질 확률. 상태 5부터 열화가 빨라진다."""
    return (0.6, 0.3, 0.1) if c < 5 else (0.3, 0.4, 0.3)


# ---------------------------------------------------------------------------
# 2) 시뮬레이터 (Gymnasium 표준 인터페이스: reset / step)
# ---------------------------------------------------------------------------
class MaintenanceEnv(gym.Env):
    """관측 = (설비 상태 0~9, 수요 0=비수기/1=성수기), 행동 = 0 생산 / 1 계획 정비"""

    def __init__(self, horizon: int = 1000):
        self.horizon = horizon
        self.observation_space = gym.spaces.MultiDiscrete([N_COND, 2])
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                 # seed = 시나리오 번호. 같은 번호·같은 행동이면 똑같이 재현된다
        self.cond, self.demand, self.t = 0, 0, 0
        return (self.cond, self.demand), {}

    def step(self, action):
        c, d = self.cond, self.demand
        event = "produce"
        if c == N_COND - 1:                                   # 고장 → 강제 수리
            reward = -(RP_COST + RP_DOWN * PROFIT[d]); self.cond = 0; event = "failure"
        elif action == 1:                                     # 계획 정비
            reward = -(PM_COST + PM_DOWN * PROFIT[d]); self.cond = 0; event = "maintain"
        else:                                                 # 생산 → 조금씩 나빠진다
            reward = PROFIT[d]
            p0, p1, _ = deg_probs(c)
            u = self.np_random.random()
            k = 0 if u < p0 else (1 if u < p0 + p1 else 2)
            self.cond = min(N_COND - 1, c + k)
        if self.np_random.random() > STAY:                    # 수요 변화
            self.demand = 1 - d
        self.t += 1
        return (self.cond, self.demand), reward, False, self.t >= self.horizon, {"event": event}


# ---------------------------------------------------------------------------
# 3) 정책(정비 방식)들 — policy(설비 상태, 수요, 마지막 정비 뒤 지난 날) -> 행동
# ---------------------------------------------------------------------------
def reactive(c, d, since):            # 사후 정비: 정비하지 않고 고장 나면 고친다
    return 0


def periodic(n):                      # 주기 정비: n일 생산할 때마다 정비
    return lambda c, d, since: int(since >= n)


def threshold(k):                     # 상태 기준 정비: 상태 k 이상이면 정비
    return lambda c, d, since: int(c >= k)


def table_policy(table):              # 점수표(Q)나 DP 결과를 정책으로: table[상태][수요] = 행동
    return lambda c, d, since: int(table[c][d])


def run_policy(policy, seed: int, days: int = 5000, trace: bool = False):
    """한 시나리오(seed)를 days일 돌려 하루 평균 이익을 돌려준다. trace=True면 날마다 기록도."""
    env = MaintenanceEnv(horizon=days)
    (c, d), _ = env.reset(seed=seed)
    total, since, log = 0.0, 0, []
    while True:
        a = policy(c, d, since)
        (c2, d2), r, _, trunc, info = env.step(a)
        if trace:
            log.append((c, d, a, info["event"], r))
        since = 0 if info["event"] in ("maintain", "failure") else since + 1
        total += r
        c, d = c2, d2
        if trunc:
            break
    return (total / days, log) if trace else total / days


def test_seeds(n_test: int):
    """비교에 쓰는 시나리오 번호. 모든 방법이 같은 번호(같은 난수)를 쓴다."""
    return list(range(2000, 2000 + n_test))


def evaluate(policy, seeds, days: int = 5000):
    """시나리오별 하루 평균 이익 (numpy 배열)."""
    return np.array([run_policy(policy, s, days) for s in seeds])


def mean_se(v):
    """평균과 표준오차."""
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std() / np.sqrt(len(v)))


def rates(policy, seeds, days: int = 5000):
    """1,000일당 고장·계획 정비 횟수."""
    f = pm = 0
    for s in seeds:
        _, log = run_policy(policy, s, days, trace=True)
        f += sum(e[3] == "failure" for e in log)
        pm += sum(e[3] == "maintain" for e in log)
    k = len(seeds) * days / 1000
    return f / k, pm / k


# ---------------------------------------------------------------------------
# 4) 정답: 동적계획법(DP) — 열화 확률을 모두 안다고 가정하고 계산
# ---------------------------------------------------------------------------
def value_iteration(gamma=0.995):
    V = np.zeros((N_COND, 2))
    while True:
        Q = np.zeros((N_COND, 2, 2))
        for c in range(N_COND):
            for d in range(2):
                for a in range(2):
                    nxt_d = [(STAY, d), (1 - STAY, 1 - d)]
                    if c == N_COND - 1 or a == 1:
                        r = -(RP_COST + RP_DOWN * PROFIT[d]) if c == N_COND - 1 else -(PM_COST + PM_DOWN * PROFIT[d])
                        Q[c, d, a] = sum(p * (r + gamma * V[0, d2]) for p, d2 in nxt_d)
                    else:
                        Q[c, d, a] = sum(pk * p * (PROFIT[d] + gamma * V[min(N_COND - 1, c + k), d2])
                                         for k, pk in enumerate(deg_probs(c)) for p, d2 in nxt_d)
        V2 = Q.max(axis=2)
        if np.abs(V2 - V).max() < 1e-8:
            return Q
        V = V2


# ---------------------------------------------------------------------------
# 5) 강화학습: 표 형태 Q-learning
# ---------------------------------------------------------------------------
def q_learning(seed, episodes=1000, horizon=1000, gamma=0.98, alpha0=0.1, alpha_min=0.005, snap_every=20):
    """에이전트는 열화 확률을 모른다. 시뮬레이터에서 겪으며 점수표 Q[상태, 수요, 행동]를 채운다.
    학습 곡선을 그리려고 snap_every 에피소드마다 그 시점의 정책(점수표에서 고른 행동)을 저장한다."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_COND, 2, 2))
    visits = np.zeros((N_COND, 2, 2))
    env = MaintenanceEnv(horizon=horizon)
    snaps = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.5 * episodes))          # 탐험 비율: 처음엔 100%, 절반 지점부터 5%
        (c, d), _ = env.reset(seed=int(rng.integers(1 << 30)))
        while True:
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[c, d]))   # 탐험 / 활용
            (c2, d2), r, _, trunc, _ = env.step(a)
            visits[c, d, a] += 1
            alpha = max(alpha_min, alpha0 / (1 + visits[c, d, a] / 500))                 # 학습률: 점점 작게
            Q[c, d, a] += alpha * (r + gamma * np.max(Q[c2, d2]) - Q[c, d, a])           # Q값 갱신
            c, d = c2, d2
            if trunc:
                break
        if ep == 0 or (ep + 1) % snap_every == 0:
            snaps.append((ep + 1, np.argmax(Q, axis=2).copy()))
    return Q, snaps


def start_states(table):
    """정책 표에서 '정비를 시작하는 상태'를 (비수기, 성수기)로. 정비하지 않으면 None."""
    out = []
    for d in range(2):
        col = [int(table[c][d]) for c in range(N_COND - 1)]   # 상태 9(고장)는 자동 수리라 제외
        out.append(col.index(1) if 1 in col else None)
    return tuple(out)


def print_policy(table, title):
    """정책 표를 글자로 출력: '.' = 생산, 'M' = 계획 정비"""
    print(f"\n{title}")
    print("  " + ljust_kr("설비 상태", 10) + ": " + " ".join(str(c) for c in range(N_COND - 1)))
    for d in range(2):
        row = " ".join("M" if int(table[c][d]) == 1 else "." for c in range(N_COND - 1))
        print("  " + ljust_kr(DEMAND_NAME[d], 10) + ": " + row)
    lo, hi = start_states(table)
    print(f"  → 정비 시작 상태: 비수기 {lo}, 성수기 {hi}   (M = 계획 정비, . = 생산)")


# ---------------------------------------------------------------------------
# 6) 출력·저장·그림 도우미
# ---------------------------------------------------------------------------
def text_width(s) -> int:
    """터미널에서 차지하는 칸 수. 한글처럼 넓은 글자는 2칸으로 센다."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(s))


def ljust_kr(s, width: int) -> str:
    """한글 폭을 고려한 왼쪽 정렬 (표 칸을 맞추는 용도)."""
    s = str(s)
    return s + " " * max(0, width - text_width(s))


def rjust_kr(s, width: int) -> str:
    """한글 폭을 고려한 오른쪽 정렬."""
    s = str(s)
    return " " * max(0, width - text_width(s)) + s


def save_json(obj, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return path


def load_json(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def use_korean_font():
    """그림에 한글이 네모(□)로 나오지 않도록 운영체제의 한글 글꼴을 찾아 쓴다."""
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if cand in names:
            matplotlib.rcParams["font.family"] = cand
            break
    else:
        print("[알림] 한글 글꼴을 찾지 못해 그림의 한글이 네모로 보일 수 있습니다. (가이드북 9장 참고)")
    matplotlib.rcParams["axes.unicode_minus"] = False


def save_figure(fig, name):
    """그림을 out 폴더에 저장하고, 창을 띄울 수 있는 환경이면 화면에도 보여 준다."""
    import matplotlib
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n그림 저장: {path}")
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template"):
        print("(그림 창을 닫으면 프로그램이 끝납니다)")
        plt.show()
    return path


def draw_trace(ax, log, title, thr=None):
    """150일 동안의 설비 상태: 노란 띠 = 성수기, 초록 점 = 계획 정비, 빨간 점 = 고장"""
    n = len(log)
    t = 0
    while t < n:
        if log[t][1] == 1:
            u = t
            while u < n and log[u][1] == 1:
                u += 1
            ax.axvspan(t, u, color="#E8A13A", alpha=0.18, lw=0)
            t = u
        else:
            t += 1
    ax.step(range(n), [e[0] for e in log], where="post", color="#2F4A63", lw=1.2)
    ax.axhline(N_COND - 1, color="#D9573B", ls="--", lw=0.8)
    if thr is not None:
        ax.axhline(thr, color="#2F4A63", ls=":", lw=0.8)
    pm = [i for i, e in enumerate(log) if e[3] == "maintain"]
    fl = [i for i, e in enumerate(log) if e[3] == "failure"]
    ax.scatter(pm, [log[i][0] for i in pm], s=16, color="#0B7A6F", zorder=3, label="계획 정비")
    ax.scatter(fl, [N_COND - 1] * len(fl), s=22, color="#D9573B", zorder=3, label="고장")
    ax.set_ylim(-0.5, N_COND - 0.3)
    ax.set_yticks([0, 3, 6, 9])
    ax.set_title(title, loc="left", fontsize=11)
