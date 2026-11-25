"""설치 확인: 파이썬과 라이브러리 버전을 보여 주고, 시뮬레이터를 하루 돌려 본다.
실행: python check_setup.py
"""
import sys

print(f"파이썬 {sys.version.split()[0]}   ({sys.executable})")
in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
print("가상환경:", "켜짐" if in_venv else "꺼짐  ← 가이드북 2장대로 가상환경을 켜고 다시 실행하세요")
if sys.version_info < (3, 10):
    print("\n파이썬 3.10 이상이 필요합니다. 가이드북 1장을 보고 새로 설치해 주세요.")
    sys.exit(1)

missing = []
for name in ("gymnasium", "numpy", "matplotlib"):
    try:
        mod = __import__(name)
        print(f"  {name:<11} {mod.__version__}")
    except ImportError:
        missing.append(name)
        print(f"  {name:<11} 설치 안 됨")
if missing:
    print("\n라이브러리가 빠졌습니다. 가상환경을 켠 상태에서 다음을 실행하세요:")
    print("  python -m pip install -r requirements.txt")
    sys.exit(1)

from maintenance_lab import MaintenanceEnv

env = MaintenanceEnv(horizon=5)
obs, _ = env.reset(seed=1)
obs2, reward, *_ = env.step(0)
print(f"\n시뮬레이터 확인: 상태 {obs} → 생산 → 상태 {obs2}, 보상 {reward}")
print("설치 완료! 이제 실습 1을 실행하세요:  python step1_simulator.py")
