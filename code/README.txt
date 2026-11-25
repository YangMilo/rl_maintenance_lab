설비 정비 시점 문제 — 강화학습 실습 (2026.11.28 특강 PART 4)

자세한 설치 방법과 설명은 '강화학습 실습 가이드북'을 보세요. 순서만 적으면:

  1) 가상환경 만들기·켜기
       Windows:  python -m venv .venv   그리고   .\.venv\Scripts\Activate.ps1
       macOS  :  python3 -m venv .venv  그리고   source .venv/bin/activate
  2) 라이브러리 설치:   python -m pip install -r requirements.txt
  3) 설치 확인:         python check_setup.py
  4) 실습 1~4:          python step1_simulator.py
                        python step2_baselines.py
                        python step3_qlearning.py
                        python step4_compare.py
     (발표자료와 같은 숫자: step2·step4에 --n-test 200을 붙여 실행)
  5) 보너스:            python bonus_exact.py   (정책의 장기 평균 이익을 정확히 계산)

그림과 결과는 out 폴더에 저장됩니다.
파일: maintenance_lab.py = 공통 코드(시뮬레이터, 정비 규칙, Q-learning, 동적계획법)
