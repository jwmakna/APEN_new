# -*- coding: utf-8 -*-

# =====================================================================
# 논문_제안_모형.py
#
# 목적: "논문 제안 모형 AR" (논문 Eq.9 - 발전계획 최적화를 회귀 학습에
#       통합한 모형) 의 nRMSE 와 optimality gap 을 구해서
#       D:\03_JiWon\APEN\Readme.md 결과표의 "논문 제안 모형 AR" 행처럼
#       뽑는다. 기본_모형_AR.py 와 입력/평가 방식은 같고, 회귀를 푸는
#       방법만 "MILP(혼합정수계획법)" 로 바뀐다 - 예측오차뿐 아니라
#       발전계획 이익도 같이 최소화하도록 계수를 학습한다.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
#
# 데이터: merged_for_simulation_z03.csv (Zone1)
# 구간(블록9): 학습 2012-11-28~2013-09-23(300일),
#             테스트 2013-09-24~2014-01-01(100일)
# 가중치: W1=1, W2=20, 벌금비용률=50% (논문 Table 3 비교 지점)
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
import gurobipy as gp                        # Gurobi 최적화 라이브러리
from gurobipy import GRB                     # Gurobi 상수


# =====================================================================
# 0. 설정값
# =====================================================================
BASE_DIR = os.path.abspath(r"D:\Users\Downloads\mlr 구로비 라이선스 돌려보려고")           # 이 파이썬 파일이 있는 폴더
MERGED_FILE = os.environ.get("MERGED_FILE", os.path.join(BASE_DIR, "merged_for_simulation_z03.csv"))  # 데이터 파일 경로 (환경변수로 덮어쓰기 가능)

HOURS_PER_DAY = 12               # 하루 낮 시간대 개수 (Sydney 현지시간 9시~20시)
LOCAL_HOUR_START = 9             # 낮 시간대 시작 시(local_hour 기준)
LOCAL_HOUR_END = 21              # 낮 시간대 끝(이 값 미만까지, 즉 9~20시)

TRAIN_START = pd.Timestamp(os.environ.get("TRAIN_START", "2013-08-25"))   # 학습 시작일
TRAIN_END = pd.Timestamp(os.environ.get("TRAIN_END",   "2013-11-22"))     # 학습 마지막일 (300일째)
TEST_START = pd.Timestamp(os.environ.get("TEST_START",  "2013-11-23"))    # 테스트 시작일
TEST_END = pd.Timestamp(os.environ.get("TEST_END",    "2013-12-22"))      # 테스트 마지막일 (100일째)
HISTORY_DATE = TRAIN_START - pd.Timedelta(days=1)   # 학습 첫날 직전 하루를 자동으로 계산

CAPACITY_MW = 30.0                # 태양광 패널 설비 최대 용량 (논문 가정)
DURATION_HOURS = 1.0              # 한 시간대의 길이(시간)
PENALTY_RATE = 0.5                # 약정 부족(shortage) 시 벌금비용률 (일간전 가격의 50%)

W1 = float(os.environ.get("SWEEP_W1", "1.0"))    # 정규화 파라미터 W1 (환경변수로 스윕 가능)
W2 = float(os.environ.get("SWEEP_W2", "20.0"))    # 정규화 파라미터 W2 (환경변수로 스윕 가능)

PAPER_NRMSE = 34.89                # 논문 Table 3, 논문 제안 모형 AR 의 nRMSE(%) - 비교용
PAPER_GAP = 13.91                  # 논문 Table 3, 논문 제안 모형 AR 의 optimality gap(%) - 비교용

CVAR_ALPHA = 0.90

# additive CVaR 가중치: 기존 전체 regret 항은 그대로 유지한다.
CVAR_LAMBDA = 0.05  # CVaR 적용

if not (0.0 < CVAR_ALPHA < 1.0):
    raise ValueError(
        "CVAR_ALPHA는 0과 1 사이여야 합니다."
    )

if CVAR_LAMBDA < 0.0:
    raise ValueError(
        "CVAR_LAMBDA는 0 이상이어야 합니다."
    )

# =====================================================================
# 1. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기
#    (기본_모형_AR.py 와 동일한 절차)
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)                       # csv 파일 전체를 한 번에 읽어옴
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])   # local_date 열을 날짜 타입으로 변환

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
# ↑ local_hour 가 9시 이상, 21시 미만(=9~20시)인 행만 True 인 판단 열을 만듦

daylight_table = raw_table[is_daylight].copy()              # 낮 시간대 행만 골라서 새 표로 복사
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START
# ↑ local_hour(9~20) 를 0~11 로 다시 번호 매김 (hour_idx)


# =====================================================================
# 2. 이력(history) / 학습(train) / 테스트(test) 구간으로 자르기
# =====================================================================
is_history_date = daylight_table["local_date"] == HISTORY_DATE     # 이력 날짜(2012-11-27)인지 판단
history_rows = daylight_table[is_history_date].copy()               # 이력 날짜 행만 골라냄
history_rows = history_rows.sort_values("hour_idx")                  # 시간대(0~11) 순서로 정렬

is_train_date = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train_date].copy()                    # 학습 구간 행만 골라냄
train_rows = train_rows.sort_values(["local_date", "hour_idx"])       # 날짜, 시간대 순서로 정렬

is_test_date = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test_date].copy()                      # 테스트 구간 행만 골라냄
test_rows = test_rows.sort_values(["local_date", "hour_idx"])         # 날짜, 시간대 순서로 정렬

print("이력 날짜:", HISTORY_DATE.date(), "행 수:", len(history_rows))          # 이력 행 수 출력 (12여야 정상)
print("학습 구간:", TRAIN_START.date(), "~", TRAIN_END.date(), "행 수:", len(train_rows))  # 학습 행 수 출력 (3600 이어야 정상)
print("테스트 구간:", TEST_START.date(), "~", TEST_END.date(), "행 수:", len(test_rows))    # 테스트 행 수 출력 (1200 이어야 정상)


# =====================================================================
# 3. (날짜 x 12시간) 모양의 숫자 배열로 바꾸기
#    - 학습 구간은 이번엔 발전량뿐 아니라 DA/RT 가격도 같이 필요함
#      (제안 모형은 학습 때 가격도 손실함수에 쓰기 때문)
# =====================================================================

# --- 3-1. 이력(history) 하루치 발전량 배열 (1, 12) ---
history_solar = np.zeros((1, HOURS_PER_DAY))          # 0으로 채운 빈 배열 준비 (1일 x 12시간)
row_counter = 0                                          # history_rows 를 순서대로 셀 카운터
for _, one_row in history_rows.iterrows():                # history_rows 를 한 줄씩 순서대로 확인
    hour_position = row_counter % HOURS_PER_DAY            # 이 행이 몇 번째 시간대인지 (0~11)
    history_solar[0, hour_position] = one_row["solar_power"]  # 해당 칸에 발전량 값을 채워 넣음
    row_counter = row_counter + 1                            # 카운터를 하나 증가시킴

# --- 3-2. 학습(train) 300일치 발전량 / DA가격 / RT가격 배열 (300, 12) ---
train_dates_sorted = sorted(train_rows["local_date"].unique())  # 학습 구간의 날짜들을 오래된 순으로 정렬한 목록
n_train_days = len(train_dates_sorted)                            # 학습 날짜 수 (300 이어야 정상)
train_solar = np.zeros((n_train_days, HOURS_PER_DAY))              # 발전량을 담을 빈 배열
train_da_price = np.zeros((n_train_days, HOURS_PER_DAY))            # DA가격을 담을 빈 배열
train_rt_price = np.zeros((n_train_days, HOURS_PER_DAY))            # RT가격을 담을 빈 배열
row_counter = 0                                                      # train_rows 를 순서대로 셀 카운터
for _, one_row in train_rows.iterrows():                             # train_rows 를 한 줄씩 순서대로 확인
    day_position = row_counter // HOURS_PER_DAY                        # 이 행이 몇 번째 날짜인지 (0~299)
    hour_position = row_counter % HOURS_PER_DAY                        # 이 행이 몇 번째 시간대인지 (0~11)
    train_solar[day_position, hour_position] = one_row["solar_power"]   # 발전량 값을 채워 넣음
    train_da_price[day_position, hour_position] = one_row["da_price"]   # DA가격 값을 채워 넣음
    train_rt_price[day_position, hour_position] = one_row["rt_price"]   # RT가격 값을 채워 넣음
    row_counter = row_counter + 1                                        # 카운터를 하나 증가시킴

# --- 3-3. 테스트(test) 100일치 발전량 / 가격 배열 (100, 12) ---
test_dates_sorted = sorted(test_rows["local_date"].unique())     # 테스트 구간의 날짜들을 오래된 순으로 정렬한 목록
n_test_days = len(test_dates_sorted)                               # 테스트 날짜 수 (100 이어야 정상)
test_solar = np.zeros((n_test_days, HOURS_PER_DAY))                 # 실제 발전량을 담을 빈 배열
test_da_price = np.zeros((n_test_days, HOURS_PER_DAY))              # DA가격을 담을 빈 배열
test_rt_price = np.zeros((n_test_days, HOURS_PER_DAY))              # RT가격을 담을 빈 배열
row_counter = 0                                                       # test_rows 를 순서대로 셀 카운터
for _, one_row in test_rows.iterrows():                               # test_rows 를 한 줄씩 순서대로 확인
    day_position = row_counter // HOURS_PER_DAY                          # 이 행이 몇 번째 날짜인지 (0~99)
    hour_position = row_counter % HOURS_PER_DAY                          # 이 행이 몇 번째 시간대인지 (0~11)
    test_solar[day_position, hour_position] = one_row["solar_power"]      # 실제 발전량 값을 채워 넣음
    test_da_price[day_position, hour_position] = one_row["da_price"]      # DA 가격 값을 채워 넣음
    test_rt_price[day_position, hour_position] = one_row["rt_price"]      # RT 가격 값을 채워 넣음
    row_counter = row_counter + 1                                          # 카운터를 하나 증가시킴


# =====================================================================
# 4. AR 학습용 입력(X) 만들기 - 논문 Eq.(3): "직전 하루"의 12시간을
#    입력으로 씀 (기본_모형_AR.py 와 동일)
# =====================================================================

history_and_train_solar = np.vstack([history_solar, train_solar])   # 이력 하루 + 학습 300일을 이어붙임 (301, 12)

n_ar_rows = n_train_days                                             # AR 학습 표본 개수 = 300개
ar_intercept_column = np.ones((n_ar_rows, 1))                        # 절편(intercept)용 1로만 채운 열 (300, 1)
ar_lag_features = np.zeros((n_ar_rows, HOURS_PER_DAY))                # "직전 하루" 12시간 값을 담을 빈 배열
for day_index in range(n_ar_rows):                                     # 학습 날짜 0번째부터 299번째까지
    previous_day_values = history_and_train_solar[day_index]            # 이 학습일의 "바로 전날" 12시간 값
    ar_lag_features[day_index] = previous_day_values[::-1]               # 시간을 거꾸로 뒤집어서 저장

ar_design_matrix = np.hstack([ar_intercept_column, ar_lag_features])   # 절편 열 + lag 12개 열 (300, 13)


# =====================================================================
# 5. 시간대(h=0~11) 마다 "제안 모형 MILP" (논문 Eq.9) 를 풀어서 계수를 구함
#
#    핵심 아이디어: 회귀 계수를 "예측오차만" 보고 정하는 게 아니라,
#    "이 계수로 예측했을 때 발전계획에서 얼마나 손해보는지(경제손실)"와
#    "예측오차(MAE)"를 W1, W2 비율로 섞은 값을 최소화하도록 정한다.
#
#    이때 필요한 "오라클(사후 최적 이익)"은 학습용과 평가용을 다르게 쓴다:
#      - 학습(여기, denominator 계산용): {0, 실제발전량, 설비최대(1.0)} 3후보
#        (operational_corrected 원래 방식 - W1,W2 스케일을 맞추는 상수일 뿐)
#      - 평가(6번 이후, 최종 gap 계산용): {0, 실제발전량} 만 (논문 Eq.13 그대로)
#    두 오라클을 같은 걸로 통일하면 W1=1,W2=20 균형이 깨져 계수가
#    발산하는 문제가 실제로 있었다 (paper_proposed_block9.py 에서 확인됨).
# =====================================================================

n_train_obs = n_ar_rows          # 시간대 하나의 학습 표본 개수 (300개)
n_features = ar_design_matrix.shape[1]   # 입력 변수 개수 (절편 포함 13개)
coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))   # 시간대별 계수(13개)를 저장할 빈 배열

for hour in range(HOURS_PER_DAY):                       # 시간대 0부터 11까지 순서대로 하나씩 처리

    y_this_hour = train_solar[:, hour]                    # 이 시간대의 정답값(실제 발전량) 300개
    da_this_hour = train_da_price[:, hour]                  # 이 시간대의 학습용 DA가격 300개
    rt_this_hour = train_rt_price[:, hour]                  # 이 시간대의 학습용 RT가격 300개

    # ---- 5-1. 학습용(training) 오라클 이익 계산: {0, 실제발전량, 설비최대(1.0)} 3후보 ----
    oracle_profit_train = np.zeros(n_train_obs)               # 학습용 오라클 이익을 담을 빈 배열
    for i in range(n_train_obs):                                 # 300개 학습 표본을 하나씩 확인
        actual_i = y_this_hour[i]                                  # 이 표본의 실제 발전량
        da_i = da_this_hour[i]                                      # 이 표본의 DA가격
        rt_i = rt_this_hour[i]                                      # 이 표본의 RT가격
        penalty_i = PENALTY_RATE * da_i                              # 부족분 벌금단가

        profit_commit_0 = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
        # ↑ 약정 0: 실제발전량 전부를 RT가격에 판매 (부족/잉여 없음)

        profit_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
        # ↑ 약정 = 실제발전량: 전부 DA가격에 판매 (부족/잉여 없음)

        surplus_if_full = max(actual_i - 1.0, 0.0)                   # 약정 1.0일 때 잉여 (actual<=1 이라 보통 0)
        shortage_if_full = max(1.0 - actual_i, 0.0)                  # 약정 1.0일 때 부족량
        profit_commit_1 = CAPACITY_MW * DURATION_HOURS * (
            da_i * 1.0 + rt_i * surplus_if_full - penalty_i * shortage_if_full
        )                                                             # ↑ 약정 = 설비최대(1.0)일 때 이익

        oracle_profit_train[i] = max(profit_commit_0, profit_commit_actual, profit_commit_1)
        # ↑ 세 후보 중 가장 큰 값 = 이 표본의 학습용 오라클 이익

    # [수정] W1 항은 정규화(오라클 합으로 나누기) 복원, 단 scale(30)은 곱하지 않음
    training_denominator = 0.0                                 # 학습용 오라클 이익의 합계(정규화 상수)
    for i in range(n_train_obs):                                 # 300개를 하나씩 순서대로
        training_denominator = training_denominator + oracle_profit_train[i]   # 누적

    # ================================================================
    # 5-2. CVaR 학습에 사용할 부족비용
    # ================================================================

    # 아래 한 줄은 수익식에 맞는 코드 하나만 남긴다.

    # 원래 3항 수익식
    shortage_price_this_hour = (
        PENALTY_RATE * da_this_hour
    )

    # 4항 수익식에서는 위 줄을 지우고 아래 사용
    # shortage_price_this_hour = (
    #     PENALTY_RATE * da_this_hour
    #     + rt_this_hour
    # )

    # Zhang 수익식에서는 위 줄을 지우고 아래 사용
    # shortage_price_this_hour = (
    #     ZHANG_C * rt_this_hour
    # )


    # ================================================================
    # 5-3. 학습용 오라클을 기존 APEN과 동일한 스케일로 변환
    #
    # oracle_profit_train에는 30MW가 곱해져 있음.
    # 기존 APEN의 W1 profit 항에는 30이 곱해져 있지 않으므로
    # CVaR 학습에서도 30을 제거한 regret을 사용한다.
    # ================================================================

    profit_scale = (
        CAPACITY_MW
        * DURATION_HOURS
    )

    oracle_profit_train_unscaled = (
        oracle_profit_train
        / profit_scale
    )


    # ================================================================
    # 5-4. 기존 전체 regret 항의 목적함수 계수
    #
    # W2 * sum(abs error)
    # + W1 * sum(regret)
    # + W1 * lambda * N * CVaR(regret)
    # ================================================================

    surplus_cost = (
        -W1
        * rt_this_hour
        + W2
    )

    shortage_cost = (
        W1
        * shortage_price_this_hour
        + W2
    )


    # ================================================================
    # 5-5. Gurobi 모형 생성
    # ================================================================

    gmodel = gp.Model(
        f"proposed_ar_cvar_hour_{hour}"
    )

    gmodel.Params.OutputFlag = 0
    gmodel.Params.MIPGap = 1e-9

    beta_var = gmodel.addMVar(
        n_features,
        lb=-GRB.INFINITY,
        name="beta"
    )

    x_var = gmodel.addMVar(
        n_train_obs,
        lb=0.0,
        ub=1.0,
        name="x"
    )

    yplus_var = gmodel.addMVar(
        n_train_obs,
        lb=0.0,
        ub=1.0,
        name="y_plus"
    )

    yminus_var = gmodel.addMVar(
        n_train_obs,
        lb=0.0,
        ub=1.0,
        name="y_minus"
    )


    # 예측값 = X beta
    gmodel.addConstr(
        x_var
        - ar_design_matrix @ beta_var
        == 0.0,
        name="commitment_eq"
    )

    # x + surplus - shortage = 실제 발전량
    gmodel.addConstr(
        x_var
        + yplus_var
        - yminus_var
        == y_this_hour,
        name="mismatch_eq"
    )


    # ================================================================
    # 5-6. surplus와 shortage의 동시 발생 방지
    #
    # z=0: surplus만 허용
    # z=1: shortage만 허용
    # ================================================================

    z_var = gmodel.addMVar(
        n_train_obs,
        vtype=GRB.BINARY,
        name="surplus_shortage_indicator"
    )

    gmodel.addConstr(
        yplus_var + z_var <= 1.0,
        name="surplus_only"
    )

    gmodel.addConstr(
        yminus_var - z_var <= 0.0,
        name="shortage_only"
    )


    # ================================================================
    # 5-7. CVaR 선형화
    #
    # zeta: VaR_alpha 문턱값
    # s_i: 문턱을 초과하는 tail regret
    #
    # s_i >= regret_i - zeta
    # ================================================================

    zeta_var = gmodel.addVar(
        lb=0.0,
        name="cvar_zeta"
    )

    s_var = gmodel.addMVar(
        n_train_obs,
        lb=0.0,
        name="cvar_excess"
    )

    for i in range(n_train_obs):

        profit_expression_i = (
            da_this_hour[i] * x_var[i]
            + rt_this_hour[i] * yplus_var[i]
            - shortage_price_this_hour[i]
            * yminus_var[i]
        )

        regret_expression_i = (
            oracle_profit_train_unscaled[i]
            - profit_expression_i
        )

        gmodel.addConstr(
            s_var[i]
            >= regret_expression_i
            - zeta_var,
            name=f"cvar_tail_{i}"
        )


    # ================================================================
    # 5-8. 최종 additive CVaR 목적함수
    #
    # 기존 전체 regret 항을 그대로 유지하고 CVaR 항만 추가한다.
    # ================================================================

    total_regret_objective = (
        (-W1 * da_this_hour)
        @ x_var
        + surplus_cost @ yplus_var
        + shortage_cost @ yminus_var
    )

    cvar_objective = (
        W1
        * CVAR_LAMBDA
        * (
            n_train_obs * zeta_var
            + s_var.sum()
            / (1.0 - CVAR_ALPHA)
        )
    )

    objective_expression = (
        total_regret_objective
        + cvar_objective
    )

    gmodel.setObjective(
        objective_expression,
        GRB.MINIMIZE
    )

    gmodel.optimize()


    # ================================================================
    # 5-9. 최적화 상태 확인
    # ================================================================

    if gmodel.Status != GRB.OPTIMAL:

        raise RuntimeError(
            f"시간대 {hour}의 CVaR 학습 실패: "
            f"Gurobi status={gmodel.Status}"
        )

    coefficients_by_hour[hour] = (
        beta_var.X
    )

    print(
        f"  시간대 {hour} CVaR 학습 완료 "
        f"(alpha={CVAR_ALPHA}, "
        f"lambda={CVAR_LAMBDA}, "
        f"이진변수={n_train_obs})"
    )
# =====================================================================
# 6. 테스트 100일을 하루씩 순서대로 예측 (rolling one-day-ahead)
#    - 기본_모형_AR.py 와 완전히 동일한 절차
# =====================================================================

test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))     # 예측 결과를 담을 빈 배열 (100, 12)
previous_day_actual = train_solar[-1]                        # 테스트 첫날 예측에 쓸 "직전 하루" = 학습 마지막 날

for day_index in range(n_test_days):                          # 테스트 0번째 날부터 99번째 날까지 순서대로

    feature_vector = np.concatenate([[1.0], previous_day_actual[::-1]])
    # ↑ [절편 1] + [직전 하루 12시간을 거꾸로 뒤집은 값]

    for hour in range(HOURS_PER_DAY):                            # 이 날의 시간대 0~11을 하나씩 예측
        raw_prediction = np.dot(coefficients_by_hour[hour], feature_vector)  # 계수와 입력을 곱해서 더함
        clipped_prediction = min(max(raw_prediction, 0.0), 1.0)              # 예측값을 0~1 범위로 잘라냄
        test_forecast[day_index, hour] = clipped_prediction                    # 예측 결과 배열에 저장

    previous_day_actual = test_solar[day_index]                # 다음날 예측을 위해 "직전 하루"를 오늘의 실제값으로 갱신


# =====================================================================
# 7. nRMSE 계산 (Eq. 11-12) - 기본_모형_AR.py 와 완전히 동일한 절차
# =====================================================================

actual_flat = test_solar.flatten()          # (100,12) 실제값 표를 1200개짜리 한 줄로 펼침
predicted_flat = test_forecast.flatten()     # (100,12) 예측값 표도 1200개짜리 한 줄로 펼침

sum_of_squared_error = 0.0                   # 제곱오차를 누적할 변수
for i in range(len(actual_flat)):              # 1200개 값을 하나씩 순서대로
    error_i = actual_flat[i] - predicted_flat[i]   # 이 값의 오차(실제-예측)
    sum_of_squared_error = sum_of_squared_error + error_i * error_i   # 오차의 제곱을 누적

mean_squared_error = sum_of_squared_error / len(actual_flat)   # 누적한 제곱오차의 평균
rmse_value = mean_squared_error ** 0.5                           # 평균제곱오차의 제곱근 = RMSE

sum_of_actual = 0.0                          # 실제값 합계를 누적할 변수
for i in range(len(actual_flat)):              # 1200개 값을 하나씩 순서대로
    sum_of_actual = sum_of_actual + actual_flat[i]   # 실제값을 누적

average_actual = sum_of_actual / len(actual_flat)   # 실제 발전량의 평균값
nrmse_percent = 100.0 * rmse_value / average_actual   # RMSE를 평균으로 나누고 100을 곱해 %로 표현


# =====================================================================
# 8. optimality gap 계산 - [수정] 평가용 오라클 {0, S_t, 1} 3후보 사용 (첨부 문서의 보정)
#    (기본_모형_AR.py 와 완전히 동일한 절차 - 5번의 학습용 오라클과는
#     다른, 논문 Eq.13 그대로의 오라클을 씀)
# =====================================================================

da_flat = test_da_price.flatten()        # (100,12) DA가격 표를 1200개짜리 한 줄로 펼침
rt_flat = test_rt_price.flatten()        # (100,12) RT가격 표를 1200개짜리 한 줄로 펼침

sum_of_realized_profit = 0.0             # 실제(제안모형 예측 기반) 총 이익을 누적할 변수
sum_of_oracle_profit = 0.0               # 오라클(사후 최적) 총 이익을 누적할 변수

for i in range(len(actual_flat)):          # 테스트 1200개 관측치를 하나씩 순서대로 처리

    actual_i = actual_flat[i]                # 이 시간의 실제 발전량
    commitment_i = predicted_flat[i]         # 이 시간의 제안모형 예측값 = 일간전 약정량
    da_i = da_flat[i]                        # 이 시간의 DA 가격
    rt_i = rt_flat[i]                        # 이 시간의 RT 가격
    penalty_cost_i = PENALTY_RATE * da_i     # 이 시간의 부족분 벌금단가

    mismatch_i = actual_i - commitment_i       # 실제 - 약정
    surplus_i = max(mismatch_i, 0.0)             # 잉여량
    shortage_i = max(-mismatch_i, 0.0)           # 부족량
    realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
        da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
    )                                             # 논문 Eq.(1a) 3항 이익함수
    sum_of_realized_profit = sum_of_realized_profit + realized_profit_i

    profit_if_commit_zero = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)     # 약정 0일 때 이익
    if actual_i <= 1.0:                                                          # S_t가 허용범위(<=1) 안이면
        profit_if_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)   # 약정=실제발전량일 때 이익
    else:                                                                         # S_t가 설비최대(1)를 넘는 경우엔
        profit_if_commit_actual = -np.inf                                          # x=S_t는 허용범위 밖이라 후보에서 제외
    surplus_if_full = max(actual_i - 1.0, 0.0)                                    # 약정 1.0일 때 잉여
    shortage_if_full = max(1.0 - actual_i, 0.0)                                   # 약정 1.0일 때 부족량
    profit_if_commit_full = CAPACITY_MW * DURATION_HOURS * (
        da_i * 1.0 + rt_i * surplus_if_full - penalty_cost_i * shortage_if_full
    )                                                                              # 약정=설비최대(1.0)일 때 이익
    oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual, profit_if_commit_full)
    # ↑ {0, S_t, 1} 3후보 중 최댓값 (수정 후 보정 오라클)
    sum_of_oracle_profit = sum_of_oracle_profit + oracle_profit_i

optimality_gap_percent = 100.0 * (sum_of_oracle_profit - sum_of_realized_profit) / sum_of_oracle_profit


# =====================================================================
# 9. 결과 출력
# =====================================================================

print()
print("=== 논문 제안 모형 AR (W1: raw·×30없음·정규화없음 / W2: 절댓값 합 / 오라클 3후보 보정) - z03 블록18(120일) 결과 ===")
print(f"nRMSE          = {nrmse_percent:.6f} %   (논문: {PAPER_NRMSE:.2f} %)")
print(f"optimality gap = {optimality_gap_percent:.6f} %   (논문: {PAPER_GAP:.2f} %)")
print()
print("| 모델 | 논문 nRMSE | 이번 구현 nRMSE | 논문 optimality gap | 이번 구현 optimality gap | Δ nRMSE | Δ optimality gap |")
print("|---|---:|---:|---:|---:|---:|---:|")
print(
    f"| 논문 제안 모형 AR | {PAPER_NRMSE:.2f}% | {nrmse_percent:.6f}% | {PAPER_GAP:.2f}% | {optimality_gap_percent:.6f}% "
    f"| {nrmse_percent - PAPER_NRMSE:+.6f}%p | {optimality_gap_percent - PAPER_GAP:+.6f}%p |"
)


# =====================================================================
# 시간대별 예측오차 + 경제적 regret 통합 분석
# =====================================================================

import matplotlib.pyplot as plt

# Baseline AR에서 사용
#MODEL_LABEL = "Baseline AR"
#OUTPUT_TAG = "baseline_ar"

# Proposed AR에서는 위 두 줄 대신 아래 두 줄 사용
MODEL_LABEL = "Proposed AR"
OUTPUT_TAG = "proposed_ar"


# =====================================================================
# 1. 시간대별 예측오차 계산
# =====================================================================

# test_forecast와 test_solar의 모양: 테스트 일수 × 12시간
forecast_error = test_forecast - test_solar

# 날짜 × 시간대별 절대오차
# 발전량이 0~1이므로 100을 곱하면 설비용량 대비 오차(%)
absolute_error_capacity_percent = (
    np.abs(forecast_error) * 100.0
)

# 시간대별 RMSE
hourly_rmse_capacity_percent = (
    np.sqrt(
        np.mean(
            forecast_error ** 2,
            axis=0
        )
    )
    * 100.0
)

# 시간대별 MAE
hourly_mae_capacity_percent = (
    np.mean(
        np.abs(forecast_error),
        axis=0
    )
    * 100.0
)

# 시간대별 예측 편향
# 양수: 과대예측
# 음수: 과소예측
hourly_bias_capacity_percent = (
    np.mean(
        forecast_error,
        axis=0
    )
    * 100.0
)


# =====================================================================
# 2. 시점별 실제 수익, 오라클 수익, regret 계산
# =====================================================================

economic_realized_profit = np.zeros(
    (n_test_days, HOURS_PER_DAY)
)

economic_oracle_profit = np.zeros(
    (n_test_days, HOURS_PER_DAY)
)

economic_regret = np.zeros(
    (n_test_days, HOURS_PER_DAY)
)


for day_index in range(n_test_days):

    for hour_index in range(HOURS_PER_DAY):

        actual_i = test_solar[
            day_index,
            hour_index
        ]

        commitment_i = test_forecast[
            day_index,
            hour_index
        ]

        da_i = test_da_price[
            day_index,
            hour_index
        ]

        rt_i = test_rt_price[
            day_index,
            hour_index
        ]

        penalty_cost_i = PENALTY_RATE * da_i

        # -------------------------------------------------------------
        # 2-1. 모형 예측값을 약정량으로 사용했을 때 실제 수익
        # -------------------------------------------------------------

        mismatch_i = actual_i - commitment_i

        surplus_i = max(
            mismatch_i,
            0.0
        )

        shortage_i = max(
            -mismatch_i,
            0.0
        )

        realized_profit_i = (
            CAPACITY_MW
            * DURATION_HOURS
            * (
                da_i * commitment_i
                + rt_i * surplus_i
                - penalty_cost_i * shortage_i
            )
        )

        # -------------------------------------------------------------
        # 2-2. 오라클 후보 1: x=0
        # -------------------------------------------------------------

        profit_commit_zero = (
            CAPACITY_MW
            * DURATION_HOURS
            * rt_i
            * actual_i
        )

        # -------------------------------------------------------------
        # 2-3. 오라클 후보 2: x=S
        # -------------------------------------------------------------

        if actual_i <= 1.0:

            profit_commit_actual = (
                CAPACITY_MW
                * DURATION_HOURS
                * da_i
                * actual_i
            )

        else:

            # actual_i가 1보다 크면 x=S는 허용범위 밖
            profit_commit_actual = -np.inf

        # -------------------------------------------------------------
        # 2-4. 오라클 후보 3: x=1
        # -------------------------------------------------------------

        surplus_if_full = max(
            actual_i - 1.0,
            0.0
        )

        shortage_if_full = max(
            1.0 - actual_i,
            0.0
        )

        profit_commit_full = (
            CAPACITY_MW
            * DURATION_HOURS
            * (
                da_i
                + rt_i * surplus_if_full
                - penalty_cost_i * shortage_if_full
            )
        )

        # -------------------------------------------------------------
        # 2-5. 세 후보 중 최대 수익
        # -------------------------------------------------------------

        oracle_profit_i = max(
            profit_commit_zero,
            profit_commit_actual,
            profit_commit_full
        )

        regret_i = (
            oracle_profit_i
            - realized_profit_i
        )

        economic_realized_profit[
            day_index,
            hour_index
        ] = realized_profit_i

        economic_oracle_profit[
            day_index,
            hour_index
        ] = oracle_profit_i

        economic_regret[
            day_index,
            hour_index
        ] = regret_i


# =====================================================================
# 3. regret 검증
# =====================================================================

minimum_regret = np.min(
    economic_regret
)

print()
print(f"=== {MODEL_LABEL}: regret 검증 ===")
print(
    f"최소 시점별 regret = "
    f"{minimum_regret:.12f}"
)

# 오라클이 제대로 계산됐다면 의미 있는 음수 regret은 없어야 함
if minimum_regret < -1e-8:

    raise ValueError(
        "음수 regret이 발견됐습니다. "
        "오라클 또는 수익 계산을 확인해야 합니다."
    )

# 부동소수점 오차 수준의 작은 값은 0으로 처리
economic_regret[
    np.abs(economic_regret) < 1e-8
] = 0.0


# =====================================================================
# 4. 시간대별 경제적 지표 계산
# =====================================================================

local_hours = np.arange(
    LOCAL_HOUR_START,
    LOCAL_HOUR_START + HOURS_PER_DAY
)

# 시간대별 평균 regret
mean_regret_by_hour = np.mean(
    economic_regret,
    axis=0
)

# 시간대별 regret의 90% 분위수
p90_regret_by_hour = np.percentile(
    economic_regret,
    90,
    axis=0
)

# 시간대별 최대 regret
max_regret_by_hour = np.max(
    economic_regret,
    axis=0
)

# =====================================================================
# 시간대별·전체 테스트 CVaR 계산
# =====================================================================

# 시간대별 테스트 표본은 30일
# alpha=0.9이면 각 시간대의 최악 3일 평균
hourly_tail_count = max(
    1,
    int(
        np.ceil(
            (1.0 - CVAR_ALPHA)
            * n_test_days
        )
    )
)

sorted_regret_by_hour = np.sort(
    economic_regret,
    axis=0
)

cvar_regret_by_hour = np.mean(
    sorted_regret_by_hour[
        -hourly_tail_count:,
        :
    ],
    axis=0
)

# 전체 360개 시점 중 최악 10%
regret_flat = economic_regret.flatten()

overall_tail_count = max(
    1,
    int(
        np.ceil(
            (1.0 - CVAR_ALPHA)
            * len(regret_flat)
        )
    )
)

sorted_regret_flat = np.sort(
    regret_flat
)

overall_cvar_regret = np.mean(
    sorted_regret_flat[
        -overall_tail_count:
    ]
)

overall_max_regret = np.max(
    regret_flat
)

print()
print(
    f"테스트 CVaR{CVAR_ALPHA:.0%} regret = "
    f"{overall_cvar_regret:.6f}"
)

print(
    f"테스트 최대 regret = "
    f"{overall_max_regret:.6f}"
)

# 시간대별 regret 합계와 오라클 수익 합계
regret_sum_by_hour = np.sum(
    economic_regret,
    axis=0
)

oracle_profit_sum_by_hour = np.sum(
    economic_oracle_profit,
    axis=0
)

# 시간대별 optimality gap
hourly_gap_percent = np.full(
    HOURS_PER_DAY,
    np.nan
)

for hour_index in range(HOURS_PER_DAY):

    if abs(
        oracle_profit_sum_by_hour[hour_index]
    ) > 1e-12:

        hourly_gap_percent[hour_index] = (
            100.0
            * regret_sum_by_hour[hour_index]
            / oracle_profit_sum_by_hour[hour_index]
        )


# =====================================================================
# 5. 전체 지표 재검증
# =====================================================================

overall_nrmse_from_hourly = (
    100.0
    * np.sqrt(
        np.mean(
            (hourly_rmse_capacity_percent / 100.0) ** 2
        )
    )
    / np.mean(test_solar)
)

overall_gap_from_regret = (
    100.0
    * np.sum(economic_regret)
    / np.sum(economic_oracle_profit)
)

print()
print(f"=== {MODEL_LABEL}: 전체 지표 검증 ===")

print(
    f"기존 nRMSE                  = "
    f"{nrmse_percent:.6f}%"
)

print(
    f"시간대별 RMSE로 복원한 nRMSE = "
    f"{overall_nrmse_from_hourly:.6f}%"
)

print(
    f"기존 optimality gap         = "
    f"{optimality_gap_percent:.6f}%"
)

print(
    f"시점별 regret으로 재계산     = "
    f"{overall_gap_from_regret:.6f}%"
)


# =====================================================================
# 6. 시간대별 통합 결과표
# =====================================================================

hourly_diagnostics_table = pd.DataFrame({
    "local_hour":
        local_hours,

    "RMSE_capacity_percent":
        hourly_rmse_capacity_percent,

    "MAE_capacity_percent":
        hourly_mae_capacity_percent,

    "bias_capacity_percent":
        hourly_bias_capacity_percent,

    "mean_economic_regret":
        mean_regret_by_hour,

    "P90_economic_regret":
        p90_regret_by_hour,

    "CVaR90_economic_regret":
        cvar_regret_by_hour,

    "max_economic_regret":
        max_regret_by_hour,

    "hourly_optimality_gap_percent":
        hourly_gap_percent,
})

print()
print(
    f"=== {MODEL_LABEL}: "
    f"시간대별 예측오차와 경제적 regret ==="
)

print(
    hourly_diagnostics_table
    .round(6)
    .to_string(index=False)
)

hourly_csv_name = (
    f"{OUTPUT_TAG}_hourly_diagnostics.csv"
)

hourly_diagnostics_table.to_csv(
    hourly_csv_name,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "시간대별 결과 저장 완료:",
    hourly_csv_name
)


# =====================================================================
# 7. 두 AR 모형의 통합 비교를 위한 결과 저장
# =====================================================================

date_labels = []

for one_date in test_dates_sorted:

    date_labels.append(
        pd.Timestamp(one_date).strftime("%m-%d")
    )

diagnostics_npz_name = (
    f"{OUTPUT_TAG}_combined_diagnostics.npz"
)

np.savez(
    f"{OUTPUT_TAG}_combined_diagnostics.npz",

    # 테스트 실제 발전량과 예측값
    actual=test_solar,
    prediction=test_forecast,

    # 시점별 절대예측오차
    absolute_error_capacity_percent=
        absolute_error_capacity_percent,

    # 시간대별 RMSE
    hourly_rmse_capacity_percent=
        hourly_rmse_capacity_percent,

    # 시점별 실제 수익
    realized_profit=
        economic_realized_profit,

    # 시점별 오라클 수익
    oracle_profit=
        economic_oracle_profit,

    # 시점별 경제적 regret
    regret=
        economic_regret,

    # 시간대별 평균 regret
    mean_regret_by_hour=
        mean_regret_by_hour,

    # 시간대별 P90 regret
    p90_regret_by_hour=
        p90_regret_by_hour,

    # 시간대별 CVaR regret
    cvar_regret_by_hour=
        cvar_regret_by_hour,

    # 시간대별 최대 regret
    max_regret_by_hour=
        max_regret_by_hour,

    # 시간대별 optimality gap
    hourly_gap_percent=
        hourly_gap_percent,

    # 테스트 날짜와 시간대
    dates=np.array(date_labels),
    hours=local_hours,

    # 전체 nRMSE
    overall_nrmse=np.array(
        nrmse_percent
    ),

    # 전체 optimality gap
    overall_gap=np.array(
        optimality_gap_percent
    ),

    # 테스트 전체 평균 regret
    overall_mean_regret=np.array(
        np.mean(economic_regret)
    ),

    # 테스트 전체 CVaR regret
    overall_cvar_regret=np.array(
        overall_cvar_regret
    ),

    # 테스트 전체 최대 regret
    overall_max_regret=np.array(
        overall_max_regret
    ),

    # CVaR 설정값
    cvar_alpha=np.array(
        CVAR_ALPHA
    ),

    cvar_lambda=np.array(
        CVAR_LAMBDA
    ),

    # 시간대별·전체 CVaR 계산에 포함된 tail 표본 수
    hourly_tail_count=np.array(
        hourly_tail_count
    ),

    overall_tail_count=np.array(
        overall_tail_count
    )
)

print(
    "통합 결과 저장 완료:",
    f"{OUTPUT_TAG}_combined_diagnostics.npz"
)
