# -*- coding: utf-8 -*-

# =====================================================================
# 방법7_zhang_MLR_gurobi_시간별지표.py
#
# 목적: "방법7 - Additive CVaR" (보고서 3.2.7절 수식)을 Zhang(2024) 비대칭가격(rho_plus=c*RT) 수익식
#       위에 얹은 MLR(다중선형회귀) 모형의 nRMSE·optimality gap과, 시간대별/
#       경제적 regret 지표까지 구한다.
#
# MLR 이 AR 과 다른 점: MLR 은 시간대별로 12번 반복하지 않고, 학습 표본
# 3600개(300일 x 12시간) 전체를 하나의 MILP로 한 번에 푼다 - 계수(beta)
# 하나만 나오고, 그 계수를 모든 시간대에 똑같이 쓴다.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
#
# 데이터: merged_for_simulation_z03.csv (Zone 3)
# 구간(z03 블록18): 학습 2013-08-25~2013-11-22(300일 x 12시간=3600행),
#                  테스트 2013-11-23~2013-12-22(100일 x 12시간=1200행)
# 가중치: W1=1, W2=20, 벌금비용률=50% (논문 Table 4 비교 지점)
# CVaR: alpha=0.90, lambda=0.05 (보고서 5.7.3절에서 정한 대표값)
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
import gurobipy as gp                        # Gurobi 최적화 라이브러리
from gurobipy import GRB                     # Gurobi 상수


# =====================================================================
# 0. 설정값
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 이 파이썬 파일이 있는 폴더(구현결과_보고서_지원)
MERGED_FILE = os.environ.get("MERGED_FILE", os.path.join(BASE_DIR, "merged_for_simulation_z03.csv"))  # 데이터 파일 경로 (환경변수로 덮어쓰기 가능)

LOCAL_HOUR_START = 9             # 낮 시간대 시작 시(local_hour 기준)
LOCAL_HOUR_END = 21              # 낮 시간대 끝(이 값 미만까지, 즉 9~20시)

TRAIN_START = pd.Timestamp(os.environ.get("TRAIN_START", "2013-08-25"))   # 학습 시작일
TRAIN_END = pd.Timestamp(os.environ.get("TRAIN_END",   "2013-11-22"))     # 학습 마지막일 (300일째)
TEST_START = pd.Timestamp(os.environ.get("TEST_START",  "2013-11-23"))    # 테스트 시작일
TEST_END = pd.Timestamp(os.environ.get("TEST_END",    "2013-12-22"))      # 테스트 마지막일 (100일째)

CAPACITY_MW = 30.0                # 태양광 패널 설비 최대 용량 (논문 가정)
DURATION_HOURS = 1.0              # 한 시간대의 길이(시간)
ZHANG_C = float(os.environ.get("ZHANG_C", "1.5"))   # [Zhang et al. 방식] 부족가격 rho_plus = ZHANG_C * RT

import os
W1 = float(os.environ.get("SWEEP_W1","1.0"))
W2 = float(os.environ.get("SWEEP_W2","20.0"))

PAPER_NRMSE = 21.92                # 논문 Table 4, 논문 제안 모형 MLR 의 nRMSE(%) - 비교용
PAPER_GAP = 11.91                  # 논문 Table 4, 논문 제안 모형 MLR 의 optimality gap(%) - 비교용

CVAR_ALPHA = 0.90

# additive CVaR 가중치: 기존 전체 regret 항은 그대로 유지한다.
CVAR_LAMBDA = 0.05  # [정정] 원본 파일엔 0.00으로 저장돼 있었음 - AR 시간별지표·AR/MLR figure
# 6개 파일이 전부 0.05를 쓰고 보고서(3.2.7·5.7절)도 0.05를 대표값으로 쓰므로 통일함

if not (0.0 < CVAR_ALPHA < 1.0):
    raise ValueError("CVAR_ALPHA는 0과 1 사이여야 합니다.")

if CVAR_LAMBDA < 0.0:
    raise ValueError("CVAR_LAMBDA는 0 이상이어야 합니다.")


# =====================================================================
# 1. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)                       # csv 파일 전체를 한 번에 읽어옴
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])   # local_date 열을 날짜 타입으로 변환

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
# ↑ local_hour 가 9시 이상, 21시 미만(=9~20시)인 행만 True 인 판단 열을 만듦

daylight_table = raw_table[is_daylight].copy()              # 낮 시간대 행만 골라서 새 표로 복사
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START
# ↑ local_hour(9~20) 를 0~11 로 다시 번호 매김 (hour_idx)


# =====================================================================
# 2. 학습(train) / 테스트(test) 구간으로 자르고, 날짜·시간대 순서로 정렬
# =====================================================================
is_train_date = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train_date].copy()                    # 학습 구간 행만 골라냄
train_rows = train_rows.sort_values(["local_date", "hour_idx"])       # 날짜, 시간대 순서로 정렬

is_test_date = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test_date].copy()                      # 테스트 구간 행만 골라냄
test_rows = test_rows.sort_values(["local_date", "hour_idx"])         # 날짜, 시간대 순서로 정렬

print("학습 구간:", TRAIN_START.date(), "~", TRAIN_END.date(), "행 수:", len(train_rows))  # 3600 이어야 정상
print("테스트 구간:", TEST_START.date(), "~", TEST_END.date(), "행 수:", len(test_rows))    # 1200 이어야 정상


# =====================================================================
# 3. 학습/테스트용 1차원 배열 만들기 (행 하나 = 관측치 하나)
#    - 이번엔 학습용 DA/RT 가격도 필요함 (제안 모형은 학습 때 가격도
#      손실함수에 쓰기 때문)
# =====================================================================

n_train_obs = len(train_rows)                    # 학습 관측치 개수 (3600 이어야 정상)
train_solar = np.zeros(n_train_obs)                # 학습용 실제 발전량을 담을 빈 배열
train_dssrd = np.zeros(n_train_obs)                # 학습용 dSSRD 값을 담을 빈 배열
train_dtsr = np.zeros(n_train_obs)                 # 학습용 dTSR 값을 담을 빈 배열
train_hour = np.zeros(n_train_obs)                 # 학습용 Hour(0~11) 값을 담을 빈 배열
train_da_price = np.zeros(n_train_obs)             # 학습용 DA가격을 담을 빈 배열
train_rt_price = np.zeros(n_train_obs)             # 학습용 RT가격을 담을 빈 배열
row_counter = 0                                      # train_rows 를 순서대로 셀 카운터
for _, one_row in train_rows.iterrows():               # train_rows 를 한 줄씩 순서대로 확인
    train_solar[row_counter] = one_row["solar_power"]    # 실제 발전량 값을 채워 넣음
    train_dssrd[row_counter] = one_row["dssrd"]           # dSSRD 값을 채워 넣음
    train_dtsr[row_counter] = one_row["dtsr"]              # dTSR 값을 채워 넣음
    train_hour[row_counter] = one_row["hour_idx"]           # Hour(0~11) 값을 채워 넣음
    train_da_price[row_counter] = one_row["da_price"]        # DA가격 값을 채워 넣음
    train_rt_price[row_counter] = one_row["rt_price"]         # RT가격 값을 채워 넣음
    row_counter = row_counter + 1                                # 카운터를 하나 증가시킴

n_test_obs = len(test_rows)                       # 테스트 관측치 개수 (1200 이어야 정상)
test_solar = np.zeros(n_test_obs)                   # 테스트용 실제 발전량을 담을 빈 배열
test_dssrd = np.zeros(n_test_obs)                   # 테스트용 dSSRD 값을 담을 빈 배열
test_dtsr = np.zeros(n_test_obs)                    # 테스트용 dTSR 값을 담을 빈 배열
test_hour = np.zeros(n_test_obs)                    # 테스트용 Hour(0~11) 값을 담을 빈 배열
test_da_price = np.zeros(n_test_obs)                # 테스트용 DA가격을 담을 빈 배열
test_rt_price = np.zeros(n_test_obs)                # 테스트용 RT가격을 담을 빈 배열
row_counter = 0                                       # test_rows 를 순서대로 셀 카운터
for _, one_row in test_rows.iterrows():                # test_rows 를 한 줄씩 순서대로 확인
    test_solar[row_counter] = one_row["solar_power"]     # 실제 발전량 값을 채워 넣음
    test_dssrd[row_counter] = one_row["dssrd"]            # dSSRD 값을 채워 넣음
    test_dtsr[row_counter] = one_row["dtsr"]               # dTSR 값을 채워 넣음
    test_hour[row_counter] = one_row["hour_idx"]            # Hour(0~11) 값을 채워 넣음
    test_da_price[row_counter] = one_row["da_price"]         # DA가격 값을 채워 넣음
    test_rt_price[row_counter] = one_row["rt_price"]          # RT가격 값을 채워 넣음
    row_counter = row_counter + 1                                # 카운터를 하나 증가시킴


# =====================================================================
# 4. 회귀 입력행렬(X) 만들기 - 논문 Eq.(6): 절편 + dSSRD + dTSR + Hour
# =====================================================================

n_features = 4                                          # 절편, dSSRD, dTSR, Hour = 4개 입력변수

X_train = np.zeros((n_train_obs, n_features))              # 학습용 입력행렬 (3600, 4)
for i in range(n_train_obs):                                 # 3600개 학습 행을 하나씩 순서대로
    X_train[i, 0] = 1.0                                        # 첫 열은 절편용 1
    X_train[i, 1] = train_dssrd[i]                              # 둘째 열은 dSSRD
    X_train[i, 2] = train_dtsr[i]                                # 셋째 열은 dTSR
    X_train[i, 3] = train_hour[i]                                 # 넷째 열은 Hour(0~11)

X_test = np.zeros((n_test_obs, n_features))                # 테스트용 입력행렬 (1200, 4)
for i in range(n_test_obs):                                   # 1200개 테스트 행을 하나씩 순서대로
    X_test[i, 0] = 1.0                                          # 첫 열은 절편용 1
    X_test[i, 1] = test_dssrd[i]                                 # 둘째 열은 dSSRD
    X_test[i, 2] = test_dtsr[i]                                   # 셋째 열은 dTSR
    X_test[i, 3] = test_hour[i]                                    # 넷째 열은 Hour(0~11)


# =====================================================================
# 5. "논문 제안 모형 MILP" (Eq.10) 를 딱 한 번 풀어서 계수 4개를 구함
#    - AR 과 달리 시간대별 반복 없이, 3600개 표본 전체로 MILP 한 번만 품
#    - 학습용 오라클({0,실제발전량,설비최대 1.0} 3후보)로 W1/W2 정규화
#      상수를 잡고, 평가는 나중에 {0,S} 오라클로 따로 함 (AR 코드와 동일한 이유)
# =====================================================================

# ---- 5-1. 학습용(training) 오라클 이익 계산: {0, 실제발전량, 설비최대(1.0)} 3후보 ----
oracle_profit_train = np.zeros(n_train_obs)               # 학습용 오라클 이익을 담을 빈 배열
for i in range(n_train_obs):                                 # 3600개 학습 표본을 하나씩 확인
    actual_i = train_solar[i]                                  # 이 표본의 실제 발전량
    da_i = train_da_price[i]                                    # 이 표본의 DA가격
    rt_i = train_rt_price[i]                                    # 이 표본의 RT가격
    penalty_i = ZHANG_C * rt_i                              # [Zhang et al.] rho_plus = c*RT

    profit_commit_0 = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)              # 약정 0
    profit_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)          # 약정=실제발전량

    surplus_if_full = max(actual_i - 1.0, 0.0)                   # 약정 1.0일 때 잉여
    shortage_if_full = max(1.0 - actual_i, 0.0)                  # 약정 1.0일 때 부족량
    profit_commit_1 = CAPACITY_MW * DURATION_HOURS * (
        da_i * 1.0 + rt_i * surplus_if_full - penalty_i * shortage_if_full
    )                                                             # 약정 = 설비최대(1.0)

    oracle_profit_train[i] = max(profit_commit_0, profit_commit_actual, profit_commit_1)
    # ↑ 세 후보 중 가장 큰 값 = 이 표본의 학습용 오라클 이익

# ---- 5-2. additive CVaR 목적함수에 사용할 항 ----
shortage_price_train = ZHANG_C * train_rt_price

profit_scale = CAPACITY_MW * DURATION_HOURS
oracle_profit_train_unscaled = oracle_profit_train / profit_scale

# 기존 전체 regret 항의 가중치는 줄이지 않는다.
surplus_cost = (-W1 * train_rt_price) + W2
shortage_cost = (W1 * shortage_price_train) + W2

# ---- 5-3. Gurobi 모형 ----
gmodel = gp.Model("proposed_mlr_additive_cvar")
gmodel.Params.OutputFlag = 0
gmodel.Params.MIPGap = 1e-9

beta_var = gmodel.addMVar(n_features, lb=-GRB.INFINITY, name="beta")
x_var = gmodel.addMVar(n_train_obs, lb=0.0, ub=1.0, name="x")
yplus_var = gmodel.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_plus")
yminus_var = gmodel.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_minus")

gmodel.addConstr(x_var - X_train @ beta_var == 0.0, name="commitment_eq")
gmodel.addConstr(
    x_var + yplus_var - yminus_var == train_solar,
    name="mismatch_eq"
)

# CVaR 수익식에서도 surplus와 shortage가 동시에 발생하지 않게 한다.
z_var = gmodel.addMVar(
    n_train_obs,
    vtype=GRB.BINARY,
    name="surplus_shortage_indicator"
)
gmodel.addConstr(yplus_var + z_var <= 1.0, name="surplus_only")
gmodel.addConstr(yminus_var - z_var <= 0.0, name="shortage_only")

# ---- 5-4. CVaR 선형화 ----
zeta_var = gmodel.addVar(lb=0.0, name="cvar_zeta")
s_var = gmodel.addMVar(n_train_obs, lb=0.0, name="cvar_excess")

for i in range(n_train_obs):
    profit_expression_i = (
        train_da_price[i] * x_var[i]
        + train_rt_price[i] * yplus_var[i]
        - shortage_price_train[i] * yminus_var[i]
    )
    regret_expression_i = (
        oracle_profit_train_unscaled[i]
        - profit_expression_i
    )
    gmodel.addConstr(
        s_var[i] >= regret_expression_i - zeta_var,
        name=f"cvar_tail_{i}"
    )

# ---- 5-5. 기존 전체 regret + additive CVaR ----
total_regret_objective = (
    (-W1 * train_da_price) @ x_var
    + surplus_cost @ yplus_var
    + shortage_cost @ yminus_var
)

cvar_objective = (
    W1
    * CVAR_LAMBDA
    * (
        n_train_obs * zeta_var
        + s_var.sum() / (1.0 - CVAR_ALPHA)
    )
)

gmodel.setObjective(
    total_regret_objective + cvar_objective,
    GRB.MINIMIZE
)
gmodel.optimize()

if gmodel.Status != GRB.OPTIMAL:
    raise RuntimeError(
        "Proposed MLR additive CVaR 학습 실패: "
        f"Gurobi status={gmodel.Status}"
    )

mlr_coefficients = beta_var.X
print(
    f"MLR additive CVaR 학습 완료 "
    f"(alpha={CVAR_ALPHA}, lambda={CVAR_LAMBDA}, "
    f"이진변수={n_train_obs}), 계수: {mlr_coefficients}"
)


# =====================================================================
# 6. 테스트 구간 예측 (한 번에 전체 예측)
# =====================================================================

test_forecast = np.zeros(n_test_obs)                        # 예측 결과를 담을 빈 배열
for i in range(n_test_obs):                                    # 테스트 1200개 행을 하나씩 순서대로
    raw_prediction = np.dot(mlr_coefficients, X_test[i])          # 계수와 입력을 곱해서 더함
    clipped_prediction = min(max(raw_prediction, 0.0), 1.0)         # 예측값을 0~1 범위로 잘라냄
    test_forecast[i] = clipped_prediction                            # 예측 결과 배열에 저장


# =====================================================================
# 7. nRMSE 계산 (Eq. 11-12)
# =====================================================================

sum_of_squared_error = 0.0                   # 제곱오차를 누적할 변수
for i in range(n_test_obs):                    # 1200개 값을 하나씩 순서대로
    error_i = test_solar[i] - test_forecast[i]   # 이 값의 오차(실제-예측)
    sum_of_squared_error = sum_of_squared_error + error_i * error_i   # 오차의 제곱을 누적

mean_squared_error = sum_of_squared_error / n_test_obs   # 누적한 제곱오차의 평균
rmse_value = mean_squared_error ** 0.5                     # 평균제곱오차의 제곱근 = RMSE

sum_of_actual = 0.0                          # 실제값 합계를 누적할 변수
for i in range(n_test_obs):                    # 1200개 값을 하나씩 순서대로
    sum_of_actual = sum_of_actual + test_solar[i]   # 실제값을 누적

average_actual = sum_of_actual / n_test_obs   # 실제 발전량의 평균값
nrmse_percent = 100.0 * rmse_value / average_actual   # RMSE를 평균으로 나누고 100을 곱해 %로 표현


# =====================================================================
# 8. optimality gap 계산 - 평가용 오라클 {0, 실제발전량} 만 사용
#    (5번의 학습용 오라클과는 다른, 논문 Eq.13 그대로의 오라클을 씀)
# =====================================================================

sum_of_realized_profit = 0.0             # 실제(제안모형 예측 기반) 총 이익을 누적할 변수
sum_of_oracle_profit = 0.0               # 오라클(사후 최적) 총 이익을 누적할 변수
test_regret = np.zeros(n_test_obs)       # 테스트 표본별 economic regret

for i in range(n_test_obs):                # 테스트 1200개 관측치를 하나씩 순서대로 처리

    actual_i = test_solar[i]                 # 이 시간의 실제 발전량
    commitment_i = test_forecast[i]          # 이 시간의 제안모형 예측값 = 일간전 약정량
    da_i = test_da_price[i]                  # 이 시간의 DA 가격
    rt_i = test_rt_price[i]                  # 이 시간의 RT 가격
    penalty_cost_i = ZHANG_C * rt_i     # [Zhang et al.] rho_plus = c*RT

    mismatch_i = actual_i - commitment_i       # 실제 - 약정
    surplus_i = max(mismatch_i, 0.0)             # 잉여량
    shortage_i = max(-mismatch_i, 0.0)           # 부족량
    realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
        da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
    )                                             # 논문 Eq.(1a) 3항 이익함수
    sum_of_realized_profit = sum_of_realized_profit + realized_profit_i

    profit_if_commit_zero = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)     # 약정 0일 때 이익
    if actual_i <= 1.0:
        profit_if_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
    else:
        profit_if_commit_actual = -np.inf
    surplus_if_full = max(actual_i - 1.0, 0.0)
    shortage_if_full = max(1.0 - actual_i, 0.0)
    profit_if_commit_full = CAPACITY_MW * DURATION_HOURS * (
        da_i * 1.0 + rt_i * surplus_if_full - penalty_cost_i * shortage_if_full
    )
    oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual, profit_if_commit_full)  # [수정] 3후보 보정
    sum_of_oracle_profit = sum_of_oracle_profit + oracle_profit_i
    test_regret[i] = oracle_profit_i - realized_profit_i

optimality_gap_percent = 100.0 * (sum_of_oracle_profit - sum_of_realized_profit) / sum_of_oracle_profit

test_tail_count = max(
    1,
    int(np.ceil((1.0 - CVAR_ALPHA) * n_test_obs))
)
sorted_test_regret = np.sort(test_regret)
overall_mean_regret = np.mean(test_regret)
overall_cvar_regret = np.mean(sorted_test_regret[-test_tail_count:])
overall_max_regret = np.max(test_regret)


# =====================================================================
# 9. 결과 출력
# =====================================================================

print()
print("=== 논문 제안 모형 MLR (Eq.10 MILP, W1=1, W2=20) - z03 블록18 결과 ===")
print(f"nRMSE          = {nrmse_percent:.6f} %   (논문: {PAPER_NRMSE:.2f} %)")
print(f"optimality gap = {optimality_gap_percent:.6f} %   (논문: {PAPER_GAP:.2f} %)")
print(f"mean regret    = {overall_mean_regret:.6f}")
print(f"CVaR{CVAR_ALPHA:.0%} regret = {overall_cvar_regret:.6f}")
print(f"maximum regret = {overall_max_regret:.6f}")
print()
print("| 모델 | 논문 nRMSE | 이번 구현 nRMSE | 논문 optimality gap | 이번 구현 optimality gap | Δ nRMSE | Δ optimality gap |")
print("|---|---:|---:|---:|---:|---:|---:|")
print(
    f"| 논문 제안 모형 MLR | {PAPER_NRMSE:.2f}% | {nrmse_percent:.6f}% | {PAPER_GAP:.2f}% | {optimality_gap_percent:.6f}% "
    f"| {nrmse_percent - PAPER_NRMSE:+.6f}%p | {optimality_gap_percent - PAPER_GAP:+.6f}%p |"
)


# =====================================================================
# 시간대별 예측오차 + 경제적 regret 통합 분석 (AR 스크립트와 동일한 틀)
# =====================================================================

MODEL_LABEL = "Proposed MLR (Zhang)"
OUTPUT_TAG = "proposed_mlr_zhang"


# =====================================================================
# 0. 테스트 구간을 (일수 x 12시간) 모양으로 재구성
#    - test_rows가 날짜,시간대 순서로 정렬되어 있으므로 그대로 reshape
#      가능하다 (AR 스크립트의 test_solar/test_forecast와 같은 모양)
# =====================================================================

HOURS_PER_DAY = 12                                   # 하루 낮 시간대 개수
n_test_days = n_test_obs // HOURS_PER_DAY              # 테스트 날짜 수 (100 이어야 정상)

test_solar_2d = test_solar.reshape(n_test_days, HOURS_PER_DAY)         # 실제 발전량 (일수, 12)
test_forecast_2d = test_forecast.reshape(n_test_days, HOURS_PER_DAY)     # 예측 발전량 (일수, 12)
test_da_price_2d = test_da_price.reshape(n_test_days, HOURS_PER_DAY)      # DA가격 (일수, 12)
test_rt_price_2d = test_rt_price.reshape(n_test_days, HOURS_PER_DAY)       # RT가격 (일수, 12)

test_dates_sorted = sorted(test_rows["local_date"].unique())          # 테스트 날짜 목록(오래된 순)


# =====================================================================
# 1. 시간대별 예측오차 계산
# =====================================================================

forecast_error = test_forecast_2d - test_solar_2d

absolute_error_capacity_percent = (
    np.abs(forecast_error) * 100.0
)

hourly_rmse_capacity_percent = (
    np.sqrt(
        np.mean(
            forecast_error ** 2,
            axis=0
        )
    )
    * 100.0
)

hourly_mae_capacity_percent = (
    np.mean(
        np.abs(forecast_error),
        axis=0
    )
    * 100.0
)

# 양수: 과대예측, 음수: 과소예측
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

        actual_i = test_solar_2d[day_index, hour_index]
        commitment_i = test_forecast_2d[day_index, hour_index]
        da_i = test_da_price_2d[day_index, hour_index]
        rt_i = test_rt_price_2d[day_index, hour_index]
        penalty_cost_i = ZHANG_C * rt_i

        # -------------------------------------------------------------
        # 2-1. 모형 예측값을 약정량으로 사용했을 때 실제 수익
        # -------------------------------------------------------------

        mismatch_i = actual_i - commitment_i
        surplus_i = max(mismatch_i, 0.0)
        shortage_i = max(-mismatch_i, 0.0)

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

        surplus_if_full = max(actual_i - 1.0, 0.0)
        shortage_if_full = max(1.0 - actual_i, 0.0)

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

        economic_realized_profit[day_index, hour_index] = realized_profit_i
        economic_oracle_profit[day_index, hour_index] = oracle_profit_i
        economic_regret[day_index, hour_index] = regret_i


# =====================================================================
# 3. regret 검증
# =====================================================================

minimum_regret = np.min(economic_regret)

print()
print(f"=== {MODEL_LABEL}: regret 검증 ===")
print(f"최소 시점별 regret = {minimum_regret:.12f}")

# 오라클이 제대로 계산됐다면 의미 있는 음수 regret은 없어야 함
if minimum_regret < -1e-8:
    raise ValueError(
        "음수 regret이 발견됐습니다. "
        "오라클 또는 수익 계산을 확인해야 합니다."
    )

# 부동소수점 오차 수준의 작은 값은 0으로 처리
economic_regret[np.abs(economic_regret) < 1e-8] = 0.0


# =====================================================================
# 4. 시간대별 경제적 지표 계산
# =====================================================================

local_hours = np.arange(
    LOCAL_HOUR_START,
    LOCAL_HOUR_START + HOURS_PER_DAY
)

# 시간대별 평균 regret
mean_regret_by_hour = np.mean(economic_regret, axis=0)

# 시간대별 regret의 90% 분위수
p90_regret_by_hour = np.percentile(economic_regret, 90, axis=0)

# 시간대별 최대 regret
max_regret_by_hour = np.max(economic_regret, axis=0)


# =====================================================================
# 시간대별·전체 테스트 CVaR 계산
# =====================================================================

# 시간대별 테스트 표본은 n_test_days일
# alpha=0.9이면 각 시간대의 최악 10% 평균
hourly_tail_count = max(
    1,
    int(np.ceil((1.0 - CVAR_ALPHA) * n_test_days))
)

sorted_regret_by_hour = np.sort(economic_regret, axis=0)

cvar_regret_by_hour = np.mean(
    sorted_regret_by_hour[-hourly_tail_count:, :],
    axis=0
)

# 전체 시점 중 최악 10%
regret_flat = economic_regret.flatten()

overall_tail_count = max(
    1,
    int(np.ceil((1.0 - CVAR_ALPHA) * len(regret_flat)))
)

sorted_regret_flat = np.sort(regret_flat)

overall_cvar_regret = np.mean(sorted_regret_flat[-overall_tail_count:])
overall_max_regret = np.max(regret_flat)

print()
print(f"테스트 CVaR{CVAR_ALPHA:.0%} regret = {overall_cvar_regret:.6f}")
print(f"테스트 최대 regret = {overall_max_regret:.6f}")

# 시간대별 regret 합계와 오라클 수익 합계
regret_sum_by_hour = np.sum(economic_regret, axis=0)
oracle_profit_sum_by_hour = np.sum(economic_oracle_profit, axis=0)

# 시간대별 optimality gap
hourly_gap_percent = np.full(HOURS_PER_DAY, np.nan)

for hour_index in range(HOURS_PER_DAY):
    if abs(oracle_profit_sum_by_hour[hour_index]) > 1e-12:
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
    / np.mean(test_solar_2d)
)

overall_gap_from_regret = (
    100.0
    * np.sum(economic_regret)
    / np.sum(economic_oracle_profit)
)

print()
print(f"=== {MODEL_LABEL}: 전체 지표 검증 ===")
print(f"기존 nRMSE                  = {nrmse_percent:.6f}%")
print(f"시간대별 RMSE로 복원한 nRMSE = {overall_nrmse_from_hourly:.6f}%")
print(f"기존 optimality gap         = {optimality_gap_percent:.6f}%")
print(f"시점별 regret으로 재계산     = {overall_gap_from_regret:.6f}%")


# =====================================================================
# 6. 시간대별 통합 결과표
# =====================================================================

hourly_diagnostics_table = pd.DataFrame({
    "local_hour": local_hours,
    "RMSE_capacity_percent": hourly_rmse_capacity_percent,
    "MAE_capacity_percent": hourly_mae_capacity_percent,
    "bias_capacity_percent": hourly_bias_capacity_percent,
    "mean_economic_regret": mean_regret_by_hour,
    "P90_economic_regret": p90_regret_by_hour,
    "CVaR90_economic_regret": cvar_regret_by_hour,
    "max_economic_regret": max_regret_by_hour,
    "hourly_optimality_gap_percent": hourly_gap_percent,
})

print()
print(f"=== {MODEL_LABEL}: 시간대별 예측오차와 경제적 regret ===")
print(hourly_diagnostics_table.round(6).to_string(index=False))

hourly_csv_name = f"{OUTPUT_TAG}_hourly_diagnostics.csv"

hourly_diagnostics_table.to_csv(
    hourly_csv_name,
    index=False,
    encoding="utf-8-sig"
)

print()
print("시간대별 결과 저장 완료:", hourly_csv_name)


# =====================================================================
# 7. 통합 비교를 위한 결과 저장 (AR 스크립트와 같은 npz 구조)
# =====================================================================

date_labels = []
for one_date in test_dates_sorted:
    date_labels.append(pd.Timestamp(one_date).strftime("%m-%d"))

diagnostics_npz_name = f"{OUTPUT_TAG}_combined_diagnostics.npz"

np.savez(
    diagnostics_npz_name,

    # 테스트 실제 발전량과 예측값
    actual=test_solar_2d,
    prediction=test_forecast_2d,

    # 시점별 절대예측오차
    absolute_error_capacity_percent=absolute_error_capacity_percent,

    # 시간대별 RMSE
    hourly_rmse_capacity_percent=hourly_rmse_capacity_percent,

    # 시점별 실제 수익
    realized_profit=economic_realized_profit,

    # 시점별 오라클 수익
    oracle_profit=economic_oracle_profit,

    # 시점별 경제적 regret
    regret=economic_regret,

    # 시간대별 평균 regret
    mean_regret_by_hour=mean_regret_by_hour,

    # 시간대별 P90 regret
    p90_regret_by_hour=p90_regret_by_hour,

    # 시간대별 CVaR regret
    cvar_regret_by_hour=cvar_regret_by_hour,

    # 시간대별 최대 regret
    max_regret_by_hour=max_regret_by_hour,

    # 시간대별 optimality gap
    hourly_gap_percent=hourly_gap_percent,

    # 테스트 날짜와 시간대
    dates=np.array(date_labels),
    hours=local_hours,

    # 전체 nRMSE
    overall_nrmse=np.array(nrmse_percent),

    # 전체 optimality gap
    overall_gap=np.array(optimality_gap_percent),

    # 테스트 전체 평균 regret
    overall_mean_regret=np.array(np.mean(economic_regret)),

    # 테스트 전체 CVaR regret
    overall_cvar_regret=np.array(overall_cvar_regret),

    # 테스트 전체 최대 regret
    overall_max_regret=np.array(overall_max_regret),

    # CVaR 설정값
    cvar_alpha=np.array(CVAR_ALPHA),
    cvar_lambda=np.array(CVAR_LAMBDA),

    # 시간대별·전체 CVaR 계산에 포함된 tail 표본 수
    hourly_tail_count=np.array(hourly_tail_count),
    overall_tail_count=np.array(overall_tail_count)
)

print("통합 결과 저장 완료:", diagnostics_npz_name)
