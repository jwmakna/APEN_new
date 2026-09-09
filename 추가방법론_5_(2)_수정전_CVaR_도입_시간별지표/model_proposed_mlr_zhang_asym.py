# -*- coding: utf-8 -*-

# =====================================================================
# 논문_제안_모형_MLR.py
#
# 목적: "논문 제안 모형 MLR" (논문 Eq.10 - 발전계획 최적화를 MLR 학습에
#       통합한 모형) 의 nRMSE 와 optimality gap 을 구해서
#       D:\03_JiWon\APEN\Readme.md 결과표의 "논문 제안 모형 MLR" 행처럼
#       뽑는다. 기본_모형_MLR.py 와 입력/평가 방식은 같고, 회귀를 푸는
#       방법만 "MILP(혼합정수계획법)" 로 바뀐다.
#
# MLR 이 AR 과 다른 점(논문_제안_모형.py 참고): MLR 은 시간대별로 12번
# 반복하지 않고, 학습 표본 3600개(300일 x 12시간) 전체를 하나의 MILP로
# 한 번에 푼다 - 계수(beta) 하나만 나오고, 그 계수를 모든 시간대에
# 똑같이 쓴다.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
#
# 데이터: merged_for_simulation_z03.csv (Zone1)
# 구간(블록9): 학습 2012-11-28~2013-09-23(300일 x 12시간=3600행),
#             테스트 2013-09-24~2014-01-01(100일 x 12시간=1200행)
# 가중치: W1=1, W2=20, 벌금비용률=50% (논문 Table 4 비교 지점)
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

training_denominator = 0.0                                 # 학습용 오라클 이익의 합계(정규화 상수)
for i in range(n_train_obs):                                 # 3600개를 하나씩 순서대로
    training_denominator = training_denominator + oracle_profit_train[i]   # 누적

# ---- 5-2. 목적함수 계수 계산 (잉여/부족 각각에 대한 비용) ----
scale = CAPACITY_MW * DURATION_HOURS                        # 이익 스케일 상수 (30)
surplus_cost = np.zeros(n_train_obs)                          # 잉여(y_plus) 1단위당 목적함수 계수
shortage_cost = np.zeros(n_train_obs)                          # 부족(y_minus) 1단위당 목적함수 계수
for i in range(n_train_obs):                                    # 3600개를 하나씩 순서대로
    penalty_i = ZHANG_C * train_rt_price[i]                  # [Zhang et al.] rho_plus = c*RT
    surplus_cost[i] = (-W1 * train_rt_price[i]) + W2   # [수정] 정규화 전부 제거 - 논문 Eq.(10a) 문자 그대로
    shortage_cost[i] = (W1 * penalty_i) + W2           # [수정] 정규화 전부 제거 - 논문 Eq.(10a) 문자 그대로

binary_row_list = []                                          # 이진변수가 필요한 표본 번호를 담을 빈 리스트
for i in range(n_train_obs):                                    # 3600개를 하나씩 순서대로 확인
    if surplus_cost[i] + shortage_cost[i] < 0.0:                  # 두 비용의 합이 음수면
        binary_row_list.append(i)                                    # 이 표본 번호를 이진변수 목록에 추가
binary_rows = np.array(binary_row_list, dtype=int)             # 리스트를 numpy 배열로 변환
n_binary = len(binary_rows)                                    # 이진변수 개수
print("이진변수가 필요한 표본 개수:", n_binary, "/", n_train_obs)   # 몇 개나 필요했는지 출력

# ---- 5-3~5-8. Gurobi로 MILP 구성 및 풀기 ----
gmodel = gp.Model("proposed_mlr")
gmodel.Params.OutputFlag = 0
gmodel.Params.MIPGap = 1e-9

beta_var = gmodel.addMVar(n_features, lb=-GRB.INFINITY, name="beta")
x_var = gmodel.addMVar(n_train_obs, lb=0.0, ub=1.0, name="x")
yplus_var = gmodel.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_plus")
yminus_var = gmodel.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_minus")

gmodel.addConstr(x_var - X_train @ beta_var == 0.0, name="commitment_eq")           # x = X@beta
gmodel.addConstr(x_var + yplus_var - yminus_var == train_solar, name="mismatch_eq")  # x+y_plus-y_minus=actual

if n_binary > 0:
    z_var = gmodel.addMVar(n_binary, vtype=GRB.BINARY, name="z")
    gmodel.addConstr(yplus_var[binary_rows] + z_var <= 1.0, name="complementarity_plus")
    gmodel.addConstr(yminus_var[binary_rows] - z_var <= 0.0, name="complementarity_minus")

objective_expr = (
    (-W1 * train_da_price) @ x_var   # [수정] 정규화 제거
    + surplus_cost @ yplus_var
    + shortage_cost @ yminus_var
)
gmodel.setObjective(objective_expr, GRB.MINIMIZE)
gmodel.optimize()

mlr_coefficients = beta_var.X                                       # beta(4개)만 뽑아 저장
print(f"MLR MILP 완료 (성공 여부: {gmodel.Status == GRB.OPTIMAL}), 계수: {mlr_coefficients}")


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

optimality_gap_percent = 100.0 * (sum_of_oracle_profit - sum_of_realized_profit) / sum_of_oracle_profit


# =====================================================================
# 9. 결과 출력
# =====================================================================

print()
print("=== 논문 제안 모형 MLR (Eq.10 MILP, W1=1, W2=20) - 블록9 결과 ===")
print(f"nRMSE          = {nrmse_percent:.6f} %   (논문: {PAPER_NRMSE:.2f} %)")
print(f"optimality gap = {optimality_gap_percent:.6f} %   (논문: {PAPER_GAP:.2f} %)")
print()
print("| 모델 | 논문 nRMSE | 이번 구현 nRMSE | 논문 optimality gap | 이번 구현 optimality gap | Δ nRMSE | Δ optimality gap |")
print("|---|---:|---:|---:|---:|---:|---:|")
print(
    f"| 논문 제안 모형 MLR | {PAPER_NRMSE:.2f}% | {nrmse_percent:.6f}% | {PAPER_GAP:.2f}% | {optimality_gap_percent:.6f}% "
    f"| {nrmse_percent - PAPER_NRMSE:+.6f}%p | {optimality_gap_percent - PAPER_GAP:+.6f}%p |"
)




# =====================================================================
# 10. Table 4 스윕 - 0/1(기존 재현) + W1/W2 비율 10개 (Zhang 비대칭)
# =====================================================================

RATIO_LIST = [
    ("0/1", 0.0, 1.0),
    ("1/20", 1.0, 20.0), ("1/10", 1.0, 10.0), ("1/5", 1.0, 5.0),
    ("1/2", 1.0, 2.0), ("1/1", 1.0, 1.0), ("2/1", 2.0, 1.0),
    ("5/1", 5.0, 1.0), ("10/1", 10.0, 1.0), ("20/1", 20.0, 1.0),
    ("1/0", 1.0, 0.0),
]

sweep_labels = []
sweep_nrmse = []
sweep_gap = []

for ratio_label, sweep_w1, sweep_w2 in RATIO_LIST:

    oracle_profit_train_sw = np.zeros(n_train_obs)
    for i in range(n_train_obs):
        actual_i = train_solar[i]
        da_i = train_da_price[i]
        rt_i = train_rt_price[i]
        penalty_i = ZHANG_C * rt_i
        profit_commit_0 = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
        profit_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
        surplus_if_full = max(actual_i - 1.0, 0.0)
        shortage_if_full = max(1.0 - actual_i, 0.0)
        profit_commit_1 = CAPACITY_MW * DURATION_HOURS * (
            da_i * 1.0 + rt_i * surplus_if_full - penalty_i * shortage_if_full
        )
        oracle_profit_train_sw[i] = max(profit_commit_0, profit_commit_actual, profit_commit_1)

    surplus_cost_sw = np.zeros(n_train_obs)
    shortage_cost_sw = np.zeros(n_train_obs)
    for i in range(n_train_obs):
        penalty_i = ZHANG_C * train_rt_price[i]
        surplus_cost_sw[i] = (-sweep_w1 * train_rt_price[i]) + sweep_w2
        shortage_cost_sw[i] = (sweep_w1 * penalty_i) + sweep_w2

    binary_row_list_sw = []
    for i in range(n_train_obs):
        if surplus_cost_sw[i] + shortage_cost_sw[i] < 0.0:
            binary_row_list_sw.append(i)
    binary_rows_sw = np.array(binary_row_list_sw, dtype=int)
    n_binary_sw = len(binary_rows_sw)

    gmodel_sw = gp.Model(f"proposed_mlr_sweep_{ratio_label}")
    gmodel_sw.Params.OutputFlag = 0
    gmodel_sw.Params.MIPGap = 1e-9

    beta_var_sw = gmodel_sw.addMVar(n_features, lb=-GRB.INFINITY, name="beta")
    x_var_sw = gmodel_sw.addMVar(n_train_obs, lb=0.0, ub=1.0, name="x")
    yplus_var_sw = gmodel_sw.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_plus")
    yminus_var_sw = gmodel_sw.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_minus")

    gmodel_sw.addConstr(x_var_sw - X_train @ beta_var_sw == 0.0)
    gmodel_sw.addConstr(x_var_sw + yplus_var_sw - yminus_var_sw == train_solar)

    if n_binary_sw > 0:
        z_var_sw = gmodel_sw.addMVar(n_binary_sw, vtype=GRB.BINARY, name="z")
        gmodel_sw.addConstr(yplus_var_sw[binary_rows_sw] + z_var_sw <= 1.0)
        gmodel_sw.addConstr(yminus_var_sw[binary_rows_sw] - z_var_sw <= 0.0)

    objective_expr_sw = (
        (-sweep_w1 * train_da_price) @ x_var_sw
        + surplus_cost_sw @ yplus_var_sw
        + shortage_cost_sw @ yminus_var_sw
    )
    gmodel_sw.setObjective(objective_expr_sw, GRB.MINIMIZE)
    gmodel_sw.optimize()

    mlr_coefficients_sw = beta_var_sw.X

    test_forecast_sw = np.zeros(n_test_obs)
    for i in range(n_test_obs):
        raw_prediction = np.dot(mlr_coefficients_sw, X_test[i])
        test_forecast_sw[i] = min(max(raw_prediction, 0.0), 1.0)

    sse_sw = 0.0
    for i in range(n_test_obs):
        e = test_solar[i] - test_forecast_sw[i]
        sse_sw = sse_sw + e * e
    rmse_sw = (sse_sw / n_test_obs) ** 0.5
    avg_actual_sw = 0.0
    for i in range(n_test_obs):
        avg_actual_sw = avg_actual_sw + test_solar[i]
    avg_actual_sw = avg_actual_sw / n_test_obs
    nrmse_sw = 100.0 * rmse_sw / avg_actual_sw

    sum_realized_sw = 0.0
    sum_oracle_sw = 0.0
    for i in range(n_test_obs):
        actual_i = test_solar[i]
        commitment_i = test_forecast_sw[i]
        da_i = test_da_price[i]
        rt_i = test_rt_price[i]
        penalty_cost_i = ZHANG_C * rt_i
        mismatch_i = actual_i - commitment_i
        surplus_i = max(mismatch_i, 0.0)
        shortage_i = max(-mismatch_i, 0.0)
        realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
            da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
        )
        sum_realized_sw = sum_realized_sw + realized_profit_i

        profit_if_commit_zero = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
        if actual_i <= 1.0:
            profit_if_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
        else:
            profit_if_commit_actual = -np.inf
        surplus_if_full = max(actual_i - 1.0, 0.0)
        shortage_if_full = max(1.0 - actual_i, 0.0)
        profit_if_commit_full = CAPACITY_MW * DURATION_HOURS * (
            da_i * 1.0 + rt_i * surplus_if_full - penalty_cost_i * shortage_if_full
        )
        oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual, profit_if_commit_full)
        sum_oracle_sw = sum_oracle_sw + oracle_profit_i

    gap_sw = 100.0 * (sum_oracle_sw - sum_realized_sw) / sum_oracle_sw

    sweep_labels.append(ratio_label)
    sweep_nrmse.append(nrmse_sw)
    sweep_gap.append(gap_sw)
    print(f"[{ratio_label:>5}] nRMSE={nrmse_sw:.4f}%  gap={gap_sw:.4f}%")

print()
print("| W1/W2 | nRMSE | optimality gap |")
print("|---|---:|---:|")
for lbl, nr, gp_ in zip(sweep_labels, sweep_nrmse, sweep_gap):
    print(f"| {lbl} | {nr:.4f}% | {gp_:.4f}% |")
    

# =====================================================================
# 11. Figure 6 재현 - nRMSE(왼쪽 축)·optimality gap(오른쪽 축)을
#     W1/W2 비율별로 그린다 (원본 3항 profit function 버전)
# =====================================================================
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

chart_labels = ["MLR" if lbl == "0/1" else lbl for lbl in sweep_labels]

paper_nrmse_list = [21.76, 21.92, 21.84, 21.75, 21.66, 22.01, 23.32, 27.75, 30.62, 35.76, 37.67]
paper_gap_list = [12.59, 11.91, 11.68, 11.15, 10.65, 10.28, 9.91, 9.51, 9.32, 9.27, 9.28]

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

ax1.plot(chart_labels, sweep_nrmse, color="#F97316", marker="o", label="nRMSE (우리 결과)")
ax1.plot(chart_labels, paper_nrmse_list, color="#FDBA74", marker="o", linestyle="--", label="nRMSE (논문 Table4)")
ax2.plot(chart_labels, sweep_gap, color="#3B82F6", marker="s", label="gap (우리 결과)")
ax2.plot(chart_labels, paper_gap_list, color="#93C5FD", marker="s", linestyle="--", label="gap (논문 Table4)")

ax1.set_xlabel("W1/W2")
ax1.set_ylabel("nRMSE (%)", color="#F97316")
ax2.set_ylabel("Optimality Gap (%)", color="#3B82F6")
ax1.tick_params(axis="y", labelcolor="#F97316")
ax2.tick_params(axis="y", labelcolor="#3B82F6")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.title("Fig.6 Zhang 비대칭(정규화X) - z03 블록18")
plt.tight_layout()
plt.savefig("fig6_mlr_zhang_asym.png", dpi=150)
print("저장 완료: fig6_mlr_zhang_asym.png")
plt.show()


# =====================================================================
# 12. Figure 8 재현 - (a) nRMSE, (b) optimality gap 을 ZHANG_C(1.0~2.0)
#     별로 그린다. W1=W2=1 고정 (Zhang 비대칭 버전)
# =====================================================================
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ZHANG_C_LIST = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
FIG8_W1 = 1.0
FIG8_W2 = 1.0

surplus_cost_f8 = np.full(n_train_obs, 1.0)
shortage_cost_f8 = np.full(n_train_obs, 1.0)

gmodel_f8 = gp.Model("mlr_baseline_fig8")
gmodel_f8.Params.OutputFlag = 0
gmodel_f8.Params.MIPGap = 1e-9

beta_var_f8 = gmodel_f8.addMVar(n_features, lb=-GRB.INFINITY, name="beta")
x_var_f8 = gmodel_f8.addMVar(n_train_obs, lb=0.0, ub=1.0, name="x")
yplus_var_f8 = gmodel_f8.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_plus")
yminus_var_f8 = gmodel_f8.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_minus")

gmodel_f8.addConstr(x_var_f8 - X_train @ beta_var_f8 == 0.0)
gmodel_f8.addConstr(x_var_f8 + yplus_var_f8 - yminus_var_f8 == train_solar)

objective_expr_f8 = surplus_cost_f8 @ yplus_var_f8 + shortage_cost_f8 @ yminus_var_f8
gmodel_f8.setObjective(objective_expr_f8, GRB.MINIMIZE)
gmodel_f8.optimize()

mlr_coefficients_f8 = beta_var_f8.X

mlr_forecast_f8 = np.zeros(n_test_obs)
for i in range(n_test_obs):
    raw_prediction = np.dot(mlr_coefficients_f8, X_test[i])
    mlr_forecast_f8[i] = min(max(raw_prediction, 0.0), 1.0)

sse_mlr_f8 = 0.0
for i in range(n_test_obs):
    e = test_solar[i] - mlr_forecast_f8[i]
    sse_mlr_f8 = sse_mlr_f8 + e * e
avg_actual_f8 = 0.0
for i in range(n_test_obs):
    avg_actual_f8 = avg_actual_f8 + test_solar[i]
avg_actual_f8 = avg_actual_f8 / n_test_obs
mlr_nrmse_f8 = 100.0 * ((sse_mlr_f8 / n_test_obs) ** 0.5) / avg_actual_f8

mlr_nrmse_list = []
mlr_gap_list = []
proposed_nrmse_list = []
proposed_gap_list = []

for zhang_c_sw in ZHANG_C_LIST:

    sum_realized_mlr = 0.0
    sum_oracle_mlr = 0.0
    for i in range(n_test_obs):
        actual_i = test_solar[i]
        commitment_i = mlr_forecast_f8[i]
        da_i = test_da_price[i]
        rt_i = test_rt_price[i]
        penalty_cost_i = zhang_c_sw * rt_i
        mismatch_i = actual_i - commitment_i
        surplus_i = max(mismatch_i, 0.0)
        shortage_i = max(-mismatch_i, 0.0)
        realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
            da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
        )
        sum_realized_mlr = sum_realized_mlr + realized_profit_i

        profit_if_commit_zero = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
        if actual_i <= 1.0:
            profit_if_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
        else:
            profit_if_commit_actual = -np.inf
        surplus_if_full = max(actual_i - 1.0, 0.0)
        shortage_if_full = max(1.0 - actual_i, 0.0)
        profit_if_commit_full = CAPACITY_MW * DURATION_HOURS * (
            da_i * 1.0 + rt_i * surplus_if_full - penalty_cost_i * shortage_if_full
        )
        oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual, profit_if_commit_full)
        sum_oracle_mlr = sum_oracle_mlr + oracle_profit_i

    gap_mlr_sw = 100.0 * (sum_oracle_mlr - sum_realized_mlr) / sum_oracle_mlr
    mlr_nrmse_list.append(mlr_nrmse_f8)
    mlr_gap_list.append(gap_mlr_sw)

    surplus_cost_p8 = np.zeros(n_train_obs)
    shortage_cost_p8 = np.zeros(n_train_obs)
    for i in range(n_train_obs):
        penalty_i = zhang_c_sw * train_rt_price[i]
        surplus_cost_p8[i] = (-FIG8_W1 * train_rt_price[i]) + FIG8_W2
        shortage_cost_p8[i] = (FIG8_W1 * penalty_i) + FIG8_W2

    binary_row_list_p8 = []
    for i in range(n_train_obs):
        if surplus_cost_p8[i] + shortage_cost_p8[i] < 0.0:
            binary_row_list_p8.append(i)
    binary_rows_p8 = np.array(binary_row_list_p8, dtype=int)
    n_binary_p8 = len(binary_rows_p8)

    gmodel_p8 = gp.Model(f"proposed_mlr_fig8_{zhang_c_sw}")
    gmodel_p8.Params.OutputFlag = 0
    gmodel_p8.Params.MIPGap = 1e-9

    beta_var_p8 = gmodel_p8.addMVar(n_features, lb=-GRB.INFINITY, name="beta")
    x_var_p8 = gmodel_p8.addMVar(n_train_obs, lb=0.0, ub=1.0, name="x")
    yplus_var_p8 = gmodel_p8.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_plus")
    yminus_var_p8 = gmodel_p8.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_minus")

    gmodel_p8.addConstr(x_var_p8 - X_train @ beta_var_p8 == 0.0)
    gmodel_p8.addConstr(x_var_p8 + yplus_var_p8 - yminus_var_p8 == train_solar)

    if n_binary_p8 > 0:
        z_var_p8 = gmodel_p8.addMVar(n_binary_p8, vtype=GRB.BINARY, name="z")
        gmodel_p8.addConstr(yplus_var_p8[binary_rows_p8] + z_var_p8 <= 1.0)
        gmodel_p8.addConstr(yminus_var_p8[binary_rows_p8] - z_var_p8 <= 0.0)

    objective_expr_p8 = (
        (-FIG8_W1 * train_da_price) @ x_var_p8
        + surplus_cost_p8 @ yplus_var_p8
        + shortage_cost_p8 @ yminus_var_p8
    )
    gmodel_p8.setObjective(objective_expr_p8, GRB.MINIMIZE)
    gmodel_p8.optimize()

    mlr_coefficients_p8 = beta_var_p8.X

    proposed_forecast_p8 = np.zeros(n_test_obs)
    for i in range(n_test_obs):
        raw_prediction = np.dot(mlr_coefficients_p8, X_test[i])
        proposed_forecast_p8[i] = min(max(raw_prediction, 0.0), 1.0)

    sse_p8 = 0.0
    for i in range(n_test_obs):
        e = test_solar[i] - proposed_forecast_p8[i]
        sse_p8 = sse_p8 + e * e
    nrmse_p8 = 100.0 * ((sse_p8 / n_test_obs) ** 0.5) / avg_actual_f8

    sum_realized_p8 = 0.0
    sum_oracle_p8 = 0.0
    for i in range(n_test_obs):
        actual_i = test_solar[i]
        commitment_i = proposed_forecast_p8[i]
        da_i = test_da_price[i]
        rt_i = test_rt_price[i]
        penalty_cost_i = zhang_c_sw * rt_i
        mismatch_i = actual_i - commitment_i
        surplus_i = max(mismatch_i, 0.0)
        shortage_i = max(-mismatch_i, 0.0)
        realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
            da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
        )
        sum_realized_p8 = sum_realized_p8 + realized_profit_i

        profit_if_commit_zero = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
        if actual_i <= 1.0:
            profit_if_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
        else:
            profit_if_commit_actual = -np.inf
        surplus_if_full = max(actual_i - 1.0, 0.0)
        shortage_if_full = max(1.0 - actual_i, 0.0)
        profit_if_commit_full = CAPACITY_MW * DURATION_HOURS * (
            da_i * 1.0 + rt_i * surplus_if_full - penalty_cost_i * shortage_if_full
        )
        oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual, profit_if_commit_full)
        sum_oracle_p8 = sum_oracle_p8 + oracle_profit_i

    gap_p8 = 100.0 * (sum_oracle_p8 - sum_realized_p8) / sum_oracle_p8

    proposed_nrmse_list.append(nrmse_p8)
    proposed_gap_list.append(gap_p8)
    print(f"[C={zhang_c_sw:.1f}] MLR nRMSE={mlr_nrmse_f8:.4f}% gap={gap_mlr_sw:.4f}% | Proposed nRMSE={nrmse_p8:.4f}% gap={gap_p8:.4f}%")

zhang_c_labels = [f"{c:.1f}" for c in ZHANG_C_LIST]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
ax_a.plot(zhang_c_labels, mlr_nrmse_list, color="#3B82F6", marker="o", label="MLR")
ax_a.plot(zhang_c_labels, proposed_nrmse_list, color="#F97316", marker="s", label="Proposed Model")
ax_a.set_xlabel("ZHANG_C")
ax_a.set_ylabel("nRMSE (%)")
ax_a.set_title("(a) nRMSE")
ax_a.legend()

ax_b.plot(zhang_c_labels, mlr_gap_list, color="#3B82F6", marker="o", label="MLR")
ax_b.plot(zhang_c_labels, proposed_gap_list, color="#F97316", marker="s", label="Proposed Model")
ax_b.set_xlabel("ZHANG_C")
ax_b.set_ylabel("Optimality Gap (%)")
ax_b.set_title("(b) Optimality Gap")
ax_b.legend()

plt.suptitle("Fig.8 Zhang 비대칭(정규화X) - z03 블록18 (W1=W2=1)")
plt.tight_layout()
plt.savefig("fig8_mlr_zhang_asym.png", dpi=150)
print("저장 완료: fig8_mlr_zhang_asym.png")
plt.show()