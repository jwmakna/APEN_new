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
BASE_DIR = os.path.abspath(__file__)           # 이 파이썬 파일이 있는 폴더
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
ZHANG_C = float(os.environ.get("ZHANG_C", "1.5"))   # [Zhang et al. 방식] 부족가격 rho_plus = ZHANG_C * RT

W1 = float(os.environ.get("SWEEP_W1", "1.0"))    # 정규화 파라미터 W1 (환경변수로 스윕 가능)
W2 = float(os.environ.get("SWEEP_W2", "20.0"))    # 정규화 파라미터 W2 (환경변수로 스윕 가능)

PAPER_NRMSE = 34.89                # 논문 Table 3, 논문 제안 모형 AR 의 nRMSE(%) - 비교용
PAPER_GAP = 13.91                  # 논문 Table 3, 논문 제안 모형 AR 의 optimality gap(%) - 비교용


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
        penalty_i = ZHANG_C * rt_i                              # [Zhang et al.] rho_plus = c*RT

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

    # ---- 5-2. 목적함수 계수 계산 (잉여/부족 각각에 대한 비용) ----
    # [수정] W2 항은 정규화(n_train_obs로 나누기) 없이 "합" 그대로 사용
    surplus_cost = np.zeros(n_train_obs)                          # 잉여(y_plus) 1단위당 목적함수 계수
    shortage_cost = np.zeros(n_train_obs)                          # 부족(y_minus) 1단위당 목적함수 계수
    for i in range(n_train_obs):                                    # 300개를 하나씩 순서대로
        penalty_i = ZHANG_C * rt_this_hour[i]                    # [Zhang et al.] rho_plus = c*RT
        surplus_cost[i] = (-W1 * rt_this_hour[i]) + W2   # [수정] 정규화 전부 제거 - 논문 Eq.(9a) 문자 그대로
        shortage_cost[i] = (W1 * penalty_i) + W2         # [수정] 정규화 전부 제거 - 논문 Eq.(9a) 문자 그대로

    binary_row_list = []                                          # 이진변수가 필요한 표본 번호를 담을 빈 리스트
    for i in range(n_train_obs):                                    # 300개를 하나씩 순서대로 확인
        if surplus_cost[i] + shortage_cost[i] < 0.0:                  # 두 비용의 합이 음수면(둘 다 켜는게 유리해지면)
            binary_row_list.append(i)                                    # 이 표본 번호를 이진변수 목록에 추가
    binary_rows = np.array(binary_row_list, dtype=int)             # 리스트를 numpy 배열로 변환
    n_binary = len(binary_rows)                                    # 이진변수 개수

    # ---- 5-3~5-8. Gurobi로 MILP 구성 및 풀기 ----
    gmodel = gp.Model(f"proposed_ar_hour_{hour}")
    gmodel.Params.OutputFlag = 0
    gmodel.Params.MIPGap = 1e-9                                     # scipy의 mip_rel_gap과 동일하게 맞춤

    beta_var = gmodel.addMVar(n_features, lb=-GRB.INFINITY, name="beta")
    x_var = gmodel.addMVar(n_train_obs, lb=0.0, ub=1.0, name="x")
    yplus_var = gmodel.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_plus")
    yminus_var = gmodel.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_minus")

    gmodel.addConstr(x_var - ar_design_matrix @ beta_var == 0.0, name="commitment_eq")   # x = X@beta
    gmodel.addConstr(x_var + yplus_var - yminus_var == y_this_hour, name="mismatch_eq")   # x+y_plus-y_minus=actual

    if n_binary > 0:
        z_var = gmodel.addMVar(n_binary, vtype=GRB.BINARY, name="z")
        gmodel.addConstr(yplus_var[binary_rows] + z_var <= 1.0, name="complementarity_plus")
        gmodel.addConstr(yminus_var[binary_rows] - z_var <= 0.0, name="complementarity_minus")

    objective_expr = (
        (-W1 * da_this_hour) @ x_var   # [수정] 정규화 제거
        + surplus_cost @ yplus_var
        + shortage_cost @ yminus_var
    )
    gmodel.setObjective(objective_expr, GRB.MINIMIZE)
    gmodel.optimize()

    coefficients_by_hour[hour] = beta_var.X                          # beta(13개)만 뽑아 저장
    milp_success = (gmodel.Status == GRB.OPTIMAL)

    print(f"  시간대 {hour} MILP 완료 (성공 여부: {milp_success}, 이진변수 개수: {n_binary})")


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
    penalty_cost_i = ZHANG_C * rt_i     # [Zhang et al.] rho_plus = c*RT

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
