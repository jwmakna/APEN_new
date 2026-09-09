# -*- coding: utf-8 -*-

# =====================================================================
# model_combination_ar_mlr_kfold.py
#
# 목적: model_combination_ar_mlr.py 의 개선판. 이전 버전은 AR·MLR을
#       "학습기간 전체"로 학습한 뒤 그 학습기간 자체 예측값(in-sample)을
#       결합계수 학습에 그대로 썼는데, 이러면 결합모델이 학습기간에서
#       우연히 잘 맞은 쪽으로 쏠려서(w_AR이 특정 시간대에서 정확히
#       1.0000에 붙는 등) 테스트에서 오히려 AR·MLR 단독보다 나빠지는
#       문제가 실제로 확인됨.
#
#       이번 버전은 표준적인 "out-of-fold stacking" 방식을 씀:
#       학습 300일을 K개 조각으로 나눠서, 매번 K-1조각으로 AR·MLR을
#       새로 학습하고 나머지 1조각을 예측(그 조각 입장에서는 완전히
#       처음 보는 데이터) - 이걸 K번 반복해서 300일 전부에 대한
#       "정직한(out-of-sample)" AR·MLR 예측값을 모은 다음, 그 예측값으로
#       결합계수를 학습한다.
#
#       테스트기간 예측에 쓸 "최종" AR·MLR은 학습 300일 전체로 별도로
#       한 번 더 학습한다(fold별 모델은 결합계수 학습에만 쓰고 버림).
#
# 수익식: 3항(원본, 논문 Eq.1a 그대로)
# 데이터: merged_for_simulation_z03.csv, z03 블록18
# =====================================================================

import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# =====================================================================
# 0. 설정값
# =====================================================================
BASE_DIR = os.path.abspath(r"D:\Users\Downloads\AR MLR 합쳐본거")
MERGED_FILE = os.environ.get("MERGED_FILE", os.path.join(BASE_DIR, "merged_for_simulation_z03.csv"))

HOURS_PER_DAY = 12
LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21

TRAIN_START = pd.Timestamp(os.environ.get("TRAIN_START", "2013-08-25"))
TRAIN_END = pd.Timestamp(os.environ.get("TRAIN_END",   "2013-11-22"))
TEST_START = pd.Timestamp(os.environ.get("TEST_START",  "2013-11-23"))
TEST_END = pd.Timestamp(os.environ.get("TEST_END",    "2013-12-22"))
HISTORY_DATE = TRAIN_START - pd.Timedelta(days=1)

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0
PENALTY_RATE = 0.5

W1 = 1.0
W2 = 20.0

K_FOLDS = 5     # out-of-fold stacking에 쓸 조각 수 (300일/5=60일씩)

# additive CVaR 설정값 (z03 블록18에서 검증한 최종 형태와 동일)
# 결합계수(w0, w_AR) 학습에만 적용 - AR·MLR 기반모형 학습에는 적용 안 함
CVAR_ALPHA = 0.90
CVAR_LAMBDA = 0.05

OUTPUT_TAG = "proposed_combination_kfold_cvar"


# =====================================================================
# 1. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START


# =====================================================================
# 2. 이력(history) / 학습(train) / 테스트(test) 구간으로 자르기
# =====================================================================
is_history_date = daylight_table["local_date"] == HISTORY_DATE
history_rows = daylight_table[is_history_date].copy()
history_rows = history_rows.sort_values("hour_idx")

is_train_date = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train_date].copy()
train_rows = train_rows.sort_values(["local_date", "hour_idx"])

is_test_date = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test_date].copy()
test_rows = test_rows.sort_values(["local_date", "hour_idx"])

print("이력 날짜:", HISTORY_DATE.date(), "행 수:", len(history_rows))
print("학습 구간:", TRAIN_START.date(), "~", TRAIN_END.date(), "행 수:", len(train_rows))
print("테스트 구간:", TEST_START.date(), "~", TEST_END.date(), "행 수:", len(test_rows))


# =====================================================================
# 3. AR용 (날짜 x 12시간) 배열과 MLR용 1차원(flat) 배열을 동시에 만든다
# =====================================================================

history_solar = np.zeros((1, HOURS_PER_DAY))
row_counter = 0
for _, one_row in history_rows.iterrows():
    hour_position = row_counter % HOURS_PER_DAY
    history_solar[0, hour_position] = one_row["solar_power"]
    row_counter = row_counter + 1

train_dates_sorted = sorted(train_rows["local_date"].unique())
n_train_days = len(train_dates_sorted)

train_solar_2d = np.zeros((n_train_days, HOURS_PER_DAY))
train_da_price_2d = np.zeros((n_train_days, HOURS_PER_DAY))
train_rt_price_2d = np.zeros((n_train_days, HOURS_PER_DAY))

n_train_obs = len(train_rows)
train_solar_flat = np.zeros(n_train_obs)
train_dssrd_flat = np.zeros(n_train_obs)
train_dtsr_flat = np.zeros(n_train_obs)
train_hour_flat = np.zeros(n_train_obs)
train_da_price_flat = np.zeros(n_train_obs)
train_rt_price_flat = np.zeros(n_train_obs)

row_counter = 0
for _, one_row in train_rows.iterrows():
    day_position = row_counter // HOURS_PER_DAY
    hour_position = row_counter % HOURS_PER_DAY

    train_solar_2d[day_position, hour_position] = one_row["solar_power"]
    train_da_price_2d[day_position, hour_position] = one_row["da_price"]
    train_rt_price_2d[day_position, hour_position] = one_row["rt_price"]

    train_solar_flat[row_counter] = one_row["solar_power"]
    train_dssrd_flat[row_counter] = one_row["dssrd"]
    train_dtsr_flat[row_counter] = one_row["dtsr"]
    train_hour_flat[row_counter] = one_row["hour_idx"]
    train_da_price_flat[row_counter] = one_row["da_price"]
    train_rt_price_flat[row_counter] = one_row["rt_price"]

    row_counter = row_counter + 1

test_dates_sorted = sorted(test_rows["local_date"].unique())
n_test_days = len(test_dates_sorted)

test_solar_2d = np.zeros((n_test_days, HOURS_PER_DAY))
test_da_price_2d = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt_price_2d = np.zeros((n_test_days, HOURS_PER_DAY))

n_test_obs = len(test_rows)
test_solar_flat = np.zeros(n_test_obs)
test_dssrd_flat = np.zeros(n_test_obs)
test_dtsr_flat = np.zeros(n_test_obs)
test_hour_flat = np.zeros(n_test_obs)
test_da_price_flat = np.zeros(n_test_obs)
test_rt_price_flat = np.zeros(n_test_obs)

row_counter = 0
for _, one_row in test_rows.iterrows():
    day_position = row_counter // HOURS_PER_DAY
    hour_position = row_counter % HOURS_PER_DAY

    test_solar_2d[day_position, hour_position] = one_row["solar_power"]
    test_da_price_2d[day_position, hour_position] = one_row["da_price"]
    test_rt_price_2d[day_position, hour_position] = one_row["rt_price"]

    test_solar_flat[row_counter] = one_row["solar_power"]
    test_dssrd_flat[row_counter] = one_row["dssrd"]
    test_dtsr_flat[row_counter] = one_row["dtsr"]
    test_hour_flat[row_counter] = one_row["hour_idx"]
    test_da_price_flat[row_counter] = one_row["da_price"]
    test_rt_price_flat[row_counter] = one_row["rt_price"]

    row_counter = row_counter + 1


# =====================================================================
# 4. AR 입력행렬 / MLR 입력행렬 만들기
# =====================================================================

history_and_train_solar = np.vstack([history_solar, train_solar_2d])
n_ar_features = HOURS_PER_DAY + 1
ar_intercept_column = np.ones((n_train_days, 1))
ar_lag_features = np.zeros((n_train_days, HOURS_PER_DAY))
for day_index in range(n_train_days):
    previous_day_values = history_and_train_solar[day_index]
    ar_lag_features[day_index] = previous_day_values[::-1]
ar_design_matrix = np.hstack([ar_intercept_column, ar_lag_features])   # (300, 13)

n_mlr_features = 4
X_train_mlr = np.zeros((n_train_obs, n_mlr_features))
for i in range(n_train_obs):
    X_train_mlr[i, 0] = 1.0
    X_train_mlr[i, 1] = train_dssrd_flat[i]
    X_train_mlr[i, 2] = train_dtsr_flat[i]
    X_train_mlr[i, 3] = train_hour_flat[i]

X_test_mlr = np.zeros((n_test_obs, n_mlr_features))
for i in range(n_test_obs):
    X_test_mlr[i, 0] = 1.0
    X_test_mlr[i, 1] = test_dssrd_flat[i]
    X_test_mlr[i, 2] = test_dtsr_flat[i]
    X_test_mlr[i, 3] = test_hour_flat[i]


# =====================================================================
# 4-C. K-fold 조각 배정 (날짜 순서대로 연속 블록 - 시계열이라 무작위
#      셔플 대신 연속 블록으로 나눈다)
# =====================================================================

fold_size = n_train_days // K_FOLDS
day_fold_id = np.minimum(np.arange(n_train_days) // fold_size, K_FOLDS - 1)
# ↑ 마지막 fold가 나머지(300이 K로 안 나눠떨어지면)를 흡수함

for fold_check in range(K_FOLDS):
    n_days_in_fold = int(np.sum(day_fold_id == fold_check))
    print(f"fold {fold_check}: {n_days_in_fold}일")


# =====================================================================
# 5-A. AR out-of-fold 예측 생성 (K번 반복)
#      - 매번 이 fold만 빼고 나머지 (K-1)/K 로 12시간 MILP를 새로 학습
#      - 학습에 쓰지 않은 이 fold의 날짜만 예측해서 채워 넣는다
# =====================================================================

ar_train_oof_2d = np.zeros((n_train_days, HOURS_PER_DAY))

for fold_idx in range(K_FOLDS):

    held_out_days = np.where(day_fold_id == fold_idx)[0]
    held_in_days = np.where(day_fold_id != fold_idx)[0]
    n_held_in = len(held_in_days)

    coefficients_by_hour_ar_fold = np.zeros((HOURS_PER_DAY, n_ar_features))

    for hour in range(HOURS_PER_DAY):

        y_this_hour = train_solar_2d[held_in_days, hour]
        da_this_hour = train_da_price_2d[held_in_days, hour]
        rt_this_hour = train_rt_price_2d[held_in_days, hour]
        design_this_hour = ar_design_matrix[held_in_days]

        surplus_cost = np.zeros(n_held_in)
        shortage_cost = np.zeros(n_held_in)
        for i in range(n_held_in):
            penalty_i = PENALTY_RATE * da_this_hour[i]
            surplus_cost[i] = (-W1 * rt_this_hour[i]) + W2
            shortage_cost[i] = (W1 * penalty_i) + W2

        binary_row_list = []
        for i in range(n_held_in):
            if surplus_cost[i] + shortage_cost[i] < 0.0:
                binary_row_list.append(i)
        binary_rows = np.array(binary_row_list, dtype=int)
        n_binary = len(binary_rows)

        gmodel_ar_fold = gp.Model(f"ar_oof_fold{fold_idx}_hour{hour}")
        gmodel_ar_fold.Params.OutputFlag = 0
        gmodel_ar_fold.Params.MIPGap = 1e-9

        beta_var = gmodel_ar_fold.addMVar(n_ar_features, lb=-GRB.INFINITY, name="beta")
        x_var = gmodel_ar_fold.addMVar(n_held_in, lb=0.0, ub=1.0, name="x")
        yplus_var = gmodel_ar_fold.addMVar(n_held_in, lb=0.0, ub=1.0, name="y_plus")
        yminus_var = gmodel_ar_fold.addMVar(n_held_in, lb=0.0, ub=1.0, name="y_minus")

        gmodel_ar_fold.addConstr(x_var - design_this_hour @ beta_var == 0.0)
        gmodel_ar_fold.addConstr(x_var + yplus_var - yminus_var == y_this_hour)

        if n_binary > 0:
            z_var = gmodel_ar_fold.addMVar(n_binary, vtype=GRB.BINARY, name="z")
            gmodel_ar_fold.addConstr(yplus_var[binary_rows] + z_var <= 1.0)
            gmodel_ar_fold.addConstr(yminus_var[binary_rows] - z_var <= 0.0)

        objective_expr = (
            (-W1 * da_this_hour) @ x_var
            + surplus_cost @ yplus_var
            + shortage_cost @ yminus_var
        )
        gmodel_ar_fold.setObjective(objective_expr, GRB.MINIMIZE)
        gmodel_ar_fold.optimize()

        coefficients_by_hour_ar_fold[hour] = beta_var.X

    # held-out 날짜만 이 fold의 계수로 예측해서 채워 넣는다 (rolling 아님,
    # AR 입력은 실제 전날값이라 그냥 대입하면 됨)
    for day_index in held_out_days:
        feature_vector = ar_design_matrix[day_index]
        for hour in range(HOURS_PER_DAY):
            raw_prediction = np.dot(coefficients_by_hour_ar_fold[hour], feature_vector)
            ar_train_oof_2d[day_index, hour] = min(max(raw_prediction, 0.0), 1.0)

    print(f"AR fold {fold_idx} 완료 (held-out {len(held_out_days)}일 예측)")


# =====================================================================
# 5-B. MLR out-of-fold 예측 생성 (K번 반복)
# =====================================================================

mlr_train_oof_2d = np.zeros((n_train_days, HOURS_PER_DAY))

for fold_idx in range(K_FOLDS):

    held_out_days = np.where(day_fold_id == fold_idx)[0]
    held_in_days = np.where(day_fold_id != fold_idx)[0]

    held_in_flat = []
    for d in held_in_days:
        for h in range(HOURS_PER_DAY):
            held_in_flat.append(d * HOURS_PER_DAY + h)
    held_in_flat = np.array(held_in_flat, dtype=int)
    n_held_in_flat = len(held_in_flat)

    y_fold = train_solar_flat[held_in_flat]
    da_fold = train_da_price_flat[held_in_flat]
    rt_fold = train_rt_price_flat[held_in_flat]
    X_fold = X_train_mlr[held_in_flat]

    surplus_cost_mlr = np.zeros(n_held_in_flat)
    shortage_cost_mlr = np.zeros(n_held_in_flat)
    for i in range(n_held_in_flat):
        penalty_i = PENALTY_RATE * da_fold[i]
        surplus_cost_mlr[i] = (-W1 * rt_fold[i]) + W2
        shortage_cost_mlr[i] = (W1 * penalty_i) + W2

    binary_row_list_mlr = []
    for i in range(n_held_in_flat):
        if surplus_cost_mlr[i] + shortage_cost_mlr[i] < 0.0:
            binary_row_list_mlr.append(i)
    binary_rows_mlr = np.array(binary_row_list_mlr, dtype=int)
    n_binary_mlr = len(binary_rows_mlr)

    gmodel_mlr_fold = gp.Model(f"mlr_oof_fold{fold_idx}")
    gmodel_mlr_fold.Params.OutputFlag = 0
    gmodel_mlr_fold.Params.MIPGap = 1e-9

    beta_var_mlr = gmodel_mlr_fold.addMVar(n_mlr_features, lb=-GRB.INFINITY, name="beta")
    x_var_mlr = gmodel_mlr_fold.addMVar(n_held_in_flat, lb=0.0, ub=1.0, name="x")
    yplus_var_mlr = gmodel_mlr_fold.addMVar(n_held_in_flat, lb=0.0, ub=1.0, name="y_plus")
    yminus_var_mlr = gmodel_mlr_fold.addMVar(n_held_in_flat, lb=0.0, ub=1.0, name="y_minus")

    gmodel_mlr_fold.addConstr(x_var_mlr - X_fold @ beta_var_mlr == 0.0)
    gmodel_mlr_fold.addConstr(x_var_mlr + yplus_var_mlr - yminus_var_mlr == y_fold)

    if n_binary_mlr > 0:
        z_var_mlr = gmodel_mlr_fold.addMVar(n_binary_mlr, vtype=GRB.BINARY, name="z")
        gmodel_mlr_fold.addConstr(yplus_var_mlr[binary_rows_mlr] + z_var_mlr <= 1.0)
        gmodel_mlr_fold.addConstr(yminus_var_mlr[binary_rows_mlr] - z_var_mlr <= 0.0)

    objective_expr_mlr = (
        (-W1 * da_fold) @ x_var_mlr
        + surplus_cost_mlr @ yplus_var_mlr
        + shortage_cost_mlr @ yminus_var_mlr
    )
    gmodel_mlr_fold.setObjective(objective_expr_mlr, GRB.MINIMIZE)
    gmodel_mlr_fold.optimize()

    mlr_coefficients_fold = beta_var_mlr.X

    for day_index in held_out_days:
        for hour in range(HOURS_PER_DAY):
            flat_row = day_index * HOURS_PER_DAY + hour
            raw_prediction = np.dot(mlr_coefficients_fold, X_train_mlr[flat_row])
            mlr_train_oof_2d[day_index, hour] = min(max(raw_prediction, 0.0), 1.0)

    print(f"MLR fold {fold_idx} 완료 (held-out {len(held_out_days)}일 예측)")


# =====================================================================
# 6-A. Proposed AR "최종" 학습 (테스트 예측용, 300일 전체)
# =====================================================================

coefficients_by_hour_ar = np.zeros((HOURS_PER_DAY, n_ar_features))

for hour in range(HOURS_PER_DAY):

    y_this_hour = train_solar_2d[:, hour]
    da_this_hour = train_da_price_2d[:, hour]
    rt_this_hour = train_rt_price_2d[:, hour]

    surplus_cost = np.zeros(n_train_days)
    shortage_cost = np.zeros(n_train_days)
    for i in range(n_train_days):
        penalty_i = PENALTY_RATE * da_this_hour[i]
        surplus_cost[i] = (-W1 * rt_this_hour[i]) + W2
        shortage_cost[i] = (W1 * penalty_i) + W2

    binary_row_list = []
    for i in range(n_train_days):
        if surplus_cost[i] + shortage_cost[i] < 0.0:
            binary_row_list.append(i)
    binary_rows = np.array(binary_row_list, dtype=int)
    n_binary = len(binary_rows)

    gmodel_ar = gp.Model(f"proposed_ar_final_hour_{hour}")
    gmodel_ar.Params.OutputFlag = 0
    gmodel_ar.Params.MIPGap = 1e-9

    beta_var_ar = gmodel_ar.addMVar(n_ar_features, lb=-GRB.INFINITY, name="beta")
    x_var_ar = gmodel_ar.addMVar(n_train_days, lb=0.0, ub=1.0, name="x")
    yplus_var_ar = gmodel_ar.addMVar(n_train_days, lb=0.0, ub=1.0, name="y_plus")
    yminus_var_ar = gmodel_ar.addMVar(n_train_days, lb=0.0, ub=1.0, name="y_minus")

    gmodel_ar.addConstr(x_var_ar - ar_design_matrix @ beta_var_ar == 0.0)
    gmodel_ar.addConstr(x_var_ar + yplus_var_ar - yminus_var_ar == y_this_hour)

    if n_binary > 0:
        z_var_ar = gmodel_ar.addMVar(n_binary, vtype=GRB.BINARY, name="z")
        gmodel_ar.addConstr(yplus_var_ar[binary_rows] + z_var_ar <= 1.0)
        gmodel_ar.addConstr(yminus_var_ar[binary_rows] - z_var_ar <= 0.0)

    objective_expr_ar = (
        (-W1 * da_this_hour) @ x_var_ar
        + surplus_cost @ yplus_var_ar
        + shortage_cost @ yminus_var_ar
    )
    gmodel_ar.setObjective(objective_expr_ar, GRB.MINIMIZE)
    gmodel_ar.optimize()

    coefficients_by_hour_ar[hour] = beta_var_ar.X

print("Proposed AR 최종 학습 완료 (300일 전체, 12개 시간대)")


# =====================================================================
# 6-B. Proposed MLR "최종" 학습 (테스트 예측용, 300일 전체)
# =====================================================================

surplus_cost_mlr = np.zeros(n_train_obs)
shortage_cost_mlr = np.zeros(n_train_obs)
for i in range(n_train_obs):
    penalty_i = PENALTY_RATE * train_da_price_flat[i]
    surplus_cost_mlr[i] = (-W1 * train_rt_price_flat[i]) + W2
    shortage_cost_mlr[i] = (W1 * penalty_i) + W2

binary_row_list_mlr = []
for i in range(n_train_obs):
    if surplus_cost_mlr[i] + shortage_cost_mlr[i] < 0.0:
        binary_row_list_mlr.append(i)
binary_rows_mlr = np.array(binary_row_list_mlr, dtype=int)
n_binary_mlr = len(binary_rows_mlr)

gmodel_mlr = gp.Model("proposed_mlr_final")
gmodel_mlr.Params.OutputFlag = 0
gmodel_mlr.Params.MIPGap = 1e-9

beta_var_mlr = gmodel_mlr.addMVar(n_mlr_features, lb=-GRB.INFINITY, name="beta")
x_var_mlr = gmodel_mlr.addMVar(n_train_obs, lb=0.0, ub=1.0, name="x")
yplus_var_mlr = gmodel_mlr.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_plus")
yminus_var_mlr = gmodel_mlr.addMVar(n_train_obs, lb=0.0, ub=1.0, name="y_minus")

gmodel_mlr.addConstr(x_var_mlr - X_train_mlr @ beta_var_mlr == 0.0)
gmodel_mlr.addConstr(x_var_mlr + yplus_var_mlr - yminus_var_mlr == train_solar_flat)

if n_binary_mlr > 0:
    z_var_mlr = gmodel_mlr.addMVar(n_binary_mlr, vtype=GRB.BINARY, name="z")
    gmodel_mlr.addConstr(yplus_var_mlr[binary_rows_mlr] + z_var_mlr <= 1.0)
    gmodel_mlr.addConstr(yminus_var_mlr[binary_rows_mlr] - z_var_mlr <= 0.0)

objective_expr_mlr = (
    (-W1 * train_da_price_flat) @ x_var_mlr
    + surplus_cost_mlr @ yplus_var_mlr
    + shortage_cost_mlr @ yminus_var_mlr
)
gmodel_mlr.setObjective(objective_expr_mlr, GRB.MINIMIZE)
gmodel_mlr.optimize()

mlr_coefficients = beta_var_mlr.X
print("Proposed MLR 최종 학습 완료 (300일 전체), 계수:", mlr_coefficients)


# =====================================================================
# 7. AR/MLR 테스트기간 예측 (최종 모델로, rolling)
# =====================================================================

ar_test_forecast_2d = np.zeros((n_test_days, HOURS_PER_DAY))
previous_day_actual = train_solar_2d[-1]
for day_index in range(n_test_days):
    feature_vector = np.concatenate([[1.0], previous_day_actual[::-1]])
    for hour in range(HOURS_PER_DAY):
        raw_prediction = np.dot(coefficients_by_hour_ar[hour], feature_vector)
        ar_test_forecast_2d[day_index, hour] = min(max(raw_prediction, 0.0), 1.0)
    previous_day_actual = test_solar_2d[day_index]

mlr_test_forecast_flat = np.zeros(n_test_obs)
for i in range(n_test_obs):
    raw_prediction = np.dot(mlr_coefficients, X_test_mlr[i])
    mlr_test_forecast_flat[i] = min(max(raw_prediction, 0.0), 1.0)
mlr_test_forecast_2d = mlr_test_forecast_flat.reshape(n_test_days, HOURS_PER_DAY)

print("AR/MLR 최종 테스트 예측 완료")


# =====================================================================
# 8. 결합계수 학습 (시간대별 12세트, 볼록결합 + 자유 절편)
#    - out-of-fold 예측값 사용 + additive CVaR(λ=0.05) 항 추가
#    commitment = MLR예측(oof) + w0 + w_AR*(AR예측(oof) - MLR예측(oof))
# =====================================================================

# ---- CVaR 항에 쓸 학습기간 오라클 (시간대별로 미리 한 번만 계산) ----
profit_scale = CAPACITY_MW * DURATION_HOURS
oracle_profit_train_unscaled_by_hour = np.zeros((HOURS_PER_DAY, n_train_days))
for hour in range(HOURS_PER_DAY):
    y_this_hour_oc = train_solar_2d[:, hour]
    da_this_hour_oc = train_da_price_2d[:, hour]
    rt_this_hour_oc = train_rt_price_2d[:, hour]
    for i in range(n_train_days):
        actual_i = y_this_hour_oc[i]
        da_i = da_this_hour_oc[i]
        rt_i = rt_this_hour_oc[i]
        penalty_i = PENALTY_RATE * da_i
        profit_commit_0 = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
        profit_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
        surplus_if_full = max(actual_i - 1.0, 0.0)
        shortage_if_full = max(1.0 - actual_i, 0.0)
        profit_commit_1 = CAPACITY_MW * DURATION_HOURS * (
            da_i * 1.0 + rt_i * surplus_if_full - penalty_i * shortage_if_full
        )
        oracle_i = max(profit_commit_0, profit_commit_actual, profit_commit_1)
        oracle_profit_train_unscaled_by_hour[hour, i] = oracle_i / profit_scale

combo_w0_by_hour = np.zeros(HOURS_PER_DAY)
combo_war_by_hour = np.zeros(HOURS_PER_DAY)

for hour in range(HOURS_PER_DAY):

    y_this_hour = train_solar_2d[:, hour]
    da_this_hour = train_da_price_2d[:, hour]
    rt_this_hour = train_rt_price_2d[:, hour]

    ar_pred_h = ar_train_oof_2d[:, hour]      # <- out-of-fold 예측
    mlr_pred_h = mlr_train_oof_2d[:, hour]    # <- out-of-fold 예측
    diff_h = ar_pred_h - mlr_pred_h
    offset_h = mlr_pred_h

    combo_design = np.zeros((n_train_days, 2))
    combo_design[:, 0] = 1.0
    combo_design[:, 1] = diff_h

    surplus_cost = np.zeros(n_train_days)
    shortage_cost = np.zeros(n_train_days)
    for i in range(n_train_days):
        penalty_i = PENALTY_RATE * da_this_hour[i]
        surplus_cost[i] = (-W1 * rt_this_hour[i]) + W2
        shortage_cost[i] = (W1 * penalty_i) + W2

    binary_row_list = []
    for i in range(n_train_days):
        if surplus_cost[i] + shortage_cost[i] < 0.0:
            binary_row_list.append(i)
    binary_rows = np.array(binary_row_list, dtype=int)
    n_binary = len(binary_rows)

    gmodel_combo = gp.Model(f"combo_hour_{hour}")
    gmodel_combo.Params.OutputFlag = 0
    gmodel_combo.Params.MIPGap = 1e-9

    beta_var_combo = gmodel_combo.addMVar(
        2,
        lb=np.array([-GRB.INFINITY, 0.0]),
        ub=np.array([GRB.INFINITY, 1.0]),
        name="beta"
    )
    x_var_combo = gmodel_combo.addMVar(n_train_days, lb=0.0, ub=1.0, name="x")
    yplus_var_combo = gmodel_combo.addMVar(n_train_days, lb=0.0, ub=1.0, name="y_plus")
    yminus_var_combo = gmodel_combo.addMVar(n_train_days, lb=0.0, ub=1.0, name="y_minus")

    gmodel_combo.addConstr(x_var_combo - combo_design @ beta_var_combo == offset_h)
    gmodel_combo.addConstr(x_var_combo + yplus_var_combo - yminus_var_combo == y_this_hour)

    if n_binary > 0:
        z_var_combo = gmodel_combo.addMVar(n_binary, vtype=GRB.BINARY, name="z")
        gmodel_combo.addConstr(yplus_var_combo[binary_rows] + z_var_combo <= 1.0)
        gmodel_combo.addConstr(yminus_var_combo[binary_rows] - z_var_combo <= 0.0)

    # ---- additive CVaR(λ=0.05) 항 ----
    zeta_var_combo = gmodel_combo.addVar(lb=0.0, name="cvar_zeta")
    s_var_combo = gmodel_combo.addMVar(n_train_days, lb=0.0, name="cvar_excess")

    shortage_price_this_hour = PENALTY_RATE * da_this_hour   # 3항 부족비용
    oracle_this_hour = oracle_profit_train_unscaled_by_hour[hour]

    for i in range(n_train_days):
        profit_expression_i = (
            da_this_hour[i] * x_var_combo[i]
            + rt_this_hour[i] * yplus_var_combo[i]
            - shortage_price_this_hour[i] * yminus_var_combo[i]
        )
        regret_expression_i = oracle_this_hour[i] - profit_expression_i
        gmodel_combo.addConstr(
            s_var_combo[i] >= regret_expression_i - zeta_var_combo,
            name=f"cvar_tail_{i}"
        )

    total_regret_objective_combo = (
        (-W1 * da_this_hour) @ x_var_combo
        + surplus_cost @ yplus_var_combo
        + shortage_cost @ yminus_var_combo
    )
    cvar_objective_combo = (
        W1 * CVAR_LAMBDA
        * (n_train_days * zeta_var_combo + s_var_combo.sum() / (1.0 - CVAR_ALPHA))
    )
    gmodel_combo.setObjective(total_regret_objective_combo + cvar_objective_combo, GRB.MINIMIZE)
    gmodel_combo.optimize()

    combo_w0_by_hour[hour] = beta_var_combo.X[0]
    combo_war_by_hour[hour] = beta_var_combo.X[1]

    print(f"  시간대 {hour}: w0={combo_w0_by_hour[hour]:.4f}, w_AR={combo_war_by_hour[hour]:.4f}, w_MLR={1.0-combo_war_by_hour[hour]:.4f}")


# =====================================================================
# 9. 결합 테스트 예측
# =====================================================================

combo_test_forecast_2d = np.zeros((n_test_days, HOURS_PER_DAY))
for hour in range(HOURS_PER_DAY):
    ar_pred_h = ar_test_forecast_2d[:, hour]
    mlr_pred_h = mlr_test_forecast_2d[:, hour]
    raw_combo = mlr_pred_h + combo_w0_by_hour[hour] + combo_war_by_hour[hour] * (ar_pred_h - mlr_pred_h)
    combo_test_forecast_2d[:, hour] = np.clip(raw_combo, 0.0, 1.0)


# =====================================================================
# 10. 평가 - AR단독 / MLR단독 / 결합 (def 없이 세 번 반복)
# =====================================================================

# --- 10-A. Proposed AR 단독 평가 ---
actual_flat_ar = test_solar_2d.flatten()
predicted_flat_ar = ar_test_forecast_2d.flatten()
da_flat_ar = test_da_price_2d.flatten()
rt_flat_ar = test_rt_price_2d.flatten()

sse_ar = 0.0
for i in range(len(actual_flat_ar)):
    e = actual_flat_ar[i] - predicted_flat_ar[i]
    sse_ar = sse_ar + e * e
rmse_ar = (sse_ar / len(actual_flat_ar)) ** 0.5
avg_actual_ar = 0.0
for i in range(len(actual_flat_ar)):
    avg_actual_ar = avg_actual_ar + actual_flat_ar[i]
avg_actual_ar = avg_actual_ar / len(actual_flat_ar)
nrmse_ar = 100.0 * rmse_ar / avg_actual_ar

sum_realized_ar = 0.0
sum_oracle_ar = 0.0
for i in range(len(actual_flat_ar)):
    actual_i = actual_flat_ar[i]
    commitment_i = predicted_flat_ar[i]
    da_i = da_flat_ar[i]
    rt_i = rt_flat_ar[i]
    penalty_cost_i = PENALTY_RATE * da_i
    mismatch_i = actual_i - commitment_i
    surplus_i = max(mismatch_i, 0.0)
    shortage_i = max(-mismatch_i, 0.0)
    realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
        da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
    )
    sum_realized_ar = sum_realized_ar + realized_profit_i

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
    sum_oracle_ar = sum_oracle_ar + oracle_profit_i

gap_ar = 100.0 * (sum_oracle_ar - sum_realized_ar) / sum_oracle_ar

# --- 10-B. Proposed MLR 단독 평가 ---
actual_flat_mlr = test_solar_2d.flatten()
predicted_flat_mlr = mlr_test_forecast_2d.flatten()
da_flat_mlr = test_da_price_2d.flatten()
rt_flat_mlr = test_rt_price_2d.flatten()

sse_mlr = 0.0
for i in range(len(actual_flat_mlr)):
    e = actual_flat_mlr[i] - predicted_flat_mlr[i]
    sse_mlr = sse_mlr + e * e
rmse_mlr = (sse_mlr / len(actual_flat_mlr)) ** 0.5
avg_actual_mlr = 0.0
for i in range(len(actual_flat_mlr)):
    avg_actual_mlr = avg_actual_mlr + actual_flat_mlr[i]
avg_actual_mlr = avg_actual_mlr / len(actual_flat_mlr)
nrmse_mlr = 100.0 * rmse_mlr / avg_actual_mlr

sum_realized_mlr = 0.0
sum_oracle_mlr_eval = 0.0
for i in range(len(actual_flat_mlr)):
    actual_i = actual_flat_mlr[i]
    commitment_i = predicted_flat_mlr[i]
    da_i = da_flat_mlr[i]
    rt_i = rt_flat_mlr[i]
    penalty_cost_i = PENALTY_RATE * da_i
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
    sum_oracle_mlr_eval = sum_oracle_mlr_eval + oracle_profit_i

gap_mlr = 100.0 * (sum_oracle_mlr_eval - sum_realized_mlr) / sum_oracle_mlr_eval

# --- 10-C. 결합모델 평가 ---
actual_flat_combo = test_solar_2d.flatten()
predicted_flat_combo = combo_test_forecast_2d.flatten()
da_flat_combo = test_da_price_2d.flatten()
rt_flat_combo = test_rt_price_2d.flatten()

sse_combo = 0.0
for i in range(len(actual_flat_combo)):
    e = actual_flat_combo[i] - predicted_flat_combo[i]
    sse_combo = sse_combo + e * e
rmse_combo = (sse_combo / len(actual_flat_combo)) ** 0.5
avg_actual_combo = 0.0
for i in range(len(actual_flat_combo)):
    avg_actual_combo = avg_actual_combo + actual_flat_combo[i]
avg_actual_combo = avg_actual_combo / len(actual_flat_combo)
nrmse_combo = 100.0 * rmse_combo / avg_actual_combo

sum_realized_combo = 0.0
sum_oracle_combo = 0.0
for i in range(len(actual_flat_combo)):
    actual_i = actual_flat_combo[i]
    commitment_i = predicted_flat_combo[i]
    da_i = da_flat_combo[i]
    rt_i = rt_flat_combo[i]
    penalty_cost_i = PENALTY_RATE * da_i
    mismatch_i = actual_i - commitment_i
    surplus_i = max(mismatch_i, 0.0)
    shortage_i = max(-mismatch_i, 0.0)
    realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
        da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
    )
    sum_realized_combo = sum_realized_combo + realized_profit_i

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
    sum_oracle_combo = sum_oracle_combo + oracle_profit_i

gap_combo = 100.0 * (sum_oracle_combo - sum_realized_combo) / sum_oracle_combo


# =====================================================================
# 11. 결과 비교표 출력
# =====================================================================

print()
print(f"=== AR / MLR / 결합(Out-of-fold, K={K_FOLDS}, CVaR λ={CVAR_LAMBDA}) 비교 - z03 블록18, 3항 ===")
print("| 모형 | nRMSE | optimality gap |")
print("|---|---:|---:|")
print(f"| Proposed AR | {nrmse_ar:.4f}% | {gap_ar:.4f}% |")
print(f"| Proposed MLR | {nrmse_mlr:.4f}% | {gap_mlr:.4f}% |")
print(f"| **결합(AR+MLR, OOF)** | **{nrmse_combo:.4f}%** | **{gap_combo:.4f}%** |")
print()
print("시간대별 결합계수 (w0, w_AR, w_MLR):")
local_hours_list = list(range(LOCAL_HOUR_START, LOCAL_HOUR_START + HOURS_PER_DAY))
for hour in range(HOURS_PER_DAY):
    print(
        f"  {local_hours_list[hour]}시: w0={combo_w0_by_hour[hour]:+.4f}, "
        f"w_AR={combo_war_by_hour[hour]:.4f}, w_MLR={1.0-combo_war_by_hour[hour]:.4f}"
    )


# =====================================================================
# 12. 시간대별 진단 + npz 저장
# =====================================================================

forecast_error = combo_test_forecast_2d - test_solar_2d
absolute_error_capacity_percent = np.abs(forecast_error) * 100.0
hourly_rmse_capacity_percent = np.sqrt(np.mean(forecast_error ** 2, axis=0)) * 100.0
hourly_mae_capacity_percent = np.mean(np.abs(forecast_error), axis=0) * 100.0
hourly_bias_capacity_percent = np.mean(forecast_error, axis=0) * 100.0

# 참고용: AR단독/MLR단독의 시간대별 RMSE도 같이 저장 (비교하기 편하게)
ar_forecast_error = ar_test_forecast_2d - test_solar_2d
mlr_forecast_error = mlr_test_forecast_2d - test_solar_2d
hourly_rmse_ar = np.sqrt(np.mean(ar_forecast_error ** 2, axis=0)) * 100.0
hourly_rmse_mlr = np.sqrt(np.mean(mlr_forecast_error ** 2, axis=0)) * 100.0

economic_realized_profit = np.zeros((n_test_days, HOURS_PER_DAY))
economic_oracle_profit = np.zeros((n_test_days, HOURS_PER_DAY))
economic_regret = np.zeros((n_test_days, HOURS_PER_DAY))

for day_index in range(n_test_days):
    for hour_index in range(HOURS_PER_DAY):
        actual_i = test_solar_2d[day_index, hour_index]
        commitment_i = combo_test_forecast_2d[day_index, hour_index]
        da_i = test_da_price_2d[day_index, hour_index]
        rt_i = test_rt_price_2d[day_index, hour_index]
        penalty_cost_i = PENALTY_RATE * da_i

        mismatch_i = actual_i - commitment_i
        surplus_i = max(mismatch_i, 0.0)
        shortage_i = max(-mismatch_i, 0.0)
        realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
            da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
        )

        profit_commit_zero = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
        if actual_i <= 1.0:
            profit_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
        else:
            profit_commit_actual = -np.inf
        surplus_if_full = max(actual_i - 1.0, 0.0)
        shortage_if_full = max(1.0 - actual_i, 0.0)
        profit_commit_full = CAPACITY_MW * DURATION_HOURS * (
            da_i * 1.0 + rt_i * surplus_if_full - penalty_cost_i * shortage_if_full
        )
        oracle_profit_i = max(profit_commit_zero, profit_commit_actual, profit_commit_full)

        economic_realized_profit[day_index, hour_index] = realized_profit_i
        economic_oracle_profit[day_index, hour_index] = oracle_profit_i
        economic_regret[day_index, hour_index] = oracle_profit_i - realized_profit_i

economic_regret[np.abs(economic_regret) < 1e-8] = 0.0

CVAR_ALPHA = 0.90
mean_regret_by_hour = np.mean(economic_regret, axis=0)
p90_regret_by_hour = np.percentile(economic_regret, 90, axis=0)
max_regret_by_hour = np.max(economic_regret, axis=0)

hourly_tail_count = max(1, int(np.ceil((1.0 - CVAR_ALPHA) * n_test_days)))
sorted_regret_by_hour = np.sort(economic_regret, axis=0)
cvar_regret_by_hour = np.mean(sorted_regret_by_hour[-hourly_tail_count:, :], axis=0)

regret_flat = economic_regret.flatten()
overall_tail_count = max(1, int(np.ceil((1.0 - CVAR_ALPHA) * len(regret_flat))))
sorted_regret_flat = np.sort(regret_flat)
overall_cvar_regret = np.mean(sorted_regret_flat[-overall_tail_count:])
overall_max_regret = np.max(regret_flat)

regret_sum_by_hour = np.sum(economic_regret, axis=0)
oracle_profit_sum_by_hour = np.sum(economic_oracle_profit, axis=0)
hourly_gap_percent = np.full(HOURS_PER_DAY, np.nan)
for hour_index in range(HOURS_PER_DAY):
    if abs(oracle_profit_sum_by_hour[hour_index]) > 1e-12:
        hourly_gap_percent[hour_index] = 100.0 * regret_sum_by_hour[hour_index] / oracle_profit_sum_by_hour[hour_index]

local_hours = np.arange(LOCAL_HOUR_START, LOCAL_HOUR_START + HOURS_PER_DAY)

hourly_diagnostics_table = pd.DataFrame({
    "local_hour": local_hours,
    "RMSE_AR": hourly_rmse_ar,
    "RMSE_MLR": hourly_rmse_mlr,
    "RMSE_combo": hourly_rmse_capacity_percent,
    "MAE_capacity_percent": hourly_mae_capacity_percent,
    "bias_capacity_percent": hourly_bias_capacity_percent,
    "mean_economic_regret": mean_regret_by_hour,
    "P90_economic_regret": p90_regret_by_hour,
    "CVaR90_economic_regret": cvar_regret_by_hour,
    "max_economic_regret": max_regret_by_hour,
    "hourly_optimality_gap_percent": hourly_gap_percent,
    "combo_w0": combo_w0_by_hour,
    "combo_w_AR": combo_war_by_hour,
    "combo_w_MLR": 1.0 - combo_war_by_hour,
})

print()
print("=== 결합모델(OOF): 시간대별 예측오차·경제적 regret·결합계수 ===")
print(hourly_diagnostics_table.round(6).to_string(index=False))

hourly_csv_name = f"{OUTPUT_TAG}_hourly_diagnostics.csv"
hourly_diagnostics_table.to_csv(hourly_csv_name, index=False, encoding="utf-8-sig")
print()
print("시간대별 결과 저장 완료:", hourly_csv_name)

date_labels = []
for one_date in test_dates_sorted:
    date_labels.append(pd.Timestamp(one_date).strftime("%m-%d"))

diagnostics_npz_name = f"{OUTPUT_TAG}_combined_diagnostics.npz"
np.savez(
    diagnostics_npz_name,
    actual=test_solar_2d,
    prediction=combo_test_forecast_2d,
    ar_prediction=ar_test_forecast_2d,
    mlr_prediction=mlr_test_forecast_2d,
    ar_train_oof=ar_train_oof_2d,
    mlr_train_oof=mlr_train_oof_2d,
    absolute_error_capacity_percent=absolute_error_capacity_percent,
    hourly_rmse_capacity_percent=hourly_rmse_capacity_percent,
    realized_profit=economic_realized_profit,
    oracle_profit=economic_oracle_profit,
    regret=economic_regret,
    mean_regret_by_hour=mean_regret_by_hour,
    p90_regret_by_hour=p90_regret_by_hour,
    cvar_regret_by_hour=cvar_regret_by_hour,
    max_regret_by_hour=max_regret_by_hour,
    hourly_gap_percent=hourly_gap_percent,
    dates=np.array(date_labels),
    hours=local_hours,
    overall_nrmse=np.array(nrmse_combo),
    overall_gap=np.array(gap_combo),
    ar_nrmse=np.array(nrmse_ar),
    ar_gap=np.array(gap_ar),
    mlr_nrmse=np.array(nrmse_mlr),
    mlr_gap=np.array(gap_mlr),
    overall_mean_regret=np.array(np.mean(economic_regret)),
    overall_cvar_regret=np.array(overall_cvar_regret),
    overall_max_regret=np.array(overall_max_regret),
    cvar_alpha=np.array(CVAR_ALPHA),
    hourly_tail_count=np.array(hourly_tail_count),
    overall_tail_count=np.array(overall_tail_count),
    combo_w0_by_hour=combo_w0_by_hour,
    combo_war_by_hour=combo_war_by_hour,
    k_folds=np.array(K_FOLDS),
)
print("통합 결과 저장 완료:", diagnostics_npz_name)
