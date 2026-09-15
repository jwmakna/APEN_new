# -*- coding: utf-8 -*-
# =====================================================================
# 방법6_pc_conditional_ar_mlr_gurobi_figure.py
#
# `방법6_pc_conditional_ar_mlr_gurobi.py`(검증 완료된 Gurobi 변환본)를 그대로
# 복사한 뒤, savefig/CSV 저장 경로에만 "_gurobi" 접미사를 붙인 버전이다.
# Figure 생성 코드 자체는 새로 짠 것이 아니라 원본 `방법6_pc_conditional_ar_
# mlr.py`(scipy) 안에 원래 있던 코드를 그대로 물려받은 것이다 — 이 파일이
# 곧 results/방법6_fig{3,5,6,8}_..._z03_block18.png를 실제로 만드는 스크립트였다
# (다른 별도 Figure 스크립트는 없음, 2026-09-15 확인). 최적화(Gurobi) 부분은
# 이전에 실제 실행·검증까지 마친 그대로 손대지 않았다.
#
# 방법6_pc_conditional_ar_mlr.py
#
# 방법 6 — PC = rate·DA·I[DA>RT] + RT (조건부 가중치 이익식 변형)
#
# 이익식(4항):
#   profit = scale × (DA·x + RT·surplus − RT·shortage − PC·shortage·I[DA>RT])
#   PC = rate·DA,  I[DA>RT] = 1 (DA>RT일 때만), 0 (RT≥DA일 때)
#   → shortage 비용은 DA>RT면 RT+PC, RT≥DA면 RT만 부과된다.
#
# raw 계수 + W2 balance weight(방법2·4·5와 같은 이유로 적용):
#   sc_i = −W1·RT + W2_eff                       (surplus cost)
#   yc_i =  W1·RT + W1·PC·I[DA>RT] + W2_eff       (shortage cost)
#   W2_eff = W2 * W2_BALANCE_WEIGHT
#
# ── 이력 메모 ──
# 이 조건부 가중치(I[DA>RT])는 원래 Elmachtoub & Grigas(2022)의 SPO+
# 프레임워크에서 shortage-cost에 주는 조건부 가중치 아이디어를 차용해
# `방법6_SPO_base_AR_MLR.py`로 구현한 것이다. 그 스크립트는 학습 목적함수를
# "후회(regret) ζ_i에 대한 3개 부등식 + oracle 3후보"로 정식화했었는데,
# oracle_i가 β와 무관한 상수라서 이 regret 최소화는 순수 이익 최대화
# (−profit)와 수학적으로 동일한 최적화 문제임이 확인됐다 — KPI 지점뿐 아니라
# Fig.3/5/6/8 sweep 42개 지점 전부에서 학습된 β·nRMSE·Gap이 완전히 일치했다
# (`방법1_방법6.1_이익식_비교분석.md` §7·§10 참고). 그래서 이 스크립트는
# regret 정식화(ζ 변수, oracle 부등식) 없이 처음부터 순수 이익 최대화로
# 짰다 — `방법6_SPO_base_AR_MLR.py`는 git 이력용으로 남겨두되 더 이상
# 쓰지 않는다. PC=RT+rate·DA(방법2와 완전 동일 이익식)였던 2차 버전
# (`방법6_SPO_plus_AR_MLR.py`)은 방법2의 결과와 소수점까지 일치해 독립된
# 실험으로서 의미가 없어 삭제했다(검증은 위 문서 §7·§10 부록 스크립트가
# 대신한다).
#
# Gap 평가: PC=rate*DA(indicator 없는 원래 4항식), oracle 3후보 {0, actual, 1.0}
#   (rate=0일 때 PC=0이 되어 2후보로 줄이면 Gap이 깨지는 방법4/5와 같은 이유로 3후보 유지)
#
# 보완성 제약: yp·ym = 0 (이진 변수 + big-M)
#
# AR(시간대별 12개 모델) + MLR(pooled 1개 모델) 통합 실행
# 대상: z03/블록18 (2013년 가을) — 방법1~5와 같은 블록으로 맞춰서 논문과 직접 비교 가능하게 함
#
# ── Gurobi 변환 메모 ──
# 이 파일은 `방법6_pc_conditional_ar_mlr.py`의 scipy.optimize.milp/linprog(HiGHS)
# 솔버 부분만 gurobipy로 옮긴 버전이다. 이익식·정규화·데이터·Gap 계산·스윕 범위·
# Figure·저장 파일명은 원본과 완전히 동일하다. Gurobi 변환 스타일은
# `기본코드_(2)_AR_MLR_baseline_proposed_3가지_profit_figure코드/`의 방법1·2·3
# Gurobi 구현(gp.Model, addMVar, addConstr, MVar 슬라이싱 complementarity)을
# 그대로 따랐다. Gurobi가 실패/라이선스 한도에 걸리면 예외를 그대로 올린다 —
# scipy로 자동 대체(fallback)하지 않는다. 원본 파일은 수정하지 않았다.
# =====================================================================

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import csv

# =====================================================================
# 0. 한글 폰트 + matplotlib 스타일
# =====================================================================
korean_font_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic"]
found_font_name = None
for font in fm.fontManager.ttflist:
    if font.name in korean_font_candidates:
        found_font_name = font.name
        break
if found_font_name:
    plt.rcParams["font.family"] = found_font_name
plt.rcParams["axes.unicode_minus"] = False

# ── 일관된 색상 팔레트 ──
C_AR_BASE   = "#4C72B0"   # AR baseline (파랑)
C_AR_PROP   = "#55A868"   # AR 제안모형 (초록)
C_MLR_BASE  = "#CC4654"   # MLR baseline (빨강)
C_MLR_PROP  = "#8CAED6"   # MLR 제안모형 (연파랑)
C_PAPER     = "#888888"   # 논문 참고값 (회색)
C_GRID      = "#DDDDDD"   # 그리드선

# =====================================================================
# 1. 설정 — z03/블록18
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "simulation_output")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21
HOURS_PER_DAY = 12

TRAIN_START = pd.Timestamp("2013-08-25")
TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START  = pd.Timestamp("2013-11-23")
TEST_END    = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0
scale = CAPACITY_MW * DURATION_HOURS

# ── 논문 KPI 조건 ──
KPI_W1 = 1
KPI_W2 = 20
KPI_RATE = 0.5

# ── 그리드 ──
GRID_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 1), (1, 0)]
GRID_PENALTY_RATES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

# ── Fig3/6: W1/W2 10개점 ──
FIG_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1),
                (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
FIG_LABELS = ["AR/MLR", "1/20", "1/10", "1/5", "1/2",
              "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]

# ── Fig5/8: penalty rate 0~100% (논문 Fig.5/8 판독값도 0~100%까지만 있음) ──
FIG_RATES = [round(0.1 * i, 1) for i in range(11)]

# ── W2 균형 보정 Weight — 방법2/4/5와 동일한 값 ──
W2_BALANCE_WEIGHT = 4

# ── 논문 참고값 (z03/블록18 — 방법1~5와 같은 블록이라 직접 비교 가능) ──
PAPER_AR_NRMSE = 34.76
PAPER_AR_GAP = 15.04
PAPER_MLR_NRMSE = 21.76
PAPER_MLR_GAP = 12.59
PAPER_PROP_NRMSE = [34.89, 35.14, 36.28, 41.09, 44.95,
                    46.11, 48.27, 49.21, 49.61, 50.07]
PAPER_PROP_GAP = [13.91, 13.42, 12.71, 11.88, 11.44,
                  11.38, 11.38, 11.36, 11.36, 11.36]
PAPER_PROP_MLR_NRMSE = [21.92, 21.84, 21.75, 21.66, 22.01,
                        23.32, 27.75, 30.62, 35.76, 37.67]
PAPER_PROP_MLR_GAP = [11.91, 11.68, 11.15, 10.65, 10.28,
                      9.91, 9.51, 9.32, 9.27, 9.28]
FIG5_PAPER_AR_NRMSE   = [34.76] * 11
FIG5_PAPER_AR_GAP     = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22]
FIG5_PAPER_PROP_NRMSE = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67]
FIG5_PAPER_PROP_GAP   = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5]
# 논문 원본 Fig.8(a)/(b) 픽셀 추출값 (references/APEN_논문.pdf p.9, 방법2 작업 때 추출)
FIG8_PAPER_MLR_NRMSE  = [21.76] * 11
FIG8_PAPER_PROP_NRMSE = [23.92, 22.56, 21.76, 21.41, 21.67, 22.01, 22.45, 23.14, 23.76, 24.98, 26.10]
FIG8_PAPER_MLR_GAP    = [8.18, 9.05, 9.93, 10.83, 11.71, 12.60, 13.48, 14.37, 15.25, 16.13, 16.98]
FIG8_PAPER_PROP_GAP   = [7.59, 8.53, 9.17, 9.56, 9.94, 10.28, 10.52, 10.68, 10.85, 10.96, 10.93]


# =====================================================================
# 2. 데이터 로딩
# =====================================================================
print("=" * 70)
print("  1. 데이터 로딩")
print("=" * 70)
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & \
              (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START

is_history = daylight_table["local_date"] == HISTORY_DATE
history_rows = daylight_table[is_history].copy().sort_values("hour_idx")
history_solar = np.zeros((1, HOURS_PER_DAY))
for _, row in history_rows.iterrows():
    history_solar[0, row["hour_idx"]] = row["solar_power"]

is_train = (daylight_table["local_date"] >= TRAIN_START) & \
           (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])

is_test = (daylight_table["local_date"] >= TEST_START) & \
          (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])

train_dates = sorted(train_rows["local_date"].unique())
n_train_days = len(train_dates)
test_dates = sorted(test_rows["local_date"].unique())
n_test_days = len(test_dates)

train_solar = np.zeros((n_train_days, HOURS_PER_DAY))
train_da_price = np.zeros((n_train_days, HOURS_PER_DAY))
train_rt_price = np.zeros((n_train_days, HOURS_PER_DAY))
for _, row in train_rows.iterrows():
    d = (row["local_date"] - TRAIN_START).days
    h = row["hour_idx"]
    train_solar[d, h] = row["solar_power"]
    train_da_price[d, h] = row["da_price"]
    train_rt_price[d, h] = row["rt_price"]

test_solar = np.zeros((n_test_days, HOURS_PER_DAY))
test_da_price = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt_price = np.zeros((n_test_days, HOURS_PER_DAY))
for _, row in test_rows.iterrows():
    d = (row["local_date"] - TEST_START).days
    h = row["hour_idx"]
    test_solar[d, h] = row["solar_power"]
    test_da_price[d, h] = row["da_price"]
    test_rt_price[d, h] = row["rt_price"]

n_train_obs = len(train_rows)
mlr_train_solar = train_rows["solar_power"].to_numpy()
mlr_train_dssrd = train_rows["dssrd"].to_numpy()
mlr_train_dtsr = train_rows["dtsr"].to_numpy()
mlr_train_hour = train_rows["hour_idx"].to_numpy(dtype=float)
mlr_train_da = train_rows["da_price"].to_numpy()
mlr_train_rt = train_rows["rt_price"].to_numpy()

n_test_obs = len(test_rows)
actual_flat = test_rows["solar_power"].to_numpy()
da_flat = test_rows["da_price"].to_numpy()
rt_flat = test_rows["rt_price"].to_numpy()

n_features_mlr = 4
X_mlr_train = np.column_stack([np.ones(n_train_obs),
                                mlr_train_dssrd, mlr_train_dtsr, mlr_train_hour])
X_mlr_test = np.column_stack([np.ones(n_test_obs),
                               test_rows["dssrd"].to_numpy(),
                               test_rows["dtsr"].to_numpy(),
                               test_rows["hour_idx"].to_numpy(dtype=float)])

n_features_ar = 13
history_and_train = np.vstack([history_solar, train_solar])
n_ar_rows = n_train_days
ar_intercept = np.ones((n_ar_rows, 1))
ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train[d][::-1]
ar_design = np.hstack([ar_intercept, ar_lag])

actual_flat_all = test_solar.flatten()
da_flat_all = test_da_price.flatten()
rt_flat_all = test_rt_price.flatten()
n_test_obs_all = len(actual_flat_all)

print(f"  train: {n_train_days}일({n_train_obs}행), test: {n_test_days}일({n_test_obs}행)")


# =====================================================================
# 3. Gap 계산 함수 — PC=rate*DA(원래 4항식), oracle 3후보 {0, actual, 1.0}
#    (이 "oracle"은 학습 손실과 무관한 평가 전용 개념 — 방법1~5와 같은 Eq.13식
#     사후 최적 결정 오라클이다. §3.2.6 "왜 구조적으로 무너지지 않는가" 참고)
# =====================================================================
def compute_gap_4term(pred_flat, penalty_rate, actual, da, rt):
    """4항 이익함수(PC=rate*DA) optimality gap 계산 (oracle: {0, actual, 1.0} 3후보)"""
    sum_realized = 0.0
    sum_oracle = 0.0
    for i in range(len(pred_flat)):
        a = actual[i]; x = pred_flat[i]
        dp = da[i]; rp = rt[i]
        pc = penalty_rate * dp  # PC = rate*DA (indicator 없는 원래 정의)
        mismatch = a - x
        surplus = max(mismatch, 0)
        shortage = max(-mismatch, 0)
        realized = scale * (dp * x + rp * surplus - rp * shortage - pc * shortage)
        sum_realized += realized
        p0 = scale * rp * a
        pa = scale * dp * a
        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + rp * s1 - rp * y1 - pc * y1)
        oracle = max(p0, pa, p1)
        sum_oracle += oracle
    return 100.0 * (sum_oracle - sum_realized) / sum_oracle if sum_oracle > 1e-10 else 0.0


def calc_nrmse(pred_flat, actual):
    rmse = np.sqrt(np.mean((actual - pred_flat) ** 2))
    return 100.0 * rmse / np.mean(actual)


def auto_ylim(*series, pad_frac=0.12, round_to=5, floor_at_zero=True):
    """그림마다 실제로 그려지는 값들의 범위에 맞춰 y축을 정한다 (고정된 30~80/0~25 대신)."""
    vals = []
    for s in series:
        vals.extend([v for v in s if v is not None])
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = span * pad_frac if span > 0 else max(hi * 0.1, round_to)
    lo -= pad; hi += pad
    lo = round_to * np.floor(lo / round_to)
    hi = round_to * np.ceil(hi / round_to)
    if floor_at_zero:
        lo = max(lo, 0)
    return float(lo), float(hi)


# =====================================================================
# 4. AR baseline — bounded LAD (시간대별 12개)
# =====================================================================
print("\n" + "=" * 70)
print("  2. AR baseline (bounded LAD, 시간대별)")
print("=" * 70)
ar_coefficients = np.zeros((HOURS_PER_DAY, n_features_ar))
for h in range(HOURS_PER_DAY):
    y_h = train_solar[:, h]

    gmodel = gp.Model(f"baseline_ar_hour_{h}")
    gmodel.Params.OutputFlag = 0

    beta_var = gmodel.addMVar(n_features_ar, lb=-GRB.INFINITY, name="beta")
    u_var = gmodel.addMVar(n_ar_rows, lb=0.0, name="u")

    fitted_expr = ar_design @ beta_var
    gmodel.addConstr(fitted_expr - y_h <= u_var, name="resid_upper")
    gmodel.addConstr(y_h - fitted_expr <= u_var, name="resid_lower")
    gmodel.setObjective(u_var.sum() / n_ar_rows, GRB.MINIMIZE)
    gmodel.optimize()
    if gmodel.Status != GRB.OPTIMAL:
        raise RuntimeError(f"AR baseline LP failed h={h}: Gurobi status={gmodel.Status}")

    ar_coefficients[h] = beta_var.X

ar_test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))
prev_day = train_solar[-1]
for d in range(n_test_days):
    feat = np.concatenate([[1.0], prev_day[::-1]])
    for h in range(HOURS_PER_DAY):
        ar_test_forecast[d, h] = np.clip(np.dot(ar_coefficients[h], feat), 0, 1)
    prev_day = test_solar[d]

ar_pred_flat = ar_test_forecast.flatten()
ar_nrmse = calc_nrmse(ar_pred_flat, actual_flat_all)
print(f"  AR baseline nRMSE = {ar_nrmse:.2f}%")


# =====================================================================
# 5. MLR baseline — bounded LAD (pooled 1개)
# =====================================================================
print("\n" + "=" * 70)
print("  3. MLR baseline (bounded LAD, pooled)")
print("=" * 70)
gmodel_mlr = gp.Model("baseline_mlr")
gmodel_mlr.Params.OutputFlag = 0

beta_var_mlr = gmodel_mlr.addMVar(n_features_mlr, lb=-GRB.INFINITY, name="beta")
u_var_mlr = gmodel_mlr.addMVar(n_train_obs, lb=0.0, name="u")

fitted_expr_mlr = X_mlr_train @ beta_var_mlr
gmodel_mlr.addConstr(fitted_expr_mlr - mlr_train_solar <= u_var_mlr, name="resid_upper")
gmodel_mlr.addConstr(mlr_train_solar - fitted_expr_mlr <= u_var_mlr, name="resid_lower")
gmodel_mlr.setObjective(u_var_mlr.sum() / n_train_obs, GRB.MINIMIZE)
gmodel_mlr.optimize()
if gmodel_mlr.Status != GRB.OPTIMAL:
    raise RuntimeError(f"MLR baseline LP failed: Gurobi status={gmodel_mlr.Status}")

mlr_coefficients = beta_var_mlr.X

mlr_pred_flat = np.clip(X_mlr_test @ mlr_coefficients, 0, 1)
mlr_nrmse = calc_nrmse(mlr_pred_flat, actual_flat)
print(f"  MLR baseline nRMSE = {mlr_nrmse:.2f}%")


# =====================================================================
# 6. 제안모형 MILP — AR (시간대별, raw 계수 + W2 balance weight)
#
# shortage cost: W1·RT + W1·PC·I[DA>RT] + W2_eff   (PC=rate*DA, indicator 조건부)
# surplus  cost: −W1·RT + W2_eff
# =====================================================================
def solve_ar_proposed(penalty_rate, W1, W2):
    """AR 제안모형: PC=rate*DA(I[DA>RT] 조건부), raw 계수 + W2 balance weight"""
    W2_eff = W2 * W2_BALANCE_WEIGHT
    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))

    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]
        da_h = train_da_price[:, hour]
        rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows

        pc_h = penalty_rate * da_h                    # PC = rate*DA (raw)
        indicator = (da_h > rt_h).astype(float)        # I[DA>RT]

        sc = -W1 * rt_h + W2_eff                        # surplus cost
        yc = W1 * rt_h + W1 * pc_h * indicator + W2_eff  # shortage cost

        bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
        bin_arr = np.array(bin_list, dtype=int)
        n_bin = len(bin_arr)

        gmodel = gp.Model(f"proposed_ar_hour_{hour}")
        gmodel.Params.OutputFlag = 0
        gmodel.Params.MIPGap = 1e-9

        beta_var = gmodel.addMVar(n_features_ar, lb=-GRB.INFINITY, name="beta")
        x_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="x")
        yplus_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="y_plus")
        yminus_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="y_minus")

        gmodel.addConstr(x_var - ar_design @ beta_var == 0.0, name="commitment_eq")
        gmodel.addConstr(x_var + yplus_var - yminus_var == y_h, name="mismatch_eq")

        if n_bin > 0:
            z_var = gmodel.addMVar(n_bin, vtype=GRB.BINARY, name="z")
            gmodel.addConstr(yplus_var[bin_arr] + z_var <= 1.0, name="complementarity_plus")
            gmodel.addConstr(yminus_var[bin_arr] - z_var <= 0.0, name="complementarity_minus")

        objective_expr = (-W1 * da_h) @ x_var + sc @ yplus_var + yc @ yminus_var
        gmodel.setObjective(objective_expr, GRB.MINIMIZE)
        gmodel.optimize()
        if gmodel.Status != GRB.OPTIMAL:
            print(f"  AR MILP 실패 h={hour}: Gurobi status={gmodel.Status}")
            coeffs[hour] = 0.0
        else:
            coeffs[hour] = beta_var.X

    fc = np.zeros((n_test_days, HOURS_PER_DAY))
    prev = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY):
            fc[d, h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = test_solar[d]
    return fc.flatten()


# =====================================================================
# 7. 제안모형 MILP — MLR (pooled, raw 계수 + W2 balance weight)
# =====================================================================
def solve_mlr_proposed(penalty_rate, W1, W2):
    """MLR 제안모형: PC=rate*DA(I[DA>RT] 조건부), raw 계수 + W2 balance weight"""
    W2_eff = W2 * W2_BALANCE_WEIGHT
    n_obs = n_train_obs

    pc_all = penalty_rate * mlr_train_da
    indicator = (mlr_train_da > mlr_train_rt).astype(float)

    sc = -W1 * mlr_train_rt + W2_eff
    yc = W1 * mlr_train_rt + W1 * pc_all * indicator + W2_eff

    bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
    bin_arr = np.array(bin_list, dtype=int)
    n_bin = len(bin_arr)

    gmodel = gp.Model("proposed_mlr")
    gmodel.Params.OutputFlag = 0
    gmodel.Params.MIPGap = 1e-9

    beta_var = gmodel.addMVar(n_features_mlr, lb=-GRB.INFINITY, name="beta")
    x_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="x")
    yplus_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="y_plus")
    yminus_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="y_minus")

    gmodel.addConstr(x_var - X_mlr_train @ beta_var == 0.0, name="commitment_eq")
    gmodel.addConstr(x_var + yplus_var - yminus_var == mlr_train_solar, name="mismatch_eq")

    if n_bin > 0:
        z_var = gmodel.addMVar(n_bin, vtype=GRB.BINARY, name="z")
        gmodel.addConstr(yplus_var[bin_arr] + z_var <= 1.0, name="complementarity_plus")
        gmodel.addConstr(yminus_var[bin_arr] - z_var <= 0.0, name="complementarity_minus")

    objective_expr = (-W1 * mlr_train_da) @ x_var + sc @ yplus_var + yc @ yminus_var
    gmodel.setObjective(objective_expr, GRB.MINIMIZE)
    gmodel.optimize()
    if gmodel.Status != GRB.OPTIMAL:
        raise RuntimeError(f"MLR MILP 실패: Gurobi status={gmodel.Status}")
    coeffs = beta_var.X
    return np.clip(X_mlr_test @ coeffs, 0, 1)


# =====================================================================
# 8. 캐시
# =====================================================================
cache_ar = {}
cache_mlr = {}

def get_ar(pr, w1, w2):
    k = (pr, w1, w2)
    if k not in cache_ar:
        pred = solve_ar_proposed(pr, w1, w2)
        cache_ar[k] = (calc_nrmse(pred, actual_flat_all),
                       compute_gap_4term(pred, pr, actual_flat_all, da_flat_all, rt_flat_all))
    return cache_ar[k]

def get_mlr(pr, w1, w2):
    k = (pr, w1, w2)
    if k not in cache_mlr:
        pred = solve_mlr_proposed(pr, w1, w2)
        cache_mlr[k] = (calc_nrmse(pred, actual_flat),
                        compute_gap_4term(pred, pr, actual_flat, da_flat, rt_flat))
    return cache_mlr[k]


# =====================================================================
# 9. 전체 그리드 스윕 (50조합) — AR + MLR
# =====================================================================
print("\n" + "=" * 70)
print("  4. 그리드 스윕 (rate 10개 x W1/W2 5개 = 50조합, 방법6 AR+MLR)")
print("=" * 70)

grid_results = []

for pr in GRID_PENALTY_RATES:
    print(f"\n  ── rate = {pr} ──")
    for W1, W2 in GRID_W_RATIOS:
        a_n, a_g = get_ar(pr, W1, W2)
        m_n, m_g = get_mlr(pr, W1, W2)
        grid_results.append((pr, W1, W2, a_n, a_g, m_n, m_g))
        label = f"{W1}/{W2}"
        print(f"  {label}: AR(nRMSE={a_n:.2f}%, Gap={a_g:.2f}%) "
              f"MLR(nRMSE={m_n:.2f}%, Gap={m_g:.2f}%)")

grid_csv = os.path.join(OUT_DIR, "grid_method6_ar_mlr_z03_block18_gurobi.csv")
with open(grid_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rate", "W1", "W2",
                "AR_nRMSE", "AR_Gap", "MLR_nRMSE", "MLR_Gap"])
    for row in grid_results:
        w.writerow([row[0], row[1], row[2],
                     round(row[3], 2), round(row[4], 2),
                     round(row[5], 2), round(row[6], 2)])
print(f"\n  saved: {grid_csv}")


# =====================================================================
# 10. 논문 KPI 조건 (W1=1, W2=20, rate=50%) 별도 출력
# =====================================================================
print("\n" + "=" * 70)
print(f"  5. 논문 KPI 조건 — W1={KPI_W1}, W2={KPI_W2}, penalty rate={KPI_RATE} (50%)")
print("=" * 70)

ar_kpi_n, ar_kpi_g = get_ar(KPI_RATE, KPI_W1, KPI_W2)
mlr_kpi_n, mlr_kpi_g = get_mlr(KPI_RATE, KPI_W1, KPI_W2)
print(f"""
  ┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
  │     모델          │  nRMSE (%)   │  Gap (%)     │  nRMSE (%)   │  Gap (%)     │
  │                  │  (제안모형)   │  (제안모형)   │  (논문 기준)  │  (논문 기준)  │
  ├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
  │  AR (z03/블록18)  │ {ar_kpi_n:>10.2f}   │ {ar_kpi_g:>10.2f}   │ {PAPER_AR_NRMSE:>10.2f}   │ {PAPER_AR_GAP:>10.2f}   │
  │  MLR (z03/블록18) │ {mlr_kpi_n:>10.2f}   │ {mlr_kpi_g:>10.2f}   │ {PAPER_MLR_NRMSE:>10.2f}   │ {PAPER_MLR_GAP:>10.2f}   │
  └──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

  ※ 논문 KPI 기준값 (z03/블록18):
      AR  nRMSE={PAPER_AR_NRMSE}%, Gap={PAPER_AR_GAP}%
      MLR nRMSE={PAPER_MLR_NRMSE}%, Gap={PAPER_MLR_GAP}%
""")


# =====================================================================
# 11. Fig.3/6 데이터 — W1/W2 10개점 스윕 (rate=0.5 고정)
# =====================================================================
FIG36_RATE = KPI_RATE
print("\n" + "=" * 70)
print(f"  6. Fig.3/6 데이터 — W1/W2 10개점 스윕 (rate={FIG36_RATE})")
print("=" * 70)

fig3_ar_n = [ar_nrmse]
fig3_ar_g = [compute_gap_4term(ar_pred_flat, FIG36_RATE,
                                actual_flat_all, da_flat_all, rt_flat_all)]
fig6_mlr_n = [mlr_nrmse]
fig6_mlr_g = [compute_gap_4term(mlr_pred_flat, FIG36_RATE,
                                 actual_flat, da_flat, rt_flat)]

for W1, W2 in FIG_W_RATIOS:
    a_n, a_g = get_ar(FIG36_RATE, W1, W2)
    m_n, m_g = get_mlr(FIG36_RATE, W1, W2)
    fig3_ar_n.append(a_n); fig3_ar_g.append(a_g)
    fig6_mlr_n.append(m_n); fig6_mlr_g.append(m_g)
    print(f"  {W1}/{W2}: AR(nRMSE={a_n:.2f}%, Gap={a_g:.2f}%) "
          f"MLR(nRMSE={m_n:.2f}%, Gap={m_g:.2f}%)")


# =====================================================================
# 12. Fig.5/8 데이터 — rate 0~100% 스윕 (W1=W2=1, W2_eff=W2*W2_BALANCE_WEIGHT)
# =====================================================================
print("\n" + "=" * 70)
print(f"  7. Fig.5/8 데이터 — rate 0~100% 스윕 (W1=W2=1, W2_eff=W2×{W2_BALANCE_WEIGHT} 균형 보정)")
print("=" * 70)

fig5_ar_n_list = []; fig5_ar_g_list = []
fig8_mlr_n_list = []; fig8_mlr_g_list = []
fig5_ar_prop_n = []; fig5_ar_prop_g = []
fig8_mlr_prop_n = []; fig8_mlr_prop_g = []

for rate in FIG_RATES:
    ar_g_r = compute_gap_4term(ar_pred_flat, rate,
                                actual_flat_all, da_flat_all, rt_flat_all)
    mlr_g_r = compute_gap_4term(mlr_pred_flat, rate,
                                 actual_flat, da_flat, rt_flat)
    fig5_ar_n_list.append(ar_nrmse); fig5_ar_g_list.append(ar_g_r)
    fig8_mlr_n_list.append(mlr_nrmse); fig8_mlr_g_list.append(mlr_g_r)

    a_n, a_g = get_ar(rate, 1, 1)
    m_n, m_g = get_mlr(rate, 1, 1)
    fig5_ar_prop_n.append(a_n); fig5_ar_prop_g.append(a_g)
    fig8_mlr_prop_n.append(m_n); fig8_mlr_prop_g.append(m_g)

    print(f"  rate={rate:.1f}: "
          f"AR(base={ar_g_r:.2f}%, prop={a_n:.2f}%/{a_g:.2f}%) "
          f"MLR(base={mlr_g_r:.2f}%, prop={m_n:.2f}%/{m_g:.2f}%)")


# =====================================================================
# 13. CSV 저장
# =====================================================================
def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)
        for row in rows:
            csv.writer(f).writerow(row)
    print(f"  saved: {path}")

save_csv(os.path.join(OUT_DIR, "fig3_method6_AR_gurobi.csv"),
         ["Label", "nRMSE", "Gap"],
         [[FIG_LABELS[i], round(fig3_ar_n[i], 2), round(fig3_ar_g[i], 2)]
          for i in range(len(FIG_LABELS))])

save_csv(os.path.join(OUT_DIR, "fig5_method6_AR_gurobi.csv"),
         ["Rate", "AR_base_nRMSE", "AR_base_Gap", "AR_prop_nRMSE", "AR_prop_Gap"],
         [[FIG_RATES[i], round(fig5_ar_n_list[i], 2), round(fig5_ar_g_list[i], 2),
           round(fig5_ar_prop_n[i], 2), round(fig5_ar_prop_g[i], 2)]
          for i in range(len(FIG_RATES))])

save_csv(os.path.join(OUT_DIR, "fig6_method6_MLR_gurobi.csv"),
         ["Label", "nRMSE", "Gap"],
         [[FIG_LABELS[i], round(fig6_mlr_n[i], 2), round(fig6_mlr_g[i], 2)]
          for i in range(len(FIG_LABELS))])

save_csv(os.path.join(OUT_DIR, "fig8_method6_MLR_gurobi.csv"),
         ["Rate", "MLR_base_nRMSE", "MLR_base_Gap", "MLR_prop_nRMSE", "MLR_prop_Gap"],
         [[FIG_RATES[i], round(fig8_mlr_n_list[i], 2), round(fig8_mlr_g_list[i], 2),
           round(fig8_mlr_prop_n[i], 2), round(fig8_mlr_prop_g[i], 2)]
          for i in range(len(FIG_RATES))])


# =====================================================================
# 14. Fig.3 — AR, dual y축 (nRMSE 좌, Gap 우)
# =====================================================================
print("\n" + "=" * 70)
print("  8. Fig.3 plotting (AR, W1/W2 스윕, dual y축)")
print("=" * 70)

x = list(range(len(FIG_LABELS)))
fig, ax_l = plt.subplots(figsize=(10, 6))
fig.suptitle(f"방법6_fig3 — AR PC=rate×DA·I[DA>RT]+RT, rate={FIG36_RATE}, W1/W2 스윕", fontsize=13)
ax_r = ax_l.twinx()

paper_nrmse_all = [PAPER_AR_NRMSE] + PAPER_PROP_NRMSE
paper_gap_all = [PAPER_AR_GAP] + PAPER_PROP_GAP
ln1 = ax_l.plot(x, paper_nrmse_all, marker="o", color=C_PAPER,
                linewidth=2, linestyle=":", label="논문 nRMSE", alpha=0.8)
ln2 = ax_l.plot(x, fig3_ar_n,   marker="s", color=C_AR_PROP,
                linewidth=2, linestyle="-", label="재현 nRMSE")
ln3 = ax_r.plot(x, paper_gap_all, marker="^", color=C_PAPER,
                linewidth=2, linestyle=":", label="논문 Gap", alpha=0.8)
ln4 = ax_r.plot(x, fig3_ar_g,   marker="d", color=C_AR_BASE,
                linewidth=2, linestyle="-", label="재현 Gap")

ax_l.set_xticks(x); ax_l.set_xticklabels(FIG_LABELS)
ax_l.set_xlabel("W1/W2"); ax_l.set_ylabel("nRMSE (%)", color=C_AR_PROP)
ax_r.set_ylabel("Optimality Gap (%)", color=C_AR_BASE)
ax_l.tick_params(axis="y", labelcolor=C_AR_PROP)
ax_r.tick_params(axis="y", labelcolor=C_AR_BASE)
ax_l.grid(True, alpha=0.3)
ax_l.set_ylim(*auto_ylim(paper_nrmse_all, fig3_ar_n))
ax_r.set_ylim(*auto_ylim(paper_gap_all, fig3_ar_g))
all_ln = ln1+ln2+ln3+ln4
ax_l.legend(all_ln, [l.get_label() for l in all_ln],
            loc="upper left", fontsize=9)
fig.tight_layout()
p3 = os.path.join(RESULTS_DIR, "방법6_fig3_ar_z03_block18_gurobi.png")
fig.savefig(p3, dpi=150); plt.close(fig)
print(f"  saved: {p3}")


# =====================================================================
# 15. Fig.5 — AR, rate 스윕 (nRMSE / Gap 2분할)
# =====================================================================
print("\n" + "=" * 70)
print("  9. Fig.5 plotting (AR, rate 스윕, 2분할)")
print("=" * 70)

x5 = list(range(len(FIG_RATES)))
lbl5 = [f"{int(r*100)}%" for r in FIG_RATES]
x5_paper = x5[:11]

fig5_fig, (ax5n, ax5g) = plt.subplots(1, 2, figsize=(13, 5))
fig5_fig.suptitle(f"방법6_fig5 — AR PC=rate×DA·I[DA>RT]+RT, W1=W2=1, penalty rate 스윕", fontsize=13)

ax5n.plot(x5_paper, FIG5_PAPER_AR_NRMSE, linestyle="--", color=C_PAPER, alpha=0.8,
          marker="o", markersize=4, label="논문 AR")
ax5n.plot(x5_paper, FIG5_PAPER_PROP_NRMSE, linestyle="--", color=C_AR_PROP, alpha=0.5,
          marker="s", markersize=4, label="논문 제안모형")
ax5n.plot(x5, fig5_ar_n_list, marker="o", color=C_AR_BASE, linewidth=2, linestyle="-", label="재현 AR")
ax5n.plot(x5, fig5_ar_prop_n, marker="s", color=C_AR_PROP, linewidth=2, linestyle="-", label="재현 제안모형")
ax5n.set_xticks(x5); ax5n.set_xticklabels(lbl5, rotation=45)
ax5n.set_xlabel("벌금비용률"); ax5n.set_ylabel("nRMSE (%)")
ax5n.set_title("nRMSE"); ax5n.grid(True, alpha=0.3); ax5n.legend(fontsize=8)
ax5n.set_ylim(*auto_ylim(FIG5_PAPER_AR_NRMSE, FIG5_PAPER_PROP_NRMSE, fig5_ar_n_list, fig5_ar_prop_n))

ax5g.plot(x5_paper, FIG5_PAPER_AR_GAP, linestyle="--", color=C_PAPER, alpha=0.8,
          marker="o", markersize=4, label="논문 AR")
ax5g.plot(x5_paper, FIG5_PAPER_PROP_GAP, linestyle="--", color=C_AR_PROP, alpha=0.5,
          marker="s", markersize=4, label="논문 제안모형")
ax5g.plot(x5, fig5_ar_g_list, marker="o", color=C_AR_BASE, linewidth=2, linestyle="-", label="재현 AR")
ax5g.plot(x5, fig5_ar_prop_g, marker="s", color=C_AR_PROP, linewidth=2, linestyle="-", label="재현 제안모형")
hl = FIG_RATES.index(KPI_RATE) if KPI_RATE in FIG_RATES else None
if hl is not None:
    ax5g.axvline(hl, color=C_GRID, linestyle=":", linewidth=1.5,
                 label=f"KPI rate={int(KPI_RATE*100)}%")
ax5g.set_xticks(x5); ax5g.set_xticklabels(lbl5, rotation=45)
ax5g.set_xlabel("벌금비용률"); ax5g.set_ylabel("Optimality Gap (%)")
ax5g.set_title("Optimality Gap"); ax5g.grid(True, alpha=0.3); ax5g.legend(fontsize=8)
ax5g.set_ylim(*auto_ylim(FIG5_PAPER_AR_GAP, FIG5_PAPER_PROP_GAP, fig5_ar_g_list, fig5_ar_prop_g))

fig5_fig.tight_layout()
p5 = os.path.join(RESULTS_DIR, "방법6_fig5_ar_z03_block18_gurobi.png")
fig5_fig.savefig(p5, dpi=150); plt.close(fig5_fig)
print(f"  saved: {p5}")


# =====================================================================
# 16. Fig.6 — MLR, dual y축 (nRMSE 좌, Gap 우)
# =====================================================================
print("\n" + "=" * 70)
print("  10. Fig.6 plotting (MLR, W1/W2 스윕, dual y축)")
print("=" * 70)

fig6_fig, ax_l = plt.subplots(figsize=(10, 6))
fig6_fig.suptitle(f"방법6_fig6 — MLR PC=rate×DA·I[DA>RT]+RT, rate={FIG36_RATE}, W1/W2 스윕", fontsize=13)
ax_r = ax_l.twinx()

paper_mlr_nrmse_all = [PAPER_MLR_NRMSE] + PAPER_PROP_MLR_NRMSE
paper_mlr_gap_all = [PAPER_MLR_GAP] + PAPER_PROP_MLR_GAP
ln1 = ax_l.plot(x, paper_mlr_nrmse_all, marker="o", color=C_PAPER,
                linewidth=2, linestyle=":", label="논문 nRMSE", alpha=0.8)
ln2 = ax_l.plot(x, fig6_mlr_n, marker="s", color=C_MLR_PROP,
                linewidth=2, linestyle="-", label="재현 nRMSE")
ln3 = ax_r.plot(x, paper_mlr_gap_all, marker="^", color=C_PAPER,
                linewidth=2, linestyle=":", label="논문 Gap", alpha=0.8)
ln4 = ax_r.plot(x, fig6_mlr_g, marker="d", color=C_MLR_BASE,
                linewidth=2, linestyle="-", label="재현 Gap")

ax_l.set_xticks(x); ax_l.set_xticklabels(FIG_LABELS)
ax_l.set_xlabel("W1/W2"); ax_l.set_ylabel("nRMSE (%)", color=C_MLR_PROP)
ax_r.set_ylabel("Optimality Gap (%)", color=C_MLR_BASE)
ax_l.tick_params(axis="y", labelcolor=C_MLR_PROP)
ax_r.tick_params(axis="y", labelcolor=C_MLR_BASE)
ax_l.grid(True, alpha=0.3)
ax_l.set_ylim(*auto_ylim(paper_mlr_nrmse_all, fig6_mlr_n))
ax_r.set_ylim(*auto_ylim(paper_mlr_gap_all, fig6_mlr_g))
all_ln = ln1+ln2+ln3+ln4
ax_l.legend(all_ln, [l.get_label() for l in all_ln],
            loc="upper left", fontsize=9)
fig6_fig.tight_layout()
p6 = os.path.join(RESULTS_DIR, "방법6_fig6_mlr_z03_block18_gurobi.png")
fig6_fig.savefig(p6, dpi=150); plt.close(fig6_fig)
print(f"  saved: {p6}")


# =====================================================================
# 17. Fig.8 — MLR, rate 스윕 (nRMSE / Gap 2분할)
# =====================================================================
print("\n" + "=" * 70)
print("  11. Fig.8 plotting (MLR, rate 스윕, 2분할)")
print("=" * 70)

fig8_fig, (ax8n, ax8g) = plt.subplots(1, 2, figsize=(13, 5))
fig8_fig.suptitle(f"방법6_fig8 — MLR PC=rate×DA·I[DA>RT]+RT, W1=W2=1, penalty rate 스윕", fontsize=13)

ax8n.plot(x5_paper, FIG8_PAPER_MLR_NRMSE, linestyle="--", color=C_PAPER, alpha=0.8,
          marker="o", markersize=4, label="논문 MLR")
ax8n.plot(x5_paper, FIG8_PAPER_PROP_NRMSE, linestyle="--", color=C_MLR_PROP, alpha=0.5,
          marker="s", markersize=4, label="논문 제안모형")
ax8n.plot(x5, fig8_mlr_n_list, marker="o", color=C_MLR_BASE, linewidth=2, linestyle="-", label="재현 MLR")
ax8n.plot(x5, fig8_mlr_prop_n, marker="s", color=C_MLR_PROP, linewidth=2, linestyle="-", label="재현 제안모형")
ax8n.set_xticks(x5); ax8n.set_xticklabels(lbl5, rotation=45)
ax8n.set_xlabel("벌금비용률"); ax8n.set_ylabel("nRMSE (%)")
ax8n.set_title("nRMSE"); ax8n.grid(True, alpha=0.3); ax8n.legend(fontsize=8)
ax8n.set_ylim(*auto_ylim(FIG8_PAPER_MLR_NRMSE, FIG8_PAPER_PROP_NRMSE, fig8_mlr_n_list, fig8_mlr_prop_n))

ax8g.plot(x5_paper, FIG8_PAPER_MLR_GAP, linestyle="--", color=C_PAPER, alpha=0.8,
          marker="o", markersize=4, label="논문 MLR")
ax8g.plot(x5_paper, FIG8_PAPER_PROP_GAP, linestyle="--", color=C_MLR_PROP, alpha=0.5,
          marker="s", markersize=4, label="논문 제안모형")
ax8g.plot(x5, fig8_mlr_g_list, marker="o", color=C_MLR_BASE, linewidth=2, linestyle="-", label="재현 MLR")
ax8g.plot(x5, fig8_mlr_prop_g, marker="s", color=C_MLR_PROP, linewidth=2, linestyle="-", label="재현 제안모형")
if hl is not None:
    ax8g.axvline(hl, color=C_GRID, linestyle=":", linewidth=1.5,
                 label=f"KPI rate={int(KPI_RATE*100)}%")
ax8g.set_xticks(x5); ax8g.set_xticklabels(lbl5, rotation=45)
ax8g.set_xlabel("벌금비용률"); ax8g.set_ylabel("Optimality Gap (%)")
ax8g.set_title("Optimality Gap"); ax8g.grid(True, alpha=0.3); ax8g.legend(fontsize=8)
ax8g.set_ylim(*auto_ylim(FIG8_PAPER_MLR_GAP, FIG8_PAPER_PROP_GAP, fig8_mlr_g_list, fig8_mlr_prop_g))

fig8_fig.tight_layout()
p8 = os.path.join(RESULTS_DIR, "방법6_fig8_mlr_z03_block18_gurobi.png")
fig8_fig.savefig(p8, dpi=150); plt.close(fig8_fig)
print(f"  saved: {p8}")


# =====================================================================
# 18. 전체 결과 요약
# =====================================================================
print("\n" + "=" * 70)
print("  완료 — 방법6 PC=rate×DA·I[DA>RT]+RT 통합 실험 (AR+MLR, z03/블록18)")
print("=" * 70)
print(f"""
  생성 파일:
    Fig.3 (AR W1/W2):   {p3}
    Fig.5 (AR rate):    {p5}
    Fig.6 (MLR W1/W2):  {p6}
    Fig.8 (MLR rate):   {p8}
    Grid CSV:           {grid_csv}

  ── KPI 조건 (W1=1, W2=20, rate=50%) ──
    AR  제안모형: nRMSE={ar_kpi_n:.2f}%, Gap={ar_kpi_g:.2f}%
    MLR 제안모형: nRMSE={mlr_kpi_n:.2f}%, Gap={mlr_kpi_g:.2f}%
    AR  논문 기준: nRMSE={PAPER_AR_NRMSE:.2f}%, Gap={PAPER_AR_GAP:.2f}%
    MLR 논문 기준: nRMSE={PAPER_MLR_NRMSE:.2f}%, Gap={PAPER_MLR_GAP:.2f}%
""")
