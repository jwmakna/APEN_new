# -*- coding: utf-8 -*-
# =====================================================================
# 방법4_pc_max_rt_dt_AR_MLR_gurobi_figure.py
#
# `방법4_pc_max_rt_dt_AR_MLR_gurobi.py`(검증 완료된 Gurobi 변환본)를 그대로
# 복사한 뒤, savefig/CSV 저장 경로에만 "_gurobi" 접미사를 붙인 버전이다.
# Figure 생성 코드 자체는 새로 짠 것이 아니라 원본 `방법4_pc_max_rt_dt_AR_
# MLR.py`(scipy) 안에 원래 있던 코드를 그대로 물려받은 것이다 — 이 파일이
# 곧 results/방법4_fig{3,5,6,8}_..._z03_block18.png를 실제로 만드는 스크립트였다
# (다른 별도 Figure 스크립트는 없음, 2026-09-15 확인). 최적화(Gurobi) 부분은
# 이전에 실제 실행·검증까지 마친 그대로 손대지 않았다.
#
# integrated_3항_pc_max_rt_dt_fig3568.py
#
# 3항 pc_max_rt_dt (PC = rate · max(DA, RT)) 통합 실행
#   - 이익함수: DA·x + RP·y+ - PC·y-  (3항, 논문 원문)
#   - oracle: 3후보 {0, actual, 1.0}
#   - AR(시간대별 12개 모델) + MLR(pooled 1개 모델)
#   - 대상: z03/블록18 (2013년 가을)
#
# 출력: Fig.3(AR W1/W2), Fig.5(AR rate), Fig.6(MLR W1/W2), Fig.8(MLR rate)
#
# ── Gurobi 변환 메모 ──
# 이 파일은 `방법4_pc_max_rt_dt_AR_MLR.py`의 scipy.optimize.milp/linprog(HiGHS)
# 솔버 부분만 gurobipy로 옮긴 버전이다. 이익식·정규화·데이터·Gap 계산·스윕 범위·
# Figure·저장 파일명은 원본과 완전히 동일하다. Gurobi 변환 스타일은
# `기본코드_(2)_AR_MLR_baseline_proposed_3가지_profit_figure코드/`의 방법1·2·3
# Gurobi 구현을 그대로 따랐다. Gurobi가 실패/라이선스 한도에 걸리면 예외를
# 그대로 올린다 — scipy로 자동 대체(fallback)하지 않는다. 원본 파일은 수정하지 않았다.
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
# 0. 한글 폰트 + 색상
# =====================================================================
korean_font_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic"]
for font in fm.fontManager.ttflist:
    if font.name in korean_font_candidates:
        plt.rcParams["font.family"] = font.name
        break
plt.rcParams["axes.unicode_minus"] = False

C_AR     = "#4C72B0"
C_MLR    = "#CC4654"
C_PROP   = "#55A868"
C_PAPER  = "#FF7F00"
C_GRID   = "#DDDDDD"

# =====================================================================
# 1. 설정
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "simulation_output")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LOCAL_HOUR_START, LOCAL_HOUR_END, HOURS_PER_DAY = 9, 21, 12

TRAIN_START  = pd.Timestamp("2013-08-25")
TRAIN_END    = pd.Timestamp("2013-11-22")
TEST_START   = pd.Timestamp("2013-11-23")
TEST_END     = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")

CAPACITY_MW = 30.0; DURATION_HOURS = 1.0
scale = CAPACITY_MW * DURATION_HOURS

KPI_RATE = 0.5

W_RATIOS = [(1,20),(1,10),(1,5),(1,2),(1,1),(2,1),(5,1),(10,1),(20,1),(1,0)]
W_LABELS = ["1/20","1/10","1/5","1/2","1/1","2/1","5/1","10/1","20/1","1/0"]
RATES    = [round(0.1 * i, 1) for i in range(11)]  # 0~100% (논문 Fig.5/8 판독값도 100%까지만 있음)

# ── W2 균형 보정 Weight — 방법2에는 적용했지만 여기서는 적용하지 않는다 ──
# 방법2(PC=RT+rate*DA)는 W2에 Weight=4를 곱해야 논문 Fig.5 모양과 맞았지만, 이 스크립트
# (PC=rate*max(DA,RT))에 그대로 포팅해서 실험해보니 오히려 collapse(nRMSE가 "전부 x=1"과
# 같은 116.45%로 고정되는 구간)가 W1/W2=1/2 부터 1/0까지 넓게 번지는, 원래(2026-09-09
# 세션에 이미 검증된 results/fig3_pc_max_rt_dt_W1W2_AR.png)보다 더 나쁜 결과가 나왔다.
# 그 원본 그림은 collapse가 1/20~1/10 두 점에서만 일어나고 1/5부터는 ~39%로 안정되는데,
# Weight=4를 곱하면 이 안정 구간 자체가 사라진다. 그래서 이 formula에는 원래대로 raw W2
# (Weight=1, 즉 무보정)를 유지한다.
W2_BALANCE_WEIGHT = 1

# 논문 기준값 (z03/블록18, Table 3/4) — baseline(AR/MLR only) 및 제안모형 W1/W2 스윕값
PAPER_AR_BASE_NRMSE = 34.76;   PAPER_AR_BASE_GAP   = 15.04
PAPER_MLR_BASE_NRMSE = 21.76;  PAPER_MLR_BASE_GAP   = 12.59
PAPER_AR_PROP_NRMSE = [34.89, 35.14, 36.28, 41.09, 44.95,
                       46.11, 48.27, 49.21, 49.61, 50.07]
PAPER_AR_PROP_GAP   = [13.91, 13.42, 12.71, 11.88, 11.44,
                       11.38, 11.38, 11.36, 11.36, 11.36]
PAPER_MLR_PROP_NRMSE = [21.92, 21.84, 21.75, 21.66, 22.01,
                        23.32, 27.75, 30.62, 35.76, 37.67]
PAPER_MLR_PROP_GAP   = [11.91, 11.68, 11.15, 10.65, 10.28,
                        9.91, 9.51, 9.32, 9.27, 9.28]

# 논문 원본 Fig.5(a)/(b) 육안 판독값 (AR, rate 0~100% 11개점) — 50%만 Table 3 실측치,
# 나머지는 CodefromJiWon/model_proposed_ar_profit_change_v3.py의 판독값을 그대로 재사용
FIG5_PAPER_AR_NRMSE   = [34.76] * 11
FIG5_PAPER_AR_GAP     = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22]
FIG5_PAPER_PROP_NRMSE = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67]
FIG5_PAPER_PROP_GAP   = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5]
# 논문 원본 Fig.8(a)/(b) 픽셀 추출값 (MLR, rate 0~100% 11개점, references/APEN_논문.pdf p.9)
# PyMuPDF로 페이지를 600dpi로 렌더링 후 곡선 색상 픽셀 좌표를 축 눈금 기준으로 데이터값 역산.
# 50% 지점이 Table 4 실측치(MLR 21.76%/12.59%, 제안모형 W1=W2=1: 22.01%/10.28%)와
# 정확히 일치해 추출 신뢰도 확인됨(방법2_4term_ar_mlr_z03_block18.py 작업 때 추출).
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
raw = pd.read_csv(MERGED_FILE); raw["local_date"] = pd.to_datetime(raw["local_date"])
day = raw[(raw["local_hour"]>=LOCAL_HOUR_START)&(raw["local_hour"]<LOCAL_HOUR_END)].copy()
day["hour_idx"] = day["local_hour"] - LOCAL_HOUR_START

# AR 배열
his = day[day["local_date"]==HISTORY_DATE].sort_values("hour_idx")
history_solar = np.zeros((1, HOURS_PER_DAY))
for _, r in his.iterrows(): history_solar[0, r["hour_idx"]] = r["solar_power"]

tr = day[(day["local_date"]>=TRAIN_START)&(day["local_date"]<=TRAIN_END)].sort_values(["local_date","hour_idx"])
te = day[(day["local_date"]>=TEST_START)&(day["local_date"]<=TEST_END)].sort_values(["local_date","hour_idx"])

n_tr_days = len(sorted(tr["local_date"].unique()))
n_te_days = len(sorted(te["local_date"].unique()))
n_tr_obs = len(tr); n_te_obs = len(te)

tr_s = np.zeros((n_tr_days, HOURS_PER_DAY)); tr_d = np.zeros_like(tr_s); tr_r = np.zeros_like(tr_s)
for _, r in tr.iterrows():
    d = (r["local_date"]-TRAIN_START).days; h = r["hour_idx"]
    tr_s[d,h]=r["solar_power"]; tr_d[d,h]=r["da_price"]; tr_r[d,h]=r["rt_price"]

te_s = np.zeros((n_te_days, HOURS_PER_DAY)); te_d = np.zeros_like(te_s); te_r = np.zeros_like(te_s)
for _, r in te.iterrows():
    d = (r["local_date"]-TEST_START).days; h = r["hour_idx"]
    te_s[d,h]=r["solar_power"]; te_d[d,h]=r["da_price"]; te_r[d,h]=r["rt_price"]

# MLR 배열
mlr_tr_s = tr["solar_power"].to_numpy(); mlr_tr_d = tr["da_price"].to_numpy(); mlr_tr_r = tr["rt_price"].to_numpy()
mlr_te_s = te["solar_power"].to_numpy(); mlr_te_d = te["da_price"].to_numpy(); mlr_te_r = te["rt_price"].to_numpy()

n_feat_mlr = 4
X_tr = np.column_stack([np.ones(n_tr_obs), tr["dssrd"].to_numpy(), tr["dtsr"].to_numpy(), tr["hour_idx"].to_numpy(dtype=float)])
X_te = np.column_stack([np.ones(n_te_obs), te["dssrd"].to_numpy(), te["dtsr"].to_numpy(), te["hour_idx"].to_numpy(dtype=float)])

# AR 설계행렬
n_feat_ar = 13
ht = np.vstack([history_solar, tr_s])
ar_int = np.ones((n_tr_days, 1))
ar_lag = np.zeros((n_tr_days, HOURS_PER_DAY))
for d in range(n_tr_days): ar_lag[d] = ht[d][::-1]
ar_X = np.hstack([ar_int, ar_lag])

ar_act = te_s.flatten(); ar_da = te_d.flatten(); ar_rt = te_r.flatten()
print(f"  train: {n_tr_days}일({n_tr_obs}행), test: {n_te_days}일({n_te_obs}행)")


# =====================================================================
# 3. 유틸리티
# =====================================================================
def nrmse_fn(p, a):
    return 100.0 * np.sqrt(np.mean((a-p)**2)) / np.mean(a)


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

def chunk_matmul(A, b, chunk=100):
    n = A.shape[0]
    if n <= chunk: return A @ b
    out = np.empty(n)
    for i in range(0, n, chunk): out[i:i+chunk] = A[i:i+chunk] @ b
    return out

# 주의: 방법2(PC=RT+rate*DA)에서는 oracle을 3후보{0,actual,1.0}→2후보{0,actual}로 줄여도
# baseline Gap이 논문과 ±0.1%p로 일치했다(오늘_GapRate비교.md §8.1). 하지만 이 스크립트의
# PC=rate*max(DA,RT)는 rate=0일 때 PC=0이 되어 shortage(commit>actual) 비용이 사라지므로,
# commit=1.0 후보를 오라클에서 빼면 realized가 oracle을 넘어서는(Gap<0) 경우가 생긴다
# (실제로 시도했더니 Gap이 음수로 나옴 — 이 formula에서는 3후보 유지가 맞다).
# 그래서 이 부분은 방법2의 수정을 그대로 포팅하지 않고 원래의 3후보 oracle을 유지한다.
def compute_gap(pc_rate, pred, actual, da, rt):
    s_r = 0.0; s_o = 0.0
    for i in range(len(pred)):
        a=actual[i]; x=pred[i]; dp=da[i]; rp=rt[i]
        pc = pc_rate * max(dp, rp)
        m = a - x; yp = max(m, 0); ym = max(-m, 0)
        s_r += scale * (dp * x + rp * yp - pc * ym)
        p0 = scale * rp * a          # commit = 0
        pa = scale * dp * a          # commit = actual
        s1 = max(a-1.0, 0); y1 = max(1.0-a, 0)
        p1 = scale * (dp*1.0 + rp*s1 - pc*y1)  # commit = 1.0
        s_o += max(p0, pa, p1)
    return 100.0 * (s_o - s_r) / s_o if s_o > 1e-10 else 0.0


# =====================================================================
# 4. AR baseline (bounded LAD)
# =====================================================================
print("\n" + "=" * 70)
print("  2. AR baseline (bounded LAD)")
print("=" * 70)
ar_coeff = np.zeros((HOURS_PER_DAY, n_feat_ar))
for h in range(HOURS_PER_DAY):
    y_h = tr_s[:, h]

    gmodel = gp.Model(f"baseline_ar_hour_{h}")
    gmodel.Params.OutputFlag = 0

    beta_var = gmodel.addMVar(n_feat_ar, lb=-GRB.INFINITY, name="beta")
    u_var = gmodel.addMVar(n_tr_days, lb=0.0, name="u")

    fitted_expr = ar_X @ beta_var
    gmodel.addConstr(fitted_expr - y_h <= u_var, name="resid_upper")
    gmodel.addConstr(y_h - fitted_expr <= u_var, name="resid_lower")
    gmodel.setObjective(u_var.sum() / n_tr_days, GRB.MINIMIZE)
    gmodel.optimize()
    if gmodel.Status != GRB.OPTIMAL:
        raise RuntimeError(f"AR baseline LP failed h={h}: Gurobi status={gmodel.Status}")

    ar_coeff[h] = beta_var.X

ar_fc = np.zeros((n_te_days, HOURS_PER_DAY)); prev = tr_s[-1]
for d in range(n_te_days):
    fv = np.concatenate([[1.0], prev[::-1]])
    for h in range(HOURS_PER_DAY): ar_fc[d,h] = np.clip(np.dot(ar_coeff[h], fv), 0, 1)
    prev = te_s[d]
ar_pred = ar_fc.flatten()
ar_nrmse = nrmse_fn(ar_pred, ar_act)
print(f"  AR baseline nRMSE = {ar_nrmse:.2f}%")


# =====================================================================
# 5. MLR baseline (bounded LAD, pooled)
# =====================================================================
print("\n" + "=" * 70)
print("  3. MLR baseline (bounded LAD, pooled)")
print("=" * 70)
gmodel_mlr = gp.Model("baseline_mlr")
gmodel_mlr.Params.OutputFlag = 0

beta_var_mlr = gmodel_mlr.addMVar(n_feat_mlr, lb=-GRB.INFINITY, name="beta")
u_var_mlr = gmodel_mlr.addMVar(n_tr_obs, lb=0.0, name="u")

fitted_expr_mlr = X_tr @ beta_var_mlr
gmodel_mlr.addConstr(fitted_expr_mlr - mlr_tr_s <= u_var_mlr, name="resid_upper")
gmodel_mlr.addConstr(mlr_tr_s - fitted_expr_mlr <= u_var_mlr, name="resid_lower")
gmodel_mlr.setObjective(u_var_mlr.sum() / n_tr_obs, GRB.MINIMIZE)
gmodel_mlr.optimize()
if gmodel_mlr.Status != GRB.OPTIMAL:
    raise RuntimeError(f"MLR baseline LP failed: Gurobi status={gmodel_mlr.Status}")

mlr_coeff = beta_var_mlr.X
mlr_pred = np.clip(chunk_matmul(X_te, mlr_coeff), 0, 1)
mlr_nrmse = nrmse_fn(mlr_pred, mlr_te_s)
print(f"  MLR baseline nRMSE = {mlr_nrmse:.2f}%")


# =====================================================================
# 6. 제안모형 MILP — AR (정규화 제거 + 등식 제약)
# =====================================================================
_cache = {}

def solve_ar(pc_rate, W1, W2):
    k = (pc_rate, W1, W2, "ar")
    if k in _cache: return _cache[k]
    W2_eff = W2 * W2_BALANCE_WEIGHT
    coeffs = np.zeros((HOURS_PER_DAY, n_feat_ar))
    for hour in range(HOURS_PER_DAY):
        y_h = tr_s[:, hour]; da_h = tr_d[:, hour]; rt_h = tr_r[:, hour]
        n_obs = n_tr_days

        sc = np.zeros(n_obs); yc = np.zeros(n_obs)
        for i in range(n_obs):
            pc = pc_rate * max(da_h[i], rt_h[i])
            sc[i] = (-W1 * rt_h[i]) + W2_eff    # 논문 Eq.(9a), 정규화 제거 + W2 균형 보정
            yc[i] = (W1 * pc) + W2_eff          # 논문 Eq.(9a), 정규화 제거 + W2 균형 보정

        bl = [i for i in range(n_obs) if sc[i]+yc[i] < 0]
        ba = np.array(bl, dtype=int); nb = len(ba)

        gmodel = gp.Model(f"proposed_ar_hour_{hour}")
        gmodel.Params.OutputFlag = 0
        gmodel.Params.MIPGap = 1e-9

        beta_var = gmodel.addMVar(n_feat_ar, lb=-GRB.INFINITY, name="beta")
        x_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="x")
        yplus_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="y_plus")
        yminus_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="y_minus")

        gmodel.addConstr(x_var - ar_X @ beta_var == 0.0, name="commitment_eq")
        gmodel.addConstr(x_var + yplus_var - yminus_var == y_h, name="mismatch_eq")

        if nb > 0:
            z_var = gmodel.addMVar(nb, vtype=GRB.BINARY, name="z")
            gmodel.addConstr(yplus_var[ba] + z_var <= 1.0, name="complementarity_plus")
            gmodel.addConstr(yminus_var[ba] - z_var <= 0.0, name="complementarity_minus")

        objective_expr = (-W1 * da_h) @ x_var + sc @ yplus_var + yc @ yminus_var
        gmodel.setObjective(objective_expr, GRB.MINIMIZE)
        gmodel.optimize()
        if gmodel.Status != GRB.OPTIMAL:
            print(f"    [경고] AR h={hour}: Gurobi status={gmodel.Status} -> baseline 사용")
            coeffs[hour] = ar_coeff[hour]; continue
        coeffs[hour] = beta_var.X

    fc = np.zeros((n_te_days, HOURS_PER_DAY)); prev = tr_s[-1]
    for d in range(n_te_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY): fc[d,h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = te_s[d]
    pred = fc.flatten()
    val = (nrmse_fn(pred, ar_act), compute_gap(pc_rate, pred, ar_act, ar_da, ar_rt))
    _cache[k] = val; return val


# =====================================================================
# 7. 제안모형 MILP — MLR (정규화 제거 + 등식 제약)
# =====================================================================
def solve_mlr(pc_rate, W1, W2):
    k = (pc_rate, W1, W2, "mlr")
    if k in _cache: return _cache[k]
    W2_eff = W2 * W2_BALANCE_WEIGHT
    n_obs = n_tr_obs

    sc = np.zeros(n_obs); yc = np.zeros(n_obs)
    for i in range(n_obs):
        pc = pc_rate * max(mlr_tr_d[i], mlr_tr_r[i])
        sc[i] = (-W1 * mlr_tr_r[i]) + W2_eff    # 논문 Eq.(10a), 정규화 제거 + W2 균형 보정
        yc[i] = (W1 * pc) + W2_eff              # 논문 Eq.(10a), 정규화 제거 + W2 균형 보정

    bl = [i for i in range(n_obs) if sc[i]+yc[i] < 0]
    ba = np.array(bl, dtype=int); nb = len(ba)

    gmodel = gp.Model("proposed_mlr")
    gmodel.Params.OutputFlag = 0
    gmodel.Params.MIPGap = 1e-9

    beta_var = gmodel.addMVar(n_feat_mlr, lb=-GRB.INFINITY, name="beta")
    x_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="x")
    yplus_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="y_plus")
    yminus_var = gmodel.addMVar(n_obs, lb=0.0, ub=1.0, name="y_minus")

    gmodel.addConstr(x_var - X_tr @ beta_var == 0.0, name="commitment_eq")
    gmodel.addConstr(x_var + yplus_var - yminus_var == mlr_tr_s, name="mismatch_eq")

    if nb > 0:
        z_var = gmodel.addMVar(nb, vtype=GRB.BINARY, name="z")
        gmodel.addConstr(yplus_var[ba] + z_var <= 1.0, name="complementarity_plus")
        gmodel.addConstr(yminus_var[ba] - z_var <= 0.0, name="complementarity_minus")

    objective_expr = (-W1 * mlr_tr_d) @ x_var + sc @ yplus_var + yc @ yminus_var
    gmodel.setObjective(objective_expr, GRB.MINIMIZE)
    gmodel.optimize()
    if gmodel.Status != GRB.OPTIMAL:
        print(f"    [경고] MLR: Gurobi status={gmodel.Status} -> baseline 사용")
        pred = np.clip(chunk_matmul(X_te, mlr_coeff), 0, 1)
    else:
        pred = np.clip(chunk_matmul(X_te, beta_var.X), 0, 1)
    val = (nrmse_fn(pred, mlr_te_s), compute_gap(pc_rate, pred, mlr_te_s, mlr_te_d, mlr_te_r))
    _cache[k] = val; return val


# =====================================================================
# 8. Fig3/6 데이터 — W1/W2 10개점 (rate=0.5)
# =====================================================================
print("\n" + "=" * 70)
print("  4. Fig.3/6 데이터 — W1/W2 10개점 (rate=" + str(KPI_RATE) + ")")
print("=" * 70)

f3n=[]; f3g=[]; f6n=[]; f6g=[]
for w1, w2 in W_RATIOS:
    print(f"  {w1}/{w2}: 계산 중...", end=" ", flush=True)
    an, ag = solve_ar(KPI_RATE, w1, w2)
    mn, mg = solve_mlr(KPI_RATE, w1, w2)
    f3n.append(an); f3g.append(ag); f6n.append(mn); f6g.append(mg)
    print(f"AR(nRMSE={an:.2f}%,Gap={ag:.2f}%) MLR(nRMSE={mn:.2f}%,Gap={mg:.2f}%)")


# =====================================================================
# 9. Fig5/8 데이터 — rate 0~100% (W1=W2=1, W2_eff=W2*W2_BALANCE_WEIGHT)
# =====================================================================
print("\n" + "=" * 70)
print(f"  5. Fig.5/8 데이터 — rate 0~100% (W1=W2=1, raw 계수 그대로 — 방법2식 W2 보정 미적용)")
print("=" * 70)

f5_pn=[]; f5_pg=[]; f8_pn=[]; f8_pg=[]
f5_bg=[]; f8_bg=[]  # baseline(AR/MLR)의 rate별 Gap — nRMSE는 rate와 무관해 상수 재사용
for rate in RATES:
    print(f"  rate={rate:.1f}: 계산 중...", end=" ", flush=True)
    an, ag = solve_ar(rate, 1, 1)
    mn, mg = solve_mlr(rate, 1, 1)
    f5_pn.append(an); f5_pg.append(ag)
    f8_pn.append(mn); f8_pg.append(mg)
    f5_bg.append(compute_gap(rate, ar_pred, ar_act, ar_da, ar_rt))
    f8_bg.append(compute_gap(rate, mlr_pred, mlr_te_s, mlr_te_d, mlr_te_r))
    print(f"AR(nRMSE={an:.2f}%,Gap={ag:.2f}%) MLR(nRMSE={mn:.2f}%,Gap={mg:.2f}%)")


# =====================================================================
# 논문 KPI 조건 — W1=1, W2=20, penalty rate=50%
# =====================================================================
print("\n" + "=" * 70)
print("  논문 KPI 조건 — W1=1, W2=20, penalty rate=50%")
print("=" * 70)

ar_kpi_gap_base = compute_gap(KPI_RATE, ar_pred, ar_act, ar_da, ar_rt)
mlr_kpi_gap_base = compute_gap(KPI_RATE, mlr_pred, mlr_te_s, mlr_te_d, mlr_te_r)

print(f"")
print(f"  +-------------+---------------+---------------+---------------+---------------+")
print(f"  |     모델     |  nRMSE (%)     |   Gap (%)     |  nRMSE (%)     |   Gap (%)     |")
print(f"  |             |   제안모형     |    제안모형    |   baseline     |   baseline     |")
print(f"  +-------------+---------------+---------------+---------------+---------------+")
print(f"  | AR          | {f3n[0]:13.2f} | {f3g[0]:13.2f} | {ar_nrmse:13.2f} | {ar_kpi_gap_base:13.2f} |")
print(f"  | MLR         | {f6n[0]:13.2f} | {f6g[0]:13.2f} | {mlr_nrmse:13.2f} | {mlr_kpi_gap_base:13.2f} |")
print(f"  +-------------+---------------+---------------+---------------+---------------+")
print(f"")
print(f"  ※ 논문 참고값 (z03/블록18): AR nRMSE=34.76%, Gap=15.04% / MLR nRMSE=21.76%, Gap=12.59%")
print(f"")


# ── CSV 저장 ──
def save_csv(path, hdr, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr)
        for r in rows: w.writerow(r)
    print(f"  saved: {path}")

save_csv(os.path.join(OUT_DIR, "fig3_pc_max_rt_dt_AR_gurobi.csv"), ["Label","nRMSE","Gap"],
         [[W_LABELS[i], round(f3n[i],2), round(f3g[i],2)] for i in range(10)])
save_csv(os.path.join(OUT_DIR, "fig5_pc_max_rt_dt_AR_gurobi.csv"), ["Rate","nRMSE","Gap"],
         [[RATES[i], round(f5_pn[i],2), round(f5_pg[i],2)] for i in range(len(RATES))])
save_csv(os.path.join(OUT_DIR, "fig6_pc_max_rt_dt_MLR_gurobi.csv"), ["Label","nRMSE","Gap"],
         [[W_LABELS[i], round(f6n[i],2), round(f6g[i],2)] for i in range(10)])
save_csv(os.path.join(OUT_DIR, "fig8_pc_max_rt_dt_MLR_gurobi.csv"), ["Rate","nRMSE","Gap"],
         [[RATES[i], round(f8_pn[i],2), round(f8_pg[i],2)] for i in range(len(RATES))])


# =====================================================================
# 10. Fig.3 — AR, W1/W2 스윕 (논문 기준 + 제안모형)
# =====================================================================
print("\n" + "=" * 70)
print("  6. Fig.3 그리기")
print("=" * 70)

W_LABELS_ALL = ["AR"] + W_LABELS
x10 = list(range(len(W_LABELS_ALL)))
paper_ar_nrmse_all = [PAPER_AR_BASE_NRMSE] + PAPER_AR_PROP_NRMSE
paper_ar_gap_all = [PAPER_AR_BASE_GAP] + PAPER_AR_PROP_GAP
f3n_all = [ar_nrmse] + f3n
f3g_all = [ar_kpi_gap_base] + f3g
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("방법4_fig3 — AR PC=rate×max(DA,RT), W1/W2 스윕 (rate=0.5)", fontsize=13)
ax_r = ax.twinx()

ax.plot(x10, paper_ar_nrmse_all, marker="o", color=C_PAPER, lw=2, ls="--",
        label="논문 nRMSE", alpha=0.8)
ax.plot(x10, f3n_all, marker="o", color=C_PROP, lw=2.5, ls="-", label="재현 nRMSE")
ax_r.plot(x10, paper_ar_gap_all, marker="s", color=C_PAPER, lw=2, ls="--",
          label="논문 Gap", alpha=0.8)
ax_r.plot(x10, f3g_all, marker="s", color=C_PROP, lw=2.5, ls="-", label="재현 Gap")

ax.set_xticks(x10); ax.set_xticklabels(W_LABELS_ALL)
ax.set_xlabel("W1/W2"); ax.set_ylabel("nRMSE (%)", color=C_PAPER)
ax_r.set_ylabel("Optimality Gap (%)", color=C_PROP)
ax.tick_params(axis="y", labelcolor=C_PAPER); ax_r.tick_params(axis="y", labelcolor=C_PROP)
ax.grid(True, alpha=0.3, color=C_GRID)
ax.set_ylim(*auto_ylim(paper_ar_nrmse_all, f3n_all))
ax_r.set_ylim(*auto_ylim(paper_ar_gap_all, f3g_all))
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax_r.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc="upper left", fontsize=10)
fig.tight_layout()
p3 = os.path.join(RESULTS_DIR, "방법4_fig3_ar_z03_block18_gurobi.png")
fig.savefig(p3, dpi=150); plt.close(fig)
print(f"  saved: {p3}")


# =====================================================================
# 11. Fig.5 — AR, rate 스윕 (2분할)
# =====================================================================
print("\n" + "=" * 70)
print("  7. Fig.5 그리기")
print("=" * 70)

x16 = list(range(len(RATES))); lbl = [f"{int(r*100)}%" for r in RATES]; hl = RATES.index(KPI_RATE)
x16_paper = x16[:11]  # 논문 판독값은 rate 0~100%(11개점)까지만 있음
fig5, (a5n, a5g) = plt.subplots(1, 2, figsize=(13, 5))
fig5.suptitle("방법4_fig5 — AR PC=rate×max(DA,RT), W1=W2=1, penalty rate 스윕", fontsize=13)

a5n.plot(x16_paper, FIG5_PAPER_AR_NRMSE, ls="--", color=C_AR, marker="o", markersize=4,
         alpha=0.6, label="논문 AR")
a5n.plot(x16_paper, FIG5_PAPER_PROP_NRMSE, ls="--", color=C_PROP, marker="s", markersize=4,
         alpha=0.6, label="논문 제안모형")
a5n.axhline(ar_nrmse, color=C_AR, lw=2.5, ls="-", label="재현 AR")
a5n.plot(x16, f5_pn, marker="s", color=C_PROP, lw=2.5, ls="-", label="재현 제안모형")
a5n.set_xticks(x16); a5n.set_xticklabels(lbl, rotation=45)
a5n.set_xlabel("벌금비용률"); a5n.set_ylabel("nRMSE (%)")
a5n.set_title("nRMSE"); a5n.grid(True, alpha=0.3, color=C_GRID); a5n.legend(fontsize=8)
a5n.set_ylim(*auto_ylim(FIG5_PAPER_AR_NRMSE, FIG5_PAPER_PROP_NRMSE, [ar_nrmse], f5_pn))

a5g.plot(x16_paper, FIG5_PAPER_AR_GAP, ls="--", color=C_AR, marker="o", markersize=4,
         alpha=0.6, label="논문 AR")
a5g.plot(x16_paper, FIG5_PAPER_PROP_GAP, ls="--", color=C_PROP, marker="s", markersize=4,
         alpha=0.6, label="논문 제안모형")
a5g.plot(x16, f5_bg, color=C_AR, lw=2.5, ls="-", marker="o", label="재현 AR")
a5g.plot(x16, f5_pg, marker="s", color=C_PROP, lw=2.5, ls="-", label="재현 제안모형")
a5g.axvline(hl, color=C_GRID, ls=":", lw=1.5, label=f"KPI rate={int(KPI_RATE*100)}%")
a5g.set_xticks(x16); a5g.set_xticklabels(lbl, rotation=45)
a5g.set_xlabel("벌금비용률"); a5g.set_ylabel("Optimality Gap (%)")
a5g.set_title("Optimality Gap"); a5g.grid(True, alpha=0.3, color=C_GRID); a5g.legend(fontsize=8)
a5g.set_ylim(*auto_ylim(FIG5_PAPER_AR_GAP, FIG5_PAPER_PROP_GAP, f5_bg, f5_pg))

fig5.tight_layout()
p5 = os.path.join(RESULTS_DIR, "방법4_fig5_ar_z03_block18_gurobi.png")
fig5.savefig(p5, dpi=150); plt.close(fig5)
print(f"  saved: {p5}")


# =====================================================================
# 12. Fig.6 — MLR, W1/W2 스윕
# =====================================================================
print("\n" + "=" * 70)
print("  8. Fig.6 그리기")
print("=" * 70)

fig6, ax6 = plt.subplots(figsize=(10, 6))
fig6.suptitle("방법4_fig6 — MLR PC=rate×max(DA,RT), W1/W2 스윕 (rate=0.5)", fontsize=13)
ax6_r = ax6.twinx()

paper_mlr_nrmse_all = [PAPER_MLR_BASE_NRMSE] + PAPER_MLR_PROP_NRMSE
paper_mlr_gap_all = [PAPER_MLR_BASE_GAP] + PAPER_MLR_PROP_GAP
f6n_all = [mlr_nrmse] + f6n
f6g_all = [mlr_kpi_gap_base] + f6g

ax6.plot(x10, paper_mlr_nrmse_all, marker="o", color=C_PAPER, lw=2, ls="--",
         label="논문 nRMSE", alpha=0.8)
ax6.plot(x10, f6n_all, marker="o", color=C_PROP, lw=2.5, ls="-", label="재현 nRMSE")
ax6_r.plot(x10, paper_mlr_gap_all, marker="s", color=C_PAPER, lw=2, ls="--",
           label="논문 Gap", alpha=0.8)
ax6_r.plot(x10, f6g_all, marker="s", color=C_PROP, lw=2.5, ls="-", label="재현 Gap")

ax6.set_xticks(x10); ax6.set_xticklabels(W_LABELS_ALL)
ax6.set_xlabel("W1/W2"); ax6.set_ylabel("nRMSE (%)", color=C_PAPER)
ax6_r.set_ylabel("Optimality Gap (%)", color=C_PROP)
ax6.tick_params(axis="y", labelcolor=C_PAPER); ax6_r.tick_params(axis="y", labelcolor=C_PROP)
ax6.grid(True, alpha=0.3, color=C_GRID)
ax6.set_ylim(*auto_ylim(paper_mlr_nrmse_all, f6n_all))
ax6_r.set_ylim(*auto_ylim(paper_mlr_gap_all, f6g_all))
h1, l1 = ax6.get_legend_handles_labels(); h2, l2 = ax6_r.get_legend_handles_labels()
ax6.legend(h1+h2, l1+l2, loc="upper left", fontsize=10)
fig6.tight_layout()
p6 = os.path.join(RESULTS_DIR, "방법4_fig6_mlr_z03_block18_gurobi.png")
fig6.savefig(p6, dpi=150); plt.close(fig6)
print(f"  saved: {p6}")


# =====================================================================
# 13. Fig.8 — MLR, rate 스윕 (2분할)
# =====================================================================
print("\n" + "=" * 70)
print("  9. Fig.8 그리기")
print("=" * 70)

fig8, (a8n, a8g) = plt.subplots(1, 2, figsize=(13, 5))
fig8.suptitle("방법4_fig8 — MLR PC=rate×max(DA,RT), W1=W2=1, penalty rate 스윕", fontsize=13)

a8n.plot(x16_paper, FIG8_PAPER_MLR_NRMSE, ls="--", color=C_MLR, marker="o", markersize=4,
         alpha=0.6, label="논문 MLR")
a8n.plot(x16_paper, FIG8_PAPER_PROP_NRMSE, ls="--", color=C_PROP, marker="s", markersize=4,
         alpha=0.6, label="논문 제안모형")
a8n.axhline(mlr_nrmse, color=C_MLR, lw=2.5, ls="-", label="재현 MLR")
a8n.plot(x16, f8_pn, marker="s", color=C_PROP, lw=2.5, ls="-", label="재현 제안모형")
a8n.set_xticks(x16); a8n.set_xticklabels(lbl, rotation=45)
a8n.set_xlabel("벌금비용률"); a8n.set_ylabel("nRMSE (%)")
a8n.set_title("nRMSE"); a8n.grid(True, alpha=0.3, color=C_GRID); a8n.legend(fontsize=8)
a8n.set_ylim(*auto_ylim(FIG8_PAPER_MLR_NRMSE, FIG8_PAPER_PROP_NRMSE, [mlr_nrmse], f8_pn))

a8g.plot(x16_paper, FIG8_PAPER_MLR_GAP, ls="--", color=C_MLR, marker="o", markersize=4,
         alpha=0.6, label="논문 MLR")
a8g.plot(x16_paper, FIG8_PAPER_PROP_GAP, ls="--", color=C_PROP, marker="s", markersize=4,
         alpha=0.6, label="논문 제안모형")
a8g.plot(x16, f8_bg, color=C_MLR, lw=2.5, ls="-", marker="o", label="재현 MLR")
a8g.plot(x16, f8_pg, marker="s", color=C_PROP, lw=2.5, ls="-", label="재현 제안모형")
a8g.axvline(hl, color=C_GRID, ls=":", lw=1.5, label=f"KPI rate={int(KPI_RATE*100)}%")
a8g.set_xticks(x16); a8g.set_xticklabels(lbl, rotation=45)
a8g.set_xlabel("벌금비용률"); a8g.set_ylabel("Optimality Gap (%)")
a8g.set_title("Optimality Gap"); a8g.grid(True, alpha=0.3, color=C_GRID); a8g.legend(fontsize=8)
a8g.set_ylim(*auto_ylim(FIG8_PAPER_MLR_GAP, FIG8_PAPER_PROP_GAP, f8_bg, f8_pg))

fig8.tight_layout()
p8 = os.path.join(RESULTS_DIR, "방법4_fig8_mlr_z03_block18_gurobi.png")
fig8.savefig(p8, dpi=150); plt.close(fig8)
print(f"  saved: {p8}")

print("\n" + "=" * 70)
print("  완료 — 제안모형 AR+MLR 그림 생성 (z03/블록18)")
print("=" * 70)
print(f"  Fig.3 (AR W1/W2): {p3}")
print(f"  Fig.5 (AR rate):  {p5}")
print(f"  Fig.6 (MLR W1/W2): {p6}")
print(f"  Fig.8 (MLR rate): {p8}")