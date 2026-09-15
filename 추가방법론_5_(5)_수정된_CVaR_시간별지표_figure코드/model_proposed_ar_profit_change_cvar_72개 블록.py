# -*- coding: utf-8 -*-

# Proposed AR + 수정 4항 profit + Additive CVaR
# Zone 1·2·3 × 24개 블록에서 lambda=0과 0.05를 같은 코드로 비교한다.
# 이 파일은 노트북의 새 셀에 통째로 붙여 넣어 실행해도 된다.

from pathlib import Path
import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


SCRIPT_DIR = (
    Path(r"D:\Users\Downloads\Github\추가방법론_5_(5)_수정된_CVaR_시간별지표_figure코드\model_proposed_ar_profit_change_cvar_72개 블록.py").resolve().parent

)
SEARCH_DIRS = [SCRIPT_DIR, SCRIPT_DIR.parent]


def find_zone_csv(zone_name):
    for data_dir in SEARCH_DIRS:
        exact_path = data_dir / f"merged_for_simulation_{zone_name}.csv"
        if exact_path.exists():
            return exact_path

    # 다운로드 과정에서 (1), (4) 등이 붙은 파일명도 자동으로 찾는다.
    candidates = []
    for data_dir in SEARCH_DIRS:
        candidates.extend(
            data_dir.glob(f"merged_for_simulation_{zone_name}*.csv")
        )
    candidates = sorted(set(path.resolve() for path in candidates))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"{zone_name} CSV를 찾지 못했습니다.\n"
            f"찾은 폴더: {[str(path) for path in SEARCH_DIRS]}\n"
            f"필요한 파일명: merged_for_simulation_{zone_name}.csv"
        )
    raise RuntimeError(
        f"{zone_name} CSV가 여러 개라 하나를 고를 수 없습니다:\n"
        + "\n".join(str(path) for path in candidates)
    )

# 처음에는 "check"로 z03/block18만 검산하고,
# 수치가 맞으면 "full"로 바꿔 72개 블록 전체를 실행한다.
RUN_SCOPE = "full"                   # 72개 블록 전체 실행

OUTPUT_DIR = SCRIPT_DIR / "results_cvar_72blocks"
RESUME = True                        # 중단 후 다시 실행할 때 완료된 조합은 건너뜀

W1 = 1.0
W2 = 20.0
PENALTY_RATE = 0.5
CVAR_ALPHA = 0.9
LAMBDAS = [0.0, 0.05]

HOURS_PER_DAY = 12
LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21
CAPACITY_MW = 30.0
DURATION_HOURS = 1.0

# block 1: train 2012-04-02~2012-06-30, test 2012-07-01~2012-07-30
# 다음 블록은 30일씩 이동한다. block 18은 z03 대표 블록과 정확히 일치한다.
FIRST_TRAIN_START = pd.Timestamp("2012-04-02")
N_BLOCKS = 24
TRAIN_DAYS = 90
TEST_DAYS = 30
BLOCK_SHIFT_DAYS = 30

# 업로드된 z03/block18 Proposed AR·수정4항·lambda=0 코드의 기준값
REFERENCE_NRMSE = 37.848992
REFERENCE_GAP = 14.387719
REFERENCE_TOL = 1e-3


# =====================================================================
# 1. 보조 함수
# =====================================================================

def cvar_upper_tail(values, alpha=0.9):
    """가장 큰 (1-alpha) 비율 표본의 평균. 360개면 상위 36개 평균."""
    values = np.asarray(values, dtype=float)
    tail_count = max(1, int(math.ceil((1.0 - alpha) * len(values) - 1e-12)))
    return float(np.sort(values)[-tail_count:].mean())


def load_zone_table(csv_path):
    raw = pd.read_csv(csv_path)
    raw["local_date"] = pd.to_datetime(raw["local_date"])
    daylight = raw[
        (raw["local_hour"] >= LOCAL_HOUR_START)
        & (raw["local_hour"] < LOCAL_HOUR_END)
    ].copy()
    daylight["hour_idx"] = daylight["local_hour"] - LOCAL_HOUR_START
    return daylight.sort_values(["local_date", "hour_idx"])


def make_block_arrays(daylight, block_number):
    train_start = FIRST_TRAIN_START + pd.Timedelta(
        days=(block_number - 1) * BLOCK_SHIFT_DAYS
    )
    train_end = train_start + pd.Timedelta(days=TRAIN_DAYS - 1)
    test_start = train_end + pd.Timedelta(days=1)
    test_end = test_start + pd.Timedelta(days=TEST_DAYS - 1)
    history_date = train_start - pd.Timedelta(days=1)

    history_rows = daylight[daylight["local_date"] == history_date].copy()
    history_rows = history_rows.sort_values("hour_idx")

    train_rows = daylight[
        (daylight["local_date"] >= train_start)
        & (daylight["local_date"] <= train_end)
    ].copy().sort_values(["local_date", "hour_idx"])

    test_rows = daylight[
        (daylight["local_date"] >= test_start)
        & (daylight["local_date"] <= test_end)
    ].copy().sort_values(["local_date", "hour_idx"])

    if len(train_rows) != TRAIN_DAYS * HOURS_PER_DAY:
        raise ValueError(
            f"block {block_number}: train 행 수가 {len(train_rows)}개입니다. "
            f"정상값은 {TRAIN_DAYS * HOURS_PER_DAY}개입니다."
        )
    if len(test_rows) != TEST_DAYS * HOURS_PER_DAY:
        raise ValueError(
            f"block {block_number}: test 행 수가 {len(test_rows)}개입니다. "
            f"정상값은 {TEST_DAYS * HOURS_PER_DAY}개입니다."
        )

    # 기존 72블록 구성과 동일하게 block 1의 불완전한 직전 하루는
    # 존재하는 6개 값을 앞에서부터 채우고 나머지를 0으로 둔다.
    # block 2~24는 직전 하루가 12개 모두 존재한다.
    history_solar = np.zeros((1, HOURS_PER_DAY))
    history_values = history_rows["solar_power"].to_numpy(dtype=float)
    if len(history_values) > HOURS_PER_DAY:
        raise ValueError(f"block {block_number}: history 행이 12개보다 많습니다.")
    history_solar[0, :len(history_values)] = history_values

    train_solar = train_rows["solar_power"].to_numpy(dtype=float).reshape(
        TRAIN_DAYS, HOURS_PER_DAY
    )
    train_da = train_rows["da_price"].to_numpy(dtype=float).reshape(
        TRAIN_DAYS, HOURS_PER_DAY
    )
    train_rt = train_rows["rt_price"].to_numpy(dtype=float).reshape(
        TRAIN_DAYS, HOURS_PER_DAY
    )

    test_solar = test_rows["solar_power"].to_numpy(dtype=float).reshape(
        TEST_DAYS, HOURS_PER_DAY
    )
    test_da = test_rows["da_price"].to_numpy(dtype=float).reshape(
        TEST_DAYS, HOURS_PER_DAY
    )
    test_rt = test_rows["rt_price"].to_numpy(dtype=float).reshape(
        TEST_DAYS, HOURS_PER_DAY
    )

    return {
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "history_rows": len(history_rows),
        "history_solar": history_solar,
        "train_solar": train_solar,
        "train_da": train_da,
        "train_rt": train_rt,
        "test_solar": test_solar,
        "test_da": test_da,
        "test_rt": test_rt,
    }


def fit_and_evaluate_block(block, cvar_lambda):
    history_solar = block["history_solar"]
    train_solar = block["train_solar"]
    train_da = block["train_da"]
    train_rt = block["train_rt"]
    test_solar = block["test_solar"]
    test_da = block["test_da"]
    test_rt = block["test_rt"]

    # 기존 AR 코드와 같은 입력: 절편 + 직전 하루 12시간을 역순으로 배치
    history_and_train = np.vstack([history_solar, train_solar])
    lag_features = np.zeros((TRAIN_DAYS, HOURS_PER_DAY))
    for day_index in range(TRAIN_DAYS):
        lag_features[day_index] = history_and_train[day_index][::-1]
    design = np.hstack([np.ones((TRAIN_DAYS, 1)), lag_features])

    n_train = TRAIN_DAYS
    n_features = design.shape[1]
    coefficients = np.zeros((HOURS_PER_DAY, n_features))

    for hour in range(HOURS_PER_DAY):
        actual = train_solar[:, hour]
        da = train_da[:, hour]
        rt = train_rt[:, hour]
        penalty = PENALTY_RATE * da

        # 수정 4항 profit의 3후보 oracle. 학습에서는 30 MW를 곱하지 않는다.
        profit_0 = rt * actual
        profit_actual = da * actual
        profit_1 = da - (penalty + rt) * np.maximum(1.0 - actual, 0.0)
        oracle = np.maximum.reduce([profit_0, profit_actual, profit_1])

        model = gp.Model(f"cvar_ar_h{hour}")
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 1e-9

        beta = model.addMVar(n_features, lb=-GRB.INFINITY, name="beta")
        x = model.addMVar(n_train, lb=0.0, ub=1.0, name="x")
        y_plus = model.addMVar(n_train, lb=0.0, ub=1.0, name="y_plus")
        y_minus = model.addMVar(n_train, lb=0.0, ub=1.0, name="y_minus")

        model.addConstr(x - design @ beta == 0.0, name="commitment")
        model.addConstr(x + y_plus - y_minus == actual, name="mismatch")

        # 기존 Proposed AR·수정4항·비정규화 목적함수와 정확히 같은 부분
        base_objective = (
            (-W1 * da) @ x
            + ((-W1 * rt) + W2) @ y_plus
            + (W1 * (penalty + rt) + W2) @ y_minus
        )

        if cvar_lambda > 0.0:
            zeta = model.addVar(lb=0.0, name="zeta")
            excess = model.addMVar(n_train, lb=0.0, name="cvar_excess")

            # regret = oracle profit - realized profit
            regret_expr = (
                oracle
                - da * x
                - rt * y_plus
                + (penalty + rt) * y_minus
            )
            model.addConstr(excess >= regret_expr - zeta, name="cvar_linearization")

            additive_cvar = W1 * cvar_lambda * (
                n_train * zeta
                + excess.sum() / (1.0 - CVAR_ALPHA)
            )
            model.setObjective(base_objective + additive_cvar, GRB.MINIMIZE)
        else:
            # lambda=0은 업로드된 기존 Proposed AR·수정4항 코드와 같은 목적함수
            model.setObjective(base_objective, GRB.MINIMIZE)

        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"hour={hour}, lambda={cvar_lambda}: "
                f"최적해를 얻지 못했습니다. Gurobi status={model.Status}"
            )
        coefficients[hour] = beta.X

    # 기존 코드와 같은 rolling one-day-ahead 테스트 예측
    forecast = np.zeros((TEST_DAYS, HOURS_PER_DAY))
    previous_actual = train_solar[-1].copy()
    for day_index in range(TEST_DAYS):
        feature = np.concatenate([[1.0], previous_actual[::-1]])
        for hour in range(HOURS_PER_DAY):
            forecast[day_index, hour] = np.clip(
                coefficients[hour] @ feature, 0.0, 1.0
            )
        previous_actual = test_solar[day_index].copy()

    actual = test_solar.ravel()
    commitment = forecast.ravel()
    da = test_da.ravel()
    rt = test_rt.ravel()
    penalty = PENALTY_RATE * da

    surplus = np.maximum(actual - commitment, 0.0)
    shortage = np.maximum(commitment - actual, 0.0)

    realized = CAPACITY_MW * DURATION_HOURS * (
        da * commitment
        + rt * surplus
        - (penalty + rt) * shortage
    )

    profit_0 = CAPACITY_MW * DURATION_HOURS * rt * actual
    profit_actual = CAPACITY_MW * DURATION_HOURS * da * actual
    profit_1 = CAPACITY_MW * DURATION_HOURS * (
        da - (penalty + rt) * np.maximum(1.0 - actual, 0.0)
    )
    oracle = np.maximum.reduce([profit_0, profit_actual, profit_1])
    regret = oracle - realized
    regret[np.abs(regret) < 1e-8] = 0.0

    if regret.min() < -1e-6:
        raise RuntimeError(
            f"최소 regret={regret.min():.9f}: 3후보 oracle 계산을 확인하세요."
        )

    squared_error = (actual - commitment) ** 2
    nrmse = 100.0 * np.sqrt(squared_error.mean()) / actual.mean()
    total_regret = regret.sum()
    gap = 100.0 * total_regret / oracle.sum()

    summary = {
        "n_obs": len(actual),
        "nrmse": nrmse,
        "optimality_gap": gap,
        "total_regret": total_regret,
        "cvar90": cvar_upper_tail(regret, CVAR_ALPHA),
        "max_regret": regret.max(),
        "sse": squared_error.sum(),
        "actual_sum": actual.sum(),
        "oracle_sum": oracle.sum(),
        "realized_sum": realized.sum(),
    }

    detail = pd.DataFrame({
        "actual": actual,
        "forecast": commitment,
        "da_price": da,
        "rt_price": rt,
        "oracle_profit": oracle,
        "realized_profit": realized,
        "regret": regret,
        "squared_error": squared_error,
    })
    detail["test_date"] = np.repeat(
        pd.date_range(block["test_start"], block["test_end"], freq="D"),
        HOURS_PER_DAY,
    )
    detail["local_hour"] = np.tile(
        np.arange(LOCAL_HOUR_START, LOCAL_HOUR_END), TEST_DAYS
    )

    return summary, detail


def pooled_metrics(detail):
    n_obs = len(detail)
    pooled_nrmse = (
        100.0
        * np.sqrt(detail["squared_error"].sum() / n_obs)
        / (detail["actual"].sum() / n_obs)
    )
    total_regret = detail["regret"].sum()
    return {
        "n_obs": n_obs,
        "total_regret": total_regret,
        "pooled_optimality_gap": (
            100.0 * total_regret / detail["oracle_profit"].sum()
        ),
        "pooled_cvar90": cvar_upper_tail(detail["regret"], CVAR_ALPHA),
        "maximum_regret": detail["regret"].max(),
        "pooled_nrmse": pooled_nrmse,
    }


# =====================================================================
# 2. 실행
# =====================================================================

if RUN_SCOPE not in {"check", "full"}:
    raise ValueError('RUN_SCOPE는 "check" 또는 "full"이어야 합니다.')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if RUN_SCOPE == "check":
    selected_zones = ["z03"]
    selected_blocks = [18]
    summary_path = OUTPUT_DIR / "check_z03_block18_summary.csv"
    detail_path = OUTPUT_DIR / "check_z03_block18_detail.csv"
else:
    selected_zones = ["z01", "z02", "z03"]
    selected_blocks = list(range(1, N_BLOCKS + 1))
    summary_path = OUTPUT_DIR / "cvar_72blocks_summary.csv"
    detail_path = OUTPUT_DIR / "cvar_72blocks_detail.csv"

ZONE_FILES = {
    zone_name: find_zone_csv(zone_name)
    for zone_name in selected_zones
}
print("사용할 CSV")
for zone_name, csv_path in ZONE_FILES.items():
    print(f"  {zone_name}: {csv_path}")

if RESUME and summary_path.exists() and detail_path.exists():
    summary_all = pd.read_csv(summary_path)
    detail_all = pd.read_csv(detail_path, parse_dates=["test_date"])
else:
    summary_all = pd.DataFrame()
    detail_all = pd.DataFrame()

completed = set()
if not summary_all.empty:
    completed = set(
        zip(
            summary_all["zone"],
            summary_all["block"].astype(int),
            summary_all["cvar_lambda"].astype(float),
        )
    )

zone_tables = {
    zone: load_zone_table(ZONE_FILES[zone])
    for zone in selected_zones
}

job_total = len(selected_zones) * len(selected_blocks) * len(LAMBDAS)
job_number = 0
start_time = time.time()

for zone in selected_zones:
    daylight = zone_tables[zone]
    for block_number in selected_blocks:
        block = make_block_arrays(daylight, block_number)

        for cvar_lambda in LAMBDAS:
            job_number += 1
            key = (zone, block_number, float(cvar_lambda))
            if key in completed:
                print(f"[{job_number}/{job_total}] 건너뜀: {key}")
                continue

            print(
                f"[{job_number}/{job_total}] 실행: {zone} block {block_number}, "
                f"lambda={cvar_lambda}"
            )
            summary, detail = fit_and_evaluate_block(block, cvar_lambda)

            summary.update({
                "zone": zone,
                "block": block_number,
                "cvar_lambda": cvar_lambda,
                "train_start": block["train_start"].date(),
                "train_end": block["train_end"].date(),
                "test_start": block["test_start"].date(),
                "test_end": block["test_end"].date(),
                "history_rows": block["history_rows"],
            })
            detail.insert(0, "cvar_lambda", cvar_lambda)
            detail.insert(0, "block", block_number)
            detail.insert(0, "zone", zone)

            summary_all = pd.concat(
                [summary_all, pd.DataFrame([summary])], ignore_index=True
            )
            detail_all = pd.concat([detail_all, detail], ignore_index=True)

            # 매 조합이 끝날 때 저장하므로 중간에 끊겨도 RESUME=True로 재시작 가능
            summary_all.to_csv(summary_path, index=False, encoding="utf-8-sig")
            detail_all.to_csv(detail_path, index=False, encoding="utf-8-sig")

            print(
                f"    nRMSE={summary['nrmse']:.6f}% | "
                f"gap={summary['optimality_gap']:.6f}% | "
                f"total regret={summary['total_regret']:.2f} | "
                f"CVaR90={summary['cvar90']:.2f}"
            )


# =====================================================================
# 3. 검산 및 최종 집계
# =====================================================================

reference_row = summary_all[
    (summary_all["zone"] == "z03")
    & (summary_all["block"] == 18)
    & np.isclose(summary_all["cvar_lambda"], 0.0)
]
if len(reference_row) != 1:
    raise RuntimeError("z03 block18 lambda=0 기준행을 하나만 찾지 못했습니다.")

reference_row = reference_row.iloc[0]
nrmse_diff = abs(reference_row["nrmse"] - REFERENCE_NRMSE)
gap_diff = abs(reference_row["optimality_gap"] - REFERENCE_GAP)

print("\n=== z03 block18 lambda=0 기존 코드 검산 ===")
print(f"nRMSE: {reference_row['nrmse']:.6f}% (기준 {REFERENCE_NRMSE:.6f}%)")
print(f"gap:   {reference_row['optimality_gap']:.6f}% (기준 {REFERENCE_GAP:.6f}%)")

if nrmse_diff > REFERENCE_TOL or gap_diff > REFERENCE_TOL:
    raise RuntimeError(
        "기존 코드와 기준값이 일치하지 않습니다. 전체 72블록 결과를 사용하지 마세요."
    )
print("검산 통과")


if RUN_SCOPE == "full":
    expected_rows = len(selected_zones) * len(selected_blocks) * len(LAMBDAS)
    if len(summary_all) != expected_rows:
        raise RuntimeError(
            f"summary 행 수가 {len(summary_all)}개입니다. 정상값은 {expected_rows}개입니다."
        )

    pooled_rows = []
    for zone_group in ["전체", "z01", "z02", "z03"]:
        if zone_group == "전체":
            zone_detail = detail_all
        else:
            zone_detail = detail_all[detail_all["zone"] == zone_group]

        for cvar_lambda in LAMBDAS:
            one = zone_detail[np.isclose(zone_detail["cvar_lambda"], cvar_lambda)]
            pooled = pooled_metrics(one)
            pooled.update({"zone": zone_group, "cvar_lambda": cvar_lambda})
            pooled_rows.append(pooled)

    pooled_table = pd.DataFrame(pooled_rows)
    pooled_path = OUTPUT_DIR / "cvar_72blocks_pooled.csv"
    pooled_table.to_csv(pooled_path, index=False, encoding="utf-8-sig")

    left = summary_all[np.isclose(summary_all["cvar_lambda"], 0.0)].copy()
    right = summary_all[np.isclose(summary_all["cvar_lambda"], 0.05)].copy()
    compare_columns = [
        "nrmse", "optimality_gap", "total_regret", "cvar90", "max_regret"
    ]
    comparison = left[["zone", "block"] + compare_columns].merge(
        right[["zone", "block"] + compare_columns],
        on=["zone", "block"],
        suffixes=("_lambda0", "_lambda005"),
    )
    for metric in compare_columns:
        comparison[f"{metric}_change"] = (
            comparison[f"{metric}_lambda005"]
            - comparison[f"{metric}_lambda0"]
        )
        if metric not in {"nrmse", "optimality_gap"}:
            comparison[f"{metric}_change_percent"] = (
                100.0
                * comparison[f"{metric}_change"]
                / comparison[f"{metric}_lambda0"]
            )

    comparison_path = OUTPUT_DIR / "cvar_72blocks_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    counts = {
        "total_regret_improved": int((comparison["total_regret_change"] < 0).sum()),
        "cvar90_improved": int((comparison["cvar90_change"] < 0).sum()),
        "nrmse_improved": int((comparison["nrmse_change"] < 0).sum()),
        "gap_improved": int((comparison["optimality_gap_change"] < 0).sum()),
        "nrmse_and_gap_improved": int(
            (
                (comparison["nrmse_change"] < 0)
                & (comparison["optimality_gap_change"] < 0)
            ).sum()
        ),
    }
    count_path = OUTPUT_DIR / "cvar_72blocks_improvement_counts.csv"
    pd.DataFrame([counts]).to_csv(count_path, index=False, encoding="utf-8-sig")

    print("\n=== 72개 블록 pooled 결과 ===")
    print(pooled_table.to_string(index=False))
    print("\n=== 개선 블록 수 ===")
    for name, value in counts.items():
        print(f"{name}: {value}/72")

    print("\n저장 파일")
    for path in [summary_path, detail_path, pooled_path, comparison_path, count_path]:
        print(path)

elapsed = time.time() - start_time
print(f"\n총 실행시간: {elapsed / 60.0:.2f}분")
