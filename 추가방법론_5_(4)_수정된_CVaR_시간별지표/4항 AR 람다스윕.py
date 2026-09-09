import os
import runpy
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# 1. 경로 설정: 여기 두 줄만 네 경로로 수정
# =====================================================================

MODEL_FILE = r"D:\Users\Downloads\mlr 구로비 라이선스 돌려보려고 - CVaR 도입 - step2_각시간별값만보는코드까지포함된거 - 복사본\model_proposed_ar_profit_change.py"

MERGED_FILE = (
    r"D:\Users\Downloads\mlr 구로비 라이선스 돌려보려고 - CVaR 도입 - step2_각시간별값만보는코드까지포함된거 - 복사본\merged_for_simulation_z03.csv"
)


# =====================================================================
# 2. Lambda sweep 설정
#    W1/W2 = 1/20, alpha = 0.90 고정
# =====================================================================

LAMBDA_LIST = [
    0.000,
    0.025,
    0.050,
    0.075,
    0.100,
]

FIXED_ALPHA = 0.90
FIXED_W1 = 1.0
FIXED_W2 = 20.0

# 4항 AR 본체가 저장하는 고정 파일명
RESULT_NPZ = os.path.abspath(
    "proposed_ar_4term_combined_diagnostics.npz"
)


# =====================================================================
# 3. 기존 환경변수 보관
# =====================================================================

environment_keys = [
    "CVAR_ALPHA",
    "CVAR_LAMBDA",
    "SWEEP_W1",
    "SWEEP_W2",
    "MERGED_FILE",
]

previous_environment = {}

for key in environment_keys:
    previous_environment[key] = os.environ.get(key)


# =====================================================================
# 4. Lambda별 반복 실행
# =====================================================================

sweep_rows = []

try:

    os.environ["CVAR_ALPHA"] = str(FIXED_ALPHA)
    os.environ["SWEEP_W1"] = str(FIXED_W1)
    os.environ["SWEEP_W2"] = str(FIXED_W2)
    os.environ["MERGED_FILE"] = MERGED_FILE

    for lambda_value in LAMBDA_LIST:

        print()
        print("=" * 70)
        print(
            f"Lambda sweep 실행: "
            f"alpha={FIXED_ALPHA}, "
            f"lambda={lambda_value}, "
            f"W1/W2={FIXED_W1}/{FIXED_W2}"
        )
        print("=" * 70)

        os.environ["CVAR_LAMBDA"] = str(lambda_value)

        # 기존 4항 Proposed AR 코드 전체 실행
        runpy.run_path(
            MODEL_FILE,
            run_name=f"__lambda_{lambda_value}__"
        )

        if not os.path.exists(RESULT_NPZ):
            raise FileNotFoundError(
                f"결과 파일을 찾을 수 없습니다: {RESULT_NPZ}"
            )

        # 결과 읽기
        with np.load(
            RESULT_NPZ,
            allow_pickle=True
        ) as result:

            nrmse_value = float(
                result["overall_nrmse"]
            )

            gap_value = float(
                result["overall_gap"]
            )

            mean_regret_value = float(
                result["overall_mean_regret"]
            )

            cvar_regret_value = float(
                result["overall_cvar_regret"]
            )

            max_regret_value = float(
                result["overall_max_regret"]
            )

            total_realized_profit = float(
                np.sum(result["realized_profit"])
            )

            total_oracle_profit = float(
                np.sum(result["oracle_profit"])
            )

        sweep_rows.append({
            "CVAR_alpha":
                FIXED_ALPHA,

            "CVAR_lambda":
                lambda_value,

            "W1":
                FIXED_W1,

            "W2":
                FIXED_W2,

            "nRMSE_percent":
                nrmse_value,

            "optimality_gap_percent":
                gap_value,

            "mean_regret":
                mean_regret_value,

            "CVaR90_regret":
                cvar_regret_value,

            "max_regret":
                max_regret_value,

            "total_realized_profit":
                total_realized_profit,

            "total_oracle_profit":
                total_oracle_profit,
        })

        # Lambda별 NPZ도 따로 보관
        lambda_tag = (
            f"{lambda_value:.3f}"
            .replace(".", "p")
        )

        saved_npz_name = (
            "proposed_ar_4term_"
            f"alpha0p90_lambda{lambda_tag}_diagnostics.npz"
        )

        shutil.copy2(
            RESULT_NPZ,
            saved_npz_name
        )

        print(
            f"Lambda={lambda_value:.3f} 완료:",
            f"nRMSE={nrmse_value:.6f}%,",
            f"gap={gap_value:.6f}%,",
            f"CVaR90={cvar_regret_value:.6f}"
        )

finally:

    # 실행 전 환경변수 상태로 복원
    for key in environment_keys:

        previous_value = previous_environment[key]

        if previous_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous_value


# =====================================================================
# 5. 결과표 생성
# =====================================================================

lambda_sweep_table = pd.DataFrame(
    sweep_rows
)

# Lambda=0을 CVaR 도입 전 기준으로 사용
reference_row = (
    lambda_sweep_table[
        lambda_sweep_table["CVAR_lambda"] == 0.0
    ]
    .iloc[0]
)

lambda_sweep_table["delta_nRMSE_pp"] = (
    lambda_sweep_table["nRMSE_percent"]
    - reference_row["nRMSE_percent"]
)

lambda_sweep_table["delta_gap_pp"] = (
    lambda_sweep_table["optimality_gap_percent"]
    - reference_row["optimality_gap_percent"]
)

lambda_sweep_table["mean_regret_change_percent"] = (
    100.0
    * (
        lambda_sweep_table["mean_regret"]
        - reference_row["mean_regret"]
    )
    / reference_row["mean_regret"]
)

lambda_sweep_table["CVaR90_change_percent"] = (
    100.0
    * (
        lambda_sweep_table["CVaR90_regret"]
        - reference_row["CVaR90_regret"]
    )
    / reference_row["CVaR90_regret"]
)

lambda_sweep_table["max_regret_change_percent"] = (
    100.0
    * (
        lambda_sweep_table["max_regret"]
        - reference_row["max_regret"]
    )
    / reference_row["max_regret"]
)

lambda_sweep_table["profit_change_percent"] = (
    100.0
    * (
        lambda_sweep_table["total_realized_profit"]
        - reference_row["total_realized_profit"]
    )
    / abs(reference_row["total_realized_profit"])
)


# =====================================================================
# 6. 출력 및 CSV 저장
# =====================================================================

print()
print("=== 4항 Proposed AR: Lambda sweep 결과 ===")

print(
    lambda_sweep_table[
        [
            "CVAR_lambda",
            "nRMSE_percent",
            "delta_nRMSE_pp",
            "optimality_gap_percent",
            "delta_gap_pp",
            "mean_regret_change_percent",
            "CVaR90_change_percent",
            "max_regret_change_percent",
            "profit_change_percent",
        ]
    ]
    .round(6)
    .to_string(index=False)
)

lambda_sweep_table.to_csv(
    "lambda_sweep_proposed_ar_4term.csv",
    index=False,
    encoding="utf-8-sig"
)


# =====================================================================
# 7. Lambda 민감도 그래프
# =====================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

lambda_values = (
    lambda_sweep_table["CVAR_lambda"]
)

axes[0, 0].plot(
    lambda_values,
    lambda_sweep_table["nRMSE_percent"],
    marker="o"
)
axes[0, 0].set_title("nRMSE")
axes[0, 0].set_ylabel("%")

axes[0, 1].plot(
    lambda_values,
    lambda_sweep_table["optimality_gap_percent"],
    marker="o",
    color="tab:orange"
)
axes[0, 1].set_title("Optimality Gap")
axes[0, 1].set_ylabel("%")

axes[1, 0].plot(
    lambda_values,
    lambda_sweep_table["CVaR90_regret"],
    marker="o",
    color="tab:red"
)
axes[1, 0].set_title("CVaR90 Regret")

axes[1, 1].plot(
    lambda_values,
    lambda_sweep_table["max_regret"],
    marker="o",
    color="tab:purple"
)
axes[1, 1].set_title("Maximum Regret")

for one_axis in axes.flatten():

    one_axis.set_xlabel("CVaR lambda")
    one_axis.grid(
        True,
        alpha=0.3
    )

plt.suptitle(
    "Proposed AR 4-term: CVaR Lambda Sweep "
    "(W1/W2 = 1/20, alpha = 0.90)"
)

plt.tight_layout()

plt.savefig(
    "lambda_sweep_proposed_ar_4term.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print()
print(
    "저장 완료:",
    "lambda_sweep_proposed_ar_4term.csv"
)