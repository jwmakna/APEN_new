# =====================================================================
# 4항 수익식
# Baseline AR vs Proposed AR 통합 비교
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# 1. 결과 불러오기
# =====================================================================

baseline_result = np.load(
    "baseline_ar_4term_combined_diagnostics.npz"
)

proposed_result = np.load(
    "proposed_ar_4term_combined_diagnostics.npz"
)

actual_baseline = baseline_result["actual"]
actual_proposed = proposed_result["actual"]

baseline_prediction = baseline_result["prediction"]
proposed_prediction = proposed_result["prediction"]

baseline_oracle = baseline_result["oracle_profit"]
proposed_oracle = proposed_result["oracle_profit"]

baseline_regret = baseline_result["regret"]
proposed_regret = proposed_result["regret"]

date_labels = baseline_result["dates"]
local_hours = baseline_result["hours"]


# =====================================================================
# 2. 동일한 테스트 데이터와 오라클인지 확인
# =====================================================================

if not np.allclose(
    actual_baseline,
    actual_proposed
):

    raise ValueError(
        "두 모형의 테스트 실제 발전량이 다릅니다."
    )

if not np.allclose(
    baseline_oracle,
    proposed_oracle
):

    raise ValueError(
        "두 모형의 4항 오라클 수익이 다릅니다."
    )

actual = actual_baseline


# =====================================================================
# 3. 예측오차 비교
# =====================================================================

baseline_error = (
    baseline_prediction - actual
)

proposed_error = (
    proposed_prediction - actual
)

baseline_absolute_error = (
    np.abs(baseline_error) * 100.0
)

proposed_absolute_error = (
    np.abs(proposed_error) * 100.0
)

baseline_hourly_rmse = (
    np.sqrt(
        np.mean(
            baseline_error ** 2,
            axis=0
        )
    )
    * 100.0
)

proposed_hourly_rmse = (
    np.sqrt(
        np.mean(
            proposed_error ** 2,
            axis=0
        )
    )
    * 100.0
)

# 음수이면 Proposed AR 개선
hourly_rmse_change = (
    proposed_hourly_rmse
    - baseline_hourly_rmse
)

absolute_error_change = (
    proposed_absolute_error
    - baseline_absolute_error
)


# =====================================================================
# 4. 경제적 regret 비교
# =====================================================================

baseline_mean_regret = np.mean(
    baseline_regret,
    axis=0
)

proposed_mean_regret = np.mean(
    proposed_regret,
    axis=0
)

# 음수이면 Proposed AR 개선
mean_regret_change = (
    proposed_mean_regret
    - baseline_mean_regret
)

regret_change = (
    proposed_regret
    - baseline_regret
)


# =====================================================================
# 5. 시간대별 optimality gap
# =====================================================================

oracle_sum_by_hour = np.sum(
    baseline_oracle,
    axis=0
)

baseline_regret_sum = np.sum(
    baseline_regret,
    axis=0
)

proposed_regret_sum = np.sum(
    proposed_regret,
    axis=0
)

baseline_gap_by_hour = np.full(
    len(local_hours),
    np.nan
)

proposed_gap_by_hour = np.full(
    len(local_hours),
    np.nan
)

np.divide(
    100.0 * baseline_regret_sum,
    oracle_sum_by_hour,
    out=baseline_gap_by_hour,
    where=np.abs(oracle_sum_by_hour) > 1e-12
)

np.divide(
    100.0 * proposed_regret_sum,
    oracle_sum_by_hour,
    out=proposed_gap_by_hour,
    where=np.abs(oracle_sum_by_hour) > 1e-12
)

gap_change_by_hour = (
    proposed_gap_by_hour
    - baseline_gap_by_hour
)


# =====================================================================
# 6. 전체 지표 출력
# =====================================================================

baseline_overall_nrmse = float(
    baseline_result["overall_nrmse"]
)

proposed_overall_nrmse = float(
    proposed_result["overall_nrmse"]
)

baseline_overall_gap = float(
    baseline_result["overall_gap"]
)

proposed_overall_gap = float(
    proposed_result["overall_gap"]
)

print()
print("=== 4항 수익식 AR 전체 성능 ===")

print(
    f"nRMSE: "
    f"{baseline_overall_nrmse:.6f}%"
    f" → {proposed_overall_nrmse:.6f}% "
    f"({proposed_overall_nrmse - baseline_overall_nrmse:+.6f}%p)"
)

print(
    f"optimality gap: "
    f"{baseline_overall_gap:.6f}%"
    f" → {proposed_overall_gap:.6f}% "
    f"({proposed_overall_gap - baseline_overall_gap:+.6f}%p)"
)


# =====================================================================
# 7. 시간대별 비교표
# =====================================================================

comparison_table = pd.DataFrame({
    "local_hour":
        local_hours,

    "Baseline_RMSE":
        baseline_hourly_rmse,

    "Proposed_RMSE":
        proposed_hourly_rmse,

    "RMSE_change":
        hourly_rmse_change,

    "Baseline_mean_regret":
        baseline_mean_regret,

    "Proposed_mean_regret":
        proposed_mean_regret,

    "mean_regret_change":
        mean_regret_change,

    "Baseline_gap_percent":
        baseline_gap_by_hour,

    "Proposed_gap_percent":
        proposed_gap_by_hour,

    "gap_change_percent_point":
        gap_change_by_hour,
})

print()
print("=== 4항 수익식 AR 시간대별 비교 ===")

print(
    comparison_table
    .round(6)
    .to_string(index=False)
)

comparison_table.to_csv(
    "ar_4term_hourly_combined_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)


# =====================================================================
# 8. Heatmap 범위
# =====================================================================

error_color_limit = np.max(
    np.abs(absolute_error_change)
)

if error_color_limit == 0.0:

    error_color_limit = 1.0

regret_color_limit = np.max(
    np.abs(regret_change)
)

if regret_color_limit == 0.0:

    regret_color_limit = 1.0


# =====================================================================
# 9. 통합 비교 그래프
# =====================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10)
)


# (a) 시간대별 RMSE
axes[0, 0].plot(
    local_hours,
    baseline_hourly_rmse,
    color="#2563EB",
    marker="o",
    linewidth=2,
    label="Baseline AR"
)

axes[0, 0].plot(
    local_hours,
    proposed_hourly_rmse,
    color="#F97316",
    marker="s",
    linewidth=2,
    label="Proposed AR"
)

axes[0, 0].set_xticks(local_hours)
axes[0, 0].set_xlabel("Local hour")
axes[0, 0].set_ylabel("RMSE (% of capacity)")
axes[0, 0].set_title("(a) Hourly RMSE — 4-term")
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()


# (b) 시간대별 평균 regret
axes[0, 1].plot(
    local_hours,
    baseline_mean_regret,
    color="#2563EB",
    marker="o",
    linewidth=2,
    label="Baseline AR"
)

axes[0, 1].plot(
    local_hours,
    proposed_mean_regret,
    color="#F97316",
    marker="s",
    linewidth=2,
    label="Proposed AR"
)

axes[0, 1].set_xticks(local_hours)
axes[0, 1].set_xlabel("Local hour")
axes[0, 1].set_ylabel("Mean economic regret")
axes[0, 1].set_title(
    "(b) Mean Economic Regret — 4-term"
)

axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend()


# (c) 절대예측오차 변화
error_heatmap = axes[1, 0].imshow(
    absolute_error_change,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-error_color_limit,
    vmax=error_color_limit,
    interpolation="nearest"
)

axes[1, 0].set_xticks(
    np.arange(len(local_hours))
)

axes[1, 0].set_xticklabels(local_hours)
axes[1, 0].set_xlabel("Local hour")
axes[1, 0].set_ylabel("Test date")

axes[1, 0].set_title(
    "(c) Absolute-Error Change — 4-term\n"
    "Proposed AR - Baseline AR"
)


# (d) 경제적 regret 변화
regret_heatmap = axes[1, 1].imshow(
    regret_change,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-regret_color_limit,
    vmax=regret_color_limit,
    interpolation="nearest"
)

axes[1, 1].set_xticks(
    np.arange(len(local_hours))
)

axes[1, 1].set_xticklabels(local_hours)
axes[1, 1].set_xlabel("Local hour")
axes[1, 1].set_ylabel("Test date")

axes[1, 1].set_title(
    "(d) Economic-Regret Change — 4-term\n"
    "Proposed AR - Baseline AR"
)


# 날짜 눈금
date_tick_positions = np.arange(
    0,
    len(date_labels),
    5
)

for heatmap_axis in [
    axes[1, 0],
    axes[1, 1]
]:

    heatmap_axis.set_yticks(
        date_tick_positions
    )

    heatmap_axis.set_yticklabels(
        date_labels[
            date_tick_positions
        ]
    )


error_colorbar = fig.colorbar(
    error_heatmap,
    ax=axes[1, 0],
    pad=0.02
)

error_colorbar.set_label(
    "Absolute-error change (%p)\n"
    "Blue = Proposed improvement"
)

regret_colorbar = fig.colorbar(
    regret_heatmap,
    ax=axes[1, 1],
    pad=0.02
)

regret_colorbar.set_label(
    "Economic-regret change\n"
    "Blue = Proposed improvement"
)

plt.tight_layout()

plt.savefig(
    "ar_4term_hourly_combined_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()