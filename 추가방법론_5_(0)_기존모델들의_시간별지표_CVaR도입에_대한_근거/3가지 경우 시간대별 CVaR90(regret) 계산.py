# =====================================================================
# 시간대별 CVaR90(regret) 계산 - A안(평가 지표로만 CVaR)
# 재학습 없이, 이미 저장된 *_combined_diagnostics.npz 의 regret 배열만 사용
#
# CVaR90(regret) = 그 시간대 30일 중 regret이 가장 큰 상위 10%(=3일)의 평균
# =====================================================================

import numpy as np
import pandas as pd

CVAR_TAIL_FRACTION = 0.10   # 상위 몇 %를 tail로 볼지 (90% 기준 -> 상위 10%)

PROFIT_TYPES = {
    "원본":  ("baseline_ar_combined_diagnostics.npz",       "proposed_ar_combined_diagnostics.npz"),
    "4항":   ("baseline_ar_4term_combined_diagnostics.npz", "proposed_ar_4term_combined_diagnostics.npz"),
    "Zhang": ("baseline_ar_zhang_combined_diagnostics.npz", "proposed_ar_zhang_combined_diagnostics.npz"),
}


# =====================================================================
# 1. 시간대별 CVaR90 계산 함수 (def 안 쓰는 코딩스타일 유지 - 반복문으로 직접 계산)
# =====================================================================

result_rows = []   # 최종 표에 넣을 행들을 담을 빈 리스트

for profit_label, (base_path, prop_path) in PROFIT_TYPES.items():   # 원본/4항/Zhang 순서대로

    base_data = np.load(base_path, allow_pickle=True)
    prop_data = np.load(prop_path, allow_pickle=True)

    base_regret = base_data["regret"]     # (n_test_days, 12) - baseline 시점별 regret
    prop_regret = prop_data["regret"]     # (n_test_days, 12) - proposed 시점별 regret
    local_hours = base_data["hours"]      # 시간대 라벨 (9~20)

    n_test_days = base_regret.shape[0]                                 # 테스트 일수 (예: 30)
    n_tail = int(np.ceil(n_test_days * CVAR_TAIL_FRACTION))            # tail로 볼 날짜 수 (30일이면 3일)
    if n_tail < 1:
        n_tail = 1                                                       # 최소 1일은 보장

    for hour_idx in range(len(local_hours)):                            # 시간대 0~11(=9~20시)을 하나씩

        local_hour = int(local_hours[hour_idx])                          # 실제 표시할 시각(9~20)

        base_col = base_regret[:, hour_idx]                              # 이 시간대의 baseline regret 30개
        prop_col = prop_regret[:, hour_idx]                              # 이 시간대의 proposed regret 30개

        base_sorted_desc = np.sort(base_col)[::-1]                       # 큰 값부터 정렬 (regret 큰 순)
        prop_sorted_desc = np.sort(prop_col)[::-1]

        base_mean = np.mean(base_col)                                    # 평균 regret
        prop_mean = np.mean(prop_col)

        base_p90 = np.percentile(base_col, 90)                           # P90 regret
        prop_p90 = np.percentile(prop_col, 90)

        base_max = np.max(base_col)                                      # 최댓값 regret
        prop_max = np.max(prop_col)

        base_cvar90 = np.mean(base_sorted_desc[:n_tail])                  # 상위 n_tail일 평균 = CVaR90
        prop_cvar90 = np.mean(prop_sorted_desc[:n_tail])

        result_rows.append({
            "profit_type": profit_label,
            "local_hour": local_hour,
            "baseline_mean_regret": base_mean,
            "proposed_mean_regret": prop_mean,
            "mean_regret_change": prop_mean - base_mean,          # 음수 = Proposed가 평균적으로 개선
            "baseline_P90_regret": base_p90,
            "proposed_P90_regret": prop_p90,
            "baseline_max_regret": base_max,
            "proposed_max_regret": prop_max,
            "baseline_CVaR90_regret": base_cvar90,
            "proposed_CVaR90_regret": prop_cvar90,
            "CVaR90_change": prop_cvar90 - base_cvar90,            # 음수 = Proposed가 tail(꼬리위험)도 개선
        })


# =====================================================================
# 2. 표로 정리 + 저장
# =====================================================================

cvar_table = pd.DataFrame(result_rows)

print()
print("=== 시간대별 CVaR90(regret) 비교 (원본 / 4항 / Zhang) ===")
print(cvar_table.round(2).to_string(index=False))

cvar_table.to_csv(
    "hourly_cvar90_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("저장 완료: hourly_cvar90_comparison.csv")


# =====================================================================
# 3. "mean은 개선인데 CVaR90은 악화"인 시간대만 따로 뽑기
#    - 평균 지표와 꼬리위험 지표가 반대로 움직이는 곳이 CVaR 도입의 핵심 근거
# =====================================================================

diverging = cvar_table[
    (cvar_table["mean_regret_change"] < 0) &     # 평균은 개선(음수)인데
    (cvar_table["CVaR90_change"] > 0)             # CVaR90은 악화(양수)인 경우
]

print()
print("=== 평균은 개선, CVaR90(꼬리위험)은 악화된 시간대 ===")
if len(diverging) == 0:
    print("해당 없음")
else:
    print(diverging.round(2).to_string(index=False))