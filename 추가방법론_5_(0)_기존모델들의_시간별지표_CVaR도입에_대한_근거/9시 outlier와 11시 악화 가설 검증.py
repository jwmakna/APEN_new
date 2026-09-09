import numpy as np

PROFIT_TYPES = {
    # 원본(3항) 파일명은 네가 그때 저장한 이름으로 바꿔줘
    "원본": ("baseline_ar_combined_diagnostics.npz", "proposed_ar_combined_diagnostics.npz"),
    "4항":  ("baseline_ar_4term_combined_diagnostics.npz", "proposed_ar_4term_combined_diagnostics.npz"),
    "Zhang": ("baseline_ar_zhang_combined_diagnostics.npz", "proposed_ar_zhang_combined_diagnostics.npz"),
}

TARGET_HOURS = {9: 0, 11: 2}   # local_hour -> hour_idx (LOCAL_HOUR_START=9 기준)

for profit_label, (base_path, prop_path) in PROFIT_TYPES.items():

    base_data = np.load(base_path)
    prop_data = np.load(prop_path)

    base_regret = base_data["regret"]     # (n_test_days, 12)
    prop_regret = prop_data["regret"]
    dates = base_data["dates"]

    for local_hour, hour_idx in TARGET_HOURS.items():

        change = prop_regret[:, hour_idx] - base_regret[:, hour_idx]   # 음수 = Proposed 개선
        order = np.argsort(change)                                      # 개선 큰 날부터 정렬

        n_days = len(change)
        total_change = np.sum(change)
        top1_change = change[order[0]]
        top3_change = np.sum(change[order[:3]])

        print(f"\n=== {profit_label} / {local_hour}시: Proposed - Baseline regret 변화 ===")
        print(f"전체 {n_days}일 합계 변화 = {total_change:+.2f}")
        print(f"가장 큰 개선 하루({dates[order[0]]}) 기여 = {top1_change:+.2f} ({100*top1_change/total_change:.1f}%)")
        print(f"상위 3일 합계 기여 = {top3_change:+.2f} ({100*top3_change/total_change:.1f}%)")
        print("날짜별 상세 (개선 큰 순):")
        for rank in range(n_days):
            idx = order[rank]
            print(f"  {dates[idx]}: {change[idx]:+.4f}")