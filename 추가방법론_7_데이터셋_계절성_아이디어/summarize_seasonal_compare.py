# -*- coding: utf-8 -*-

# =====================================================================
# summarize_seasonal_compare.py
#
# 목적: run_seasonal_compare.sh 가 만든 results_seasonal_compare.tsv를
#       읽어서, "계절매칭이 가능했던 블록(13~24)에서만" 원래 방식과
#       계절매칭 방식의 평균 nRMSE·gap을 공정하게 비교한다.
#       (원래 방식은 24블록 전체 결과가 있지만, 계절매칭은 12블록뿐이라
#        그냥 전체 평균끼리 비교하면 불공정함 - 같은 12블록끼리만 비교)
#
# 사용법: python3 summarize_seasonal_compare.py
# =====================================================================

import pandas as pd

df = pd.read_csv("results_seasonal_compare.tsv", sep="\t")

# 계절매칭이 존재하는 블록 번호만 추출
seasonal_blocks = sorted(int(b) for b in df[df["approach"] == "seasonal"]["block"].unique())
print(f"계절매칭이 가능했던 블록: {seasonal_blocks} (총 {len(seasonal_blocks)}개)")
print()

matched = df[df["block"].isin(seasonal_blocks)]

print("=== 같은 블록(계절매칭 가능한 12개)에서 원래 방식 vs 계절매칭 방식 평균 ===")
summary = matched.groupby(["approach", "model"])[["nRMSE", "gap"]].mean().round(4)
print(summary)
print()

print("=== 블록별 상세 (gap 기준, 원래 vs 계절매칭) ===")
for model in sorted(df["model"].unique()):
    print(f"--- {model} ---")
    sub = matched[matched["model"] == model]
    pivot = sub.pivot(index="block", columns="approach", values="gap")
    if "original" in pivot.columns and "seasonal" in pivot.columns:
        pivot["개선(원래-계절)"] = pivot["original"] - pivot["seasonal"]
        print(pivot.round(4).to_string())
    print()

print("=== 참고: 원래 방식 24블록 전체 평균 (계절매칭 없는 1~12번 포함) ===")
orig_all = df[df["approach"] == "original"].groupby("model")[["nRMSE", "gap"]].mean().round(4)
print(orig_all)
