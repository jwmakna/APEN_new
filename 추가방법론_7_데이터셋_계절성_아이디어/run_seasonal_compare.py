# -*- coding: utf-8 -*-

# =====================================================================
# run_seasonal_compare.py
#
# run_seasonal_compare.sh 의 순수 파이썬 버전 (Windows에서도 그냥 실행됨,
# bash/WSL/Git Bash 필요 없음).
#
# 목적: block_dates_seasonal_compare.tsv 에 있는 24개 블록 중
#       계절매칭이 가능한 블록(13~24)에 대해 "원래 방식(직전 90일 학습)"과
#       "계절매칭 방식(1년 전 90일 학습)"을 AR/MLR 각각 돌려서
#       nRMSE·gap을 비교한다.
#
# 사용법 (커맨드 프롬프트/PowerShell 아무 곳에서나):
#   같은 폴더에 다음 파일들을 전부 놓고:
#     - merged_for_simulation_z03.csv
#     - model_proposed_ar.py                   (원래 방식 AR)
#     - model_proposed_mlr.py                  (원래 방식 MLR)
#     - model_proposed_ar_seasonal_train.py     (계절매칭 AR)
#     - model_proposed_mlr_seasonal_train.py    (계절매칭 MLR)
#     - block_dates_seasonal_compare.tsv
#   그 폴더에서:
#     python run_seasonal_compare.py
#
# 결과: results_seasonal_compare.tsv (block, approach, model, nRMSE, gap)
# =====================================================================

import os
import re
import sys
import subprocess
import pandas as pd

PYTHON_EXE = sys.executable   # 지금 이 스크립트를 실행 중인 파이썬을 그대로 재사용 (python/python3 헷갈릴 일 없음)

block_table = pd.read_csv("block_dates_seasonal_compare.tsv", sep="\t")

result_rows = []   # 각 실행 결과를 담을 리스트

for _, block_row in block_table.iterrows():

    block_num = int(block_row["block"])
    test_start = block_row["test_start"]
    test_end = block_row["test_end"]
    orig_train_start = block_row["orig_train_start"]
    orig_train_end = block_row["orig_train_end"]
    seasonal_feasible = block_row["seasonal_feasible"]   # True/False (문자열일 수도 있어 아래서 처리)

    print(f"=== 블록 {block_num} (테스트 {test_start}~{test_end}) ===")

    # ---- 원래 방식(직전 90일 학습): AR, MLR ----
    original_scripts = [("ar", "model_proposed_ar.py"), ("mlr", "model_proposed_mlr.py")]
    for model_name, script_name in original_scripts:
        env_vars = os.environ.copy()
        env_vars["TRAIN_START"] = str(orig_train_start)
        env_vars["TRAIN_END"] = str(orig_train_end)
        env_vars["TEST_START"] = str(test_start)
        env_vars["TEST_END"] = str(test_end)

        run_result = subprocess.run(
            [PYTHON_EXE, script_name],
            env=env_vars,
            capture_output=True,
            text=True,
        )
        output_text = run_result.stdout + run_result.stderr

        nrmse_match = re.search(r"nRMSE\s*=\s*([0-9.]+)", output_text)
        gap_match = re.search(r"optimality gap\s*=\s*(-?[0-9.]+)", output_text)
        nrmse_value = nrmse_match.group(1) if nrmse_match else ""
        gap_value = gap_match.group(1) if gap_match else ""

        if nrmse_value == "" or gap_value == "":
            print(f"  [경고] 원래-{model_name}: 결과를 못 찾음 (스크립트 에러일 수 있음)")
            print("  ----- 스크립트 출력 마지막 20줄 -----")
            for line in output_text.splitlines()[-20:]:
                print("   ", line)
        else:
            print(f"  원래-{model_name}: nRMSE={nrmse_value}% gap={gap_value}%")

        result_rows.append({
            "block": block_num, "approach": "original", "model": model_name,
            "nRMSE": nrmse_value, "gap": gap_value,
        })

    # ---- 계절매칭 방식(1년 전 90일 학습): 가능한 블록만 ----
    is_feasible = str(seasonal_feasible).strip().lower() == "true"
    if is_feasible:
        seasonal_scripts = [("ar", "model_proposed_ar_seasonal_train.py"), ("mlr", "model_proposed_mlr_seasonal_train.py")]
        for model_name, script_name in seasonal_scripts:
            env_vars = os.environ.copy()
            env_vars["TEST_START"] = str(test_start)
            env_vars["TEST_END"] = str(test_end)
            # TRAIN_START/TRAIN_END 는 일부러 안 줌 - 스크립트가 TEST 기준으로 자동 계산함

            run_result = subprocess.run(
                [PYTHON_EXE, script_name],
                env=env_vars,
                capture_output=True,
                text=True,
            )
            output_text = run_result.stdout + run_result.stderr

            nrmse_match = re.search(r"nRMSE\s*=\s*([0-9.]+)", output_text)
            gap_match = re.search(r"optimality gap\s*=\s*(-?[0-9.]+)", output_text)
            nrmse_value = nrmse_match.group(1) if nrmse_match else ""
            gap_value = gap_match.group(1) if gap_match else ""

            if nrmse_value == "" or gap_value == "":
                print(f"  [경고] 계절-{model_name}: 결과를 못 찾음 (스크립트 에러일 수 있음)")
                print("  ----- 스크립트 출력 마지막 20줄 -----")
                for line in output_text.splitlines()[-20:]:
                    print("   ", line)
            else:
                print(f"  계절-{model_name}: nRMSE={nrmse_value}% gap={gap_value}%")

            result_rows.append({
                "block": block_num, "approach": "seasonal", "model": model_name,
                "nRMSE": nrmse_value, "gap": gap_value,
            })
    else:
        print("  (계절매칭 데이터 부족 - 건너뜀)")

result_table = pd.DataFrame(result_rows)
result_table.to_csv("results_seasonal_compare.tsv", sep="\t", index=False)
print()
print("전체 완료 -> results_seasonal_compare.tsv")
print()
print("이어서 다음을 실행하면 블록별 평균 비교가 나와:")
print("  python summarize_seasonal_compare.py")
