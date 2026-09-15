# -*- coding: utf-8 -*-
"""
50조합 그리드/Fig 스윕을 다 돌리지 않고, solve_mlr(또는 solve_mlr_proposed) 정의까지만
실행한 뒤 KPI 지점 하나만 직접 호출해서 nRMSE/Gap을 뽑는다. 원본 파일은 전혀 수정하지 않음
(메모리 안에서 BASE_DIR만 고쳐서 실행).
사용: python kpi_only.py <파일명.py> <truncate_marker> <call_expr>
"""
import sys
import re
from pathlib import Path

DIR = Path(r"D:\Users\Downloads\Github\구현결과_보고서_지원")
target_name, marker, call_expr = sys.argv[1], sys.argv[2], sys.argv[3]

src = DIR / target_name
text = src.read_text(encoding="utf-8")

text, n = re.subn(
    r'BASE_DIR = os\.path\.abspath\(r".*?"\)',
    lambda m: f'BASE_DIR = r"{DIR}"',
    text,
    count=1,
)
print(f">>> BASE_DIR 치환: {n}건", file=sys.stderr)

idx = text.index(marker)
setup_code = text[:idx]

ns = {"__file__": str(src)}
exec(compile(setup_code, str(src), "exec"), ns)
print(">>> setup 실행 완료, KPI 지점 계산 중...", file=sys.stderr)

result = eval(call_expr, ns)
print(f">>> 결과: {result}")
