# -*- coding: utf-8 -*-
"""
Figure sweep 코드에서 스윕 리스트(RATIO_LIST/PENALTY_LIST/ZHANG_C_LIST 등)를
"KPI 지점 + 끝점 1개"짜리 2개 원소 리스트로 메모리 안에서만 축소한 뒤,
truncate_marker 직전까지만 실행해서 print_expr 을 출력한다.
원본 파일은 전혀 수정하지 않는다(디스크에 다시 쓰지 않음).

사용:
python sample2.py <파일명.py> <listname1>=<파이썬 리터럴> [<listname2>=<파이썬 리터럴> ...] -- <truncate_marker> <print_expr>
"""
import sys
import re
from pathlib import Path

DIR = Path(r"D:\Users\Downloads\Github\구현결과_보고서_지원")

args = sys.argv[1:]
target_name = args[0]
sep = args.index("--")
list_patches = args[1:sep]
marker = args[sep + 1]
print_expr = args[sep + 2]

src = DIR / target_name
text = src.read_text(encoding="utf-8")

text, n = re.subn(
    r'BASE_DIR = os\.path\.abspath\(r".*?"\)',
    lambda m: f'BASE_DIR = r"{DIR}"',
    text,
    count=1,
)
print(f">>> BASE_DIR 치환(구식 하드코딩 패턴): {n}건", file=sys.stderr)

for patch in list_patches:
    name, val = patch.split("=", 1)
    pattern = re.compile(re.escape(name) + r"\s*=\s*\[.*?\]", re.DOTALL)
    new_text, cnt = pattern.subn(lambda m, name=name, val=val: f"{name} = {val}", text, count=1)
    if cnt == 0:
        print(f"[경고] {name} 패턴을 못 찾음", file=sys.stderr)
    else:
        text = new_text
        print(f">>> {name} 축소 치환 완료 -> {val}", file=sys.stderr)

idx = text.index(marker)
setup_code = text[:idx]

ns = {"__file__": str(src)}
exec(compile(setup_code, str(src), "exec"), ns)
print(">>> setup 실행 완료", file=sys.stderr)

result = eval(print_expr, ns)
print(f">>> 결과: {result}")
