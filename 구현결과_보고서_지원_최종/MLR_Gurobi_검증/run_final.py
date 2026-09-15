# -*- coding: utf-8 -*-
"""
원본 파일을 전혀 건드리지 않고, 메모리 안에서만 BASE_DIR 한 줄을 올바른 경로로
바꿔서 그대로 실행한다(파일에 다시 쓰지 않음 - 검증 목적의 1회성 실행).
사용: python run_final.py <파일이름.py>
"""
import sys
import re
import runpy
from pathlib import Path

DIR = Path(r"D:\Users\Downloads\Github\구현결과_보고서_지원")
target_name = sys.argv[1]
src = DIR / target_name
text = src.read_text(encoding="utf-8")

new_text, n = re.subn(
    r'BASE_DIR = os\.path\.abspath\(r".*?"\)',
    lambda m: f'BASE_DIR = r"{DIR}"',
    text,
    count=1,
)
if n == 0:
    print("[경고] BASE_DIR 패턴을 못 찾음 - 원본 그대로 실행 시도")
    new_text = text

tmp = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"C:\Users\moon1\AppData\Local\Temp\claude\d--Users-Downloads-Github\94d1b6c0-3232-4d3b-b140-a6b376a8ba31\scratchpad") / ("_run_" + target_name)
tmp.write_text(new_text, encoding="utf-8")
print(f">>> 임시 실행용 사본 생성(원본 미수정): {tmp}")
runpy.run_path(str(tmp), run_name="__main__")
