r"""5.3 표준오차 페이지의 그림 세 장을 다시 그린다.

그림을 따로 손으로 그리지 않고 **페이지에 실린 코드 블록을 그대로 실행한다.**
본문의 코드와 그림이 어긋날 수 없게 하려는 것이다. 바꾸는 것은 두 가지뿐이다.

  - 글꼴: 실행 환경에 한글 글꼴이 없을 수 있으므로 찾아서 지정한다.
  - plt.show(): 화면에 띄우는 대신 PNG 로 저장한다.

페이지에는 파이썬 블록이 넷 있고 그중 그림을 그리는 것은 셋이다.

  블록 1  예제 1  한 파일 판                  -> se_xbar_single_file.png
  블록 2  예제 2  global_name_space.py        (설정만, 그림 없음)
  블록 3  예제 3  standard_error_of_x_bar.py  -> se_xbar_module.png
  블록 4  예제 4  standard_error_of_s_square.py -> se_s2_module.png

블록 3 과 4 는 서로, 그리고 블록 2 를 import 하므로 임시 디렉터리에 실제
파일로 풀어 놓고 실행한다. 시드는 블록 2 의 기본값(1)을 쓴다.

실행:  python3 scripts/redraw_ch05_standard_error_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG 는 커밋되므로 CI 에서 다시 그리지 않는다.
"""

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PAGE = pathlib.Path("docs/ch05/applications/standard_error.md")
OUT = pathlib.Path("docs/ch05/applications/img")

# 블록 번호(0부터) -> (파일 이름, 저장할 PNG 이름)
PLAN = {
    0: ("standard_error_single_file.py", "se_xbar_single_file.png"),
    1: ("global_name_space.py", None),
    2: ("standard_error_of_x_bar.py", "se_xbar_module.png"),
    3: ("standard_error_of_s_square.py", "se_s2_module.png"),
}

DPI = 130          # 원래 커밋된 그림과 같은 해상도

# 각 파일을 새 프로세스에서 실행하는 껍데기. plt.show() 를 savefig 로 바꾸고,
# 이 컴퓨터에 실제로 있는 한글 글꼴을 입힌다.
RUNNER = r'''
import runpy, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

target, out, font, dpi = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

def fake_show(*_a, **_k):
    fig = plt.gcf()
    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontfamily(font)
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

plt.show = fake_show
sys.argv = [target]        # argparse 가 이쪽 인자를 보지 않도록
runpy.run_path(target, run_name="__main__")
'''


def korean_font():
    """이 컴퓨터에 있는 한글 글꼴을 하나 고른다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic",
                 "NanumGothic", "Noto Sans CJK KR"):
        if name in installed:
            return name
    raise SystemExit("한글 글꼴을 찾지 못했다. NanumGothic 등을 설치해야 한다.")


def python_blocks(text):
    return re.findall(r"```python\n(.*?)```", text, re.S)


def main():
    # 페이지 뒤쪽 연습문제에도 파이썬 블록이 있다. 예제는 앞의 넷이다.
    blocks = python_blocks(PAGE.read_text(encoding="utf-8"))
    if len(blocks) < len(PLAN):
        raise SystemExit(f"파이썬 블록이 {len(blocks)}개뿐이다. "
                         f"앞의 {len(PLAN)}개가 필요하다.")

    font = korean_font()
    work = pathlib.Path(tempfile.mkdtemp(prefix="se_figs_"))
    try:
        for i, (fname, _) in PLAN.items():
            (work / fname).write_text(blocks[i], encoding="utf-8")
        (work / "_runner.py").write_text(RUNNER, encoding="utf-8")

        # 파일마다 **새 프로세스**로 돌린다. 한 프로세스에서 이어 돌리면
        # 앞 파일이 난수를 써 버려 뒤 파일의 결과가 달라지고, global_name_space
        # 가 이미 import 된 상태라 시드가 다시 걸리지도 않는다.
        for i, (fname, png) in PLAN.items():
            if png is None:
                continue
            print(f"블록 {i + 1}  ({fname})")
            r = subprocess.run(
                [sys.executable, str(work / "_runner.py"), fname,
                 str(OUT.resolve() / png), font, str(DPI)],
                cwd=work, capture_output=True, text=True)
            sys.stdout.write(r.stdout)
            if r.returncode:
                sys.stderr.write(r.stderr)
                raise SystemExit(f"{fname} 실행 실패")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
