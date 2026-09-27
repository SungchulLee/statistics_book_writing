r"""matplotlib_basics 쪽의 깨진 그림 네 장을 한글이 보이게 다시 그린다.

docs/ch00/tools/img/ 의 네 장(_280, _332, _393, _446)은 한글 폰트를 지정하지
않은 채 만들어져 제목·축 이름·범례의 한글이 모두 두부(□)로 깨져 있었다.
나머지 여섯 장은 라벨이 영문이라 문제가 없다.

이 스크립트는 쪽에 실려 있는 코드 블록을 **그대로 읽어다가** 실행한다.
코드를 손으로 옮겨 적지 않으므로 그림과 본문이 어긋날 수 없다. 바꾸는 것은
폰트 설정과 plt.show() 를 savefig 로 바꾸는 것뿐이다.

실행:  python3 scripts/redraw_matplotlib_basics_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

PAGE = pathlib.Path("docs/ch00/tools/matplotlib_basics.md")
IMG = pathlib.Path("docs/ch00/tools/img")
TARGETS = ["280", "332", "393", "446"]


def python_blocks(lines):
    """쪽 안의 모든 ```python 블록을 (시작줄, 코드) 로 모은다."""
    blocks, start = [], None
    for k, line in enumerate(lines):
        if start is None:
            if line.strip().startswith("```python"):
                start = k
        elif line.strip() == "```":
            blocks.append((start, textwrap.dedent("\n".join(lines[start + 1:k]))))
            start = None
    return blocks


def code_block_before(lines, needle):
    """그림 줄보다 앞에 있는 마지막 python 블록을 고른다."""
    i = next(k for k, l in enumerate(lines) if needle in l)
    before = [(s, c) for s, c in python_blocks(lines) if s < i]
    return before[-1][1]


def main():
    lines = PAGE.read_text().split("\n")
    for tag in TARGETS:
        out = IMG / f"matplotlib_basics_{tag}.png"
        src = code_block_before(lines, f"matplotlib_basics_{tag}.png")

        saved = []

        def fake_show(*_args, **_kwargs):
            fig = plt.gcf()
            fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
            saved.append(out)
            plt.close(fig)

        ns = {"__name__": "__redraw__"}
        real_show = plt.show
        plt.show = fake_show
        try:
            exec(compile(src, f"<{out.name}>", "exec"), ns)
        finally:
            plt.show = real_show
            plt.close("all")

        if saved:
            print(f"saved {out}")
        else:
            print(f"!! {out} — plt.show() 가 호출되지 않았다")


if __name__ == "__main__":
    main()
