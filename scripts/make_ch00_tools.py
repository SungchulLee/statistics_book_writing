r"""0장 4절(도구)의 코드 세 쪽에 들어갈 개념 도식 세 장을 생성한다.

세 쪽 모두 코드를 가르치는 쪽이므로, 그림은 코드 화면을 흉내 내지 않고
초보자가 가장 자주 틀리는 자리의 **머릿속 모형**을 그린다.

만드는 파일:

  ch00/tools/img/name_vs_object.png
      파이썬 기초 — 대입은 복사가 아니라 이름표 붙이기다.
      같은 리스트에 이름 둘이 붙었을 때와 .copy() 로 떼어 냈을 때를 나란히 그린다.

  ch00/tools/img/broadcasting_rule.png
      NumPy — 브로드캐스팅. 모양을 뒤에서부터 맞추고 크기 1인 축을 늘린다.
      맞춰지는 경우 (3,1)+(3,) 와 맞지 않아 ValueError 가 나는 (5,)*(3,) 를 함께 보인다.

  ch00/tools/img/groupby_split_apply_combine.png
      pandas — groupby 의 분할·적용·결합. 같은 분할과 같은 집계에서
      agg 는 그룹당 한 행으로 줄이고 transform 은 원래 행 수로 되돌린다.

그림에 적힌 값은 모두 실제로 실행해 얻은 것이다(numpy 1.26, pandas 2.2).
한글은 mathtext 에 글리프가 없으므로 수식 바깥에만 쓴다.

실행:  python3 scripts/make_ch00_tools.py   (저장소 최상위에서)
필요:  matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG 로 커밋되므로 CI 에서 다시 그리지 않는다.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# === 공통 설정 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch00/tools/img/"
MONO = "monospace"


# === 공통 도우미 ===
def blank_axes(ax, xlim, ylim):
    """눈금과 테두리를 지운 도화지."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("auto")


def arrow(ax, p, q, color=INK, lw=1.6, scale=14, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=scale,
                                 color=color, linewidth=lw,
                                 shrinkA=0, shrinkB=0, zorder=5))


def tag(ax, x, ycen, name, color, face, w=1.5, h=0.85, fs=12):
    """이름표(변수 이름) 하나. 오른쪽 끝의 x 좌표를 돌려준다."""
    ax.add_patch(FancyBboxPatch((x, ycen - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.16",
                                facecolor=face, edgecolor=color,
                                linewidth=1.6, zorder=4))
    ax.text(x + w / 2, ycen, name, ha="center", va="center",
            fontsize=fs, color=color, family=MONO, zorder=5)
    return x + w


def cells(ax, x, ybot, values, edge, faces, cw=1.12, ch=1.0, fs=11.5,
          dashed_from=None):
    """값 하나에 칸 하나. dashed_from 이후의 칸은 점선(복제된 칸)으로 그린다."""
    for i, v in enumerate(values):
        dashed = dashed_from is not None and i >= dashed_from
        ax.add_patch(Rectangle((x + i * cw, ybot), cw, ch,
                               facecolor=faces[i], edgecolor=edge,
                               linewidth=1.3, zorder=3,
                               linestyle=(0, (3, 2)) if dashed else "-"))
        ax.text(x + (i + 0.5) * cw, ybot + ch / 2, str(v),
                ha="center", va="center", fontsize=fs,
                color=INK if not dashed else MUTED, family=MONO, zorder=4)
    return x + len(values) * cw


def obj_box(ax, x, ycen, values, edge, face, ch=1.0, cw=1.12, pad=0.3):
    """리스트 객체 하나 — 칸들을 둥근 테두리로 감싼다. 오른쪽 끝 x 를 돌려준다."""
    n = len(values)
    ax.add_patch(FancyBboxPatch((x - pad, ycen - ch / 2 - pad),
                                n * cw + 2 * pad, ch + 2 * pad,
                                boxstyle="round,pad=0,rounding_size=0.2",
                                facecolor="white", edgecolor=edge,
                                linewidth=1.7, zorder=2))
    cells(ax, x, ycen - ch / 2, values, edge, [face] * n, cw=cw, ch=ch)
    return x + n * cw + pad


def draw_table(ax, x, ytop, headers, rows, colw, rowh=0.8,
               header_face="#ECEFF1", row_faces=None, edge=INK,
               fs=10.5, title=None, title_color=INK, title_fs=11):
    """머리글 한 줄과 자료 여러 줄로 이루어진 표. (너비, 높이) 를 돌려준다."""
    cx = x
    for j, h in enumerate(headers):
        ax.add_patch(Rectangle((cx, ytop - rowh), colw[j], rowh,
                               facecolor=header_face, edgecolor=edge,
                               linewidth=1.1, zorder=3))
        ax.text(cx + colw[j] / 2, ytop - rowh / 2, h, ha="center", va="center",
                fontsize=fs, color=INK, zorder=4)
        cx += colw[j]
    for i, r in enumerate(rows):
        y = ytop - (i + 2) * rowh
        face = row_faces[i] if row_faces else "white"
        cx = x
        for j, v in enumerate(r):
            ax.add_patch(Rectangle((cx, y), colw[j], rowh, facecolor=face,
                                   edgecolor=edge, linewidth=1.1, zorder=3))
            ax.text(cx + colw[j] / 2, y + rowh / 2, str(v),
                    ha="center", va="center", fontsize=fs, color=INK,
                    family=MONO, zorder=4)
            cx += colw[j]
    if title is not None:
        ax.text(x, ytop + 0.24, title, ha="left", va="bottom",
                fontsize=title_fs, color=title_color)
    return sum(colw), (len(rows) + 1) * rowh


# === 그림 1. 이름과 객체 ===
def fig_name_vs_object(path=OUT + "name_vs_object.png"):
    """대입이 무엇을 복사하고 무엇을 복사하지 않는지를 그린다."""
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 6.9))
    X0 = [0.4, 11.6, 22.8]           # 세 장면의 왼쪽 끝
    PW = 10.6                        # 장면 하나의 너비
    NAME_X = 0.7                     # 이름표 x (장면 기준)
    OBJ_X = 4.9                      # 객체 첫 칸 x (장면 기준)

    def scene_frame(ax, k, code):
        x0 = X0[k]
        if k > 0:
            ax.plot([x0 - 0.55, x0 - 0.55], [0.55, 8.5], color=MUTED,
                    linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)
        ax.text(x0 + 0.1, 9.45, code, ha="left", va="top", fontsize=12.5,
                color=PURPLE, family=MONO)
        return x0

    # --- 윗줄: 별칭 ---
    ax = axes[0]
    blank_axes(ax, (0, 33.4), (0, 10.2))
    ax.set_title("가변 객체에 이름 둘이 붙으면 — 별칭",
                 loc="left", fontsize=13.5, color=INK, pad=8)

    x0 = scene_frame(ax, 0, "a = [1, 2, 3]")
    xr = tag(ax, x0 + NAME_X, 5.0, "a", BLUE, BLUE_F)
    obj_box(ax, x0 + OBJ_X, 5.0, [1, 2, 3], INK, "#F7F9FA")
    arrow(ax, (xr + 0.25, 5.0), (x0 + OBJ_X - 0.45, 5.0), color=BLUE)
    ax.text(x0 + OBJ_X - 0.35, 6.55, "리스트 객체", ha="left", va="center",
            fontsize=11, color=MUTED)

    x0 = scene_frame(ax, 1, "b = a")
    xr = tag(ax, x0 + NAME_X, 6.4, "a", BLUE, BLUE_F)
    tag(ax, x0 + NAME_X, 3.6, "b", ORANGE, ORANGE_F)
    obj_box(ax, x0 + OBJ_X, 5.0, [1, 2, 3], INK, "#F7F9FA")
    arrow(ax, (xr + 0.25, 6.4), (x0 + OBJ_X - 0.45, 5.35), color=BLUE)
    arrow(ax, (xr + 0.25, 3.6), (x0 + OBJ_X - 0.45, 4.65), color=ORANGE)
    ax.text(x0 + 0.1, 1.5, "새 객체는 만들어지지 않는다.\n이름표 하나가 더 붙었을 뿐이다.",
            ha="left", va="center", fontsize=11, color=INK)

    x0 = scene_frame(ax, 2, "b.append(4)")
    xr = tag(ax, x0 + NAME_X, 6.4, "a", BLUE, BLUE_F)
    tag(ax, x0 + NAME_X, 3.6, "b", ORANGE, ORANGE_F)
    obj_box(ax, x0 + OBJ_X, 5.0, [1, 2, 3, 4], INK, "#F7F9FA")
    arrow(ax, (xr + 0.25, 6.4), (x0 + OBJ_X - 0.45, 5.35), color=BLUE)
    arrow(ax, (xr + 0.25, 3.6), (x0 + OBJ_X - 0.45, 4.65), color=ORANGE)
    ax.text(x0 + 0.1, 1.9, "건드리지 않은 a 까지 [1, 2, 3, 4] 가 된다.",
            ha="left", va="center", fontsize=11.5, color=RED)
    ax.text(x0 + 0.1, 0.95, "a is b 는 True", ha="left", va="center",
            fontsize=11.5, color=RED)

    # --- 아랫줄: 복사 ---
    ax = axes[1]
    blank_axes(ax, (0, 33.4), (0, 10.2))
    ax.set_title("복사본을 만들면 — 이름마다 다른 객체",
                 loc="left", fontsize=13.5, color=INK, pad=8)

    x0 = scene_frame(ax, 0, "a = [1, 2, 3]")
    xr = tag(ax, x0 + NAME_X, 5.0, "a", BLUE, BLUE_F)
    obj_box(ax, x0 + OBJ_X, 5.0, [1, 2, 3], INK, "#F7F9FA")
    arrow(ax, (xr + 0.25, 5.0), (x0 + OBJ_X - 0.45, 5.0), color=BLUE)

    x0 = scene_frame(ax, 1, "b = a.copy()")
    xr = tag(ax, x0 + NAME_X, 6.6, "a", BLUE, BLUE_F)
    tag(ax, x0 + NAME_X, 3.4, "b", ORANGE, ORANGE_F)
    obj_box(ax, x0 + OBJ_X, 6.6, [1, 2, 3], INK, "#F7F9FA")
    obj_box(ax, x0 + OBJ_X, 3.4, [1, 2, 3], INK, "#F7F9FA")
    arrow(ax, (xr + 0.25, 6.6), (x0 + OBJ_X - 0.45, 6.6), color=BLUE)
    arrow(ax, (xr + 0.25, 3.4), (x0 + OBJ_X - 0.45, 3.4), color=ORANGE)
    ax.text(x0 + 0.1, 1.4, "값이 같은 객체가 둘 생겼다.", ha="left",
            va="center", fontsize=11, color=INK)

    x0 = scene_frame(ax, 2, "b.append(4)")
    xr = tag(ax, x0 + NAME_X, 6.6, "a", BLUE, BLUE_F)
    tag(ax, x0 + NAME_X, 3.4, "b", ORANGE, ORANGE_F)
    obj_box(ax, x0 + OBJ_X, 6.6, [1, 2, 3], INK, "#F7F9FA")
    obj_box(ax, x0 + OBJ_X, 3.4, [1, 2, 3, 4], INK, GREEN_F)
    arrow(ax, (xr + 0.25, 6.6), (x0 + OBJ_X - 0.45, 6.6), color=BLUE)
    arrow(ax, (xr + 0.25, 3.4), (x0 + OBJ_X - 0.45, 3.4), color=ORANGE)
    ax.text(x0 + 0.1, 1.9, "a 는 [1, 2, 3] 그대로다.", ha="left",
            va="center", fontsize=11.5, color=GREEN)
    ax.text(x0 + 0.1, 0.95, "a is b 는 False", ha="left", va="center",
            fontsize=11.5, color=GREEN)

    fig.subplots_adjust(hspace=0.22, top=0.90, bottom=0.03,
                        left=0.01, right=0.99)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# === 그림 2. 브로드캐스팅 ===
def fig_broadcasting(path=OUT + "broadcasting_rule.png"):
    """모양이 다른 두 배열이 어떻게 맞춰지는지, 그리고 언제 맞지 않는지."""
    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.62, 1.0],
                          hspace=0.30, wspace=0.16,
                          left=0.02, right=0.98, top=0.93, bottom=0.03)
    ax = fig.add_subplot(gs[0, :])
    blank_axes(ax, (0, 18.4), (0, 8.4))

    CW = 1.22
    col_vals = [1, 2, 3]            # 모양 (3, 1)
    row_vals = [10, 20, 30]         # 모양 (3,)
    res = [[11, 21, 31], [12, 22, 32], [13, 23, 33]]   # 실행해 얻은 값

    def grid3(x, ybot, values, edge, face, keep):
        """3x3 격자. keep='col' 이면 0열만, 'row' 면 0행만 원본이다."""
        for i in range(3):          # i = 행 번호(위가 0행)
            for j in range(3):
                original = (keep == "col" and j == 0) or \
                           (keep == "row" and i == 0) or keep == "all"
                yb = ybot + (2 - i) * CW
                ax.add_patch(Rectangle((x + j * CW, yb), CW, CW,
                                       facecolor=face if original else "white",
                                       edgecolor=edge if original else MUTED,
                                       linewidth=1.5 if original else 1.0,
                                       linestyle="-" if original else (0, (3, 2)),
                                       zorder=3))
                ax.text(x + (j + 0.5) * CW, yb + CW / 2, str(values[i][j]),
                        ha="center", va="center", family=MONO,
                        fontsize=12 if original else 11,
                        color=INK if original else MUTED, zorder=4)

    A = [[v] * 3 for v in col_vals]
    B = [list(row_vals) for _ in range(3)]

    xa, xb, xc = 1.3, 7.0, 12.7
    yb = 2.3
    grid3(xa, yb, A, BLUE, BLUE_F, "col")
    grid3(xb, yb, B, ORANGE, ORANGE_F, "row")
    grid3(xc, yb, res, GREEN, GREEN_F, "all")

    ax.text(xa + 3 * CW + 0.75, yb + 1.5 * CW, "+", ha="center", va="center",
            fontsize=24, color=INK)
    ax.text(xb + 3 * CW + 0.75, yb + 1.5 * CW, "=", ha="center", va="center",
            fontsize=24, color=INK)

    ax.text(xa, yb + 3 * CW + 0.55, "col  —  모양 (3, 1)", ha="left",
            va="bottom", fontsize=12.5, color=BLUE)
    ax.text(xb, yb + 3 * CW + 0.55, "row  —  모양 (3,)", ha="left",
            va="bottom", fontsize=12.5, color=ORANGE)
    ax.text(xc, yb + 3 * CW + 0.55, "col + row  —  모양 (3, 3)", ha="left",
            va="bottom", fontsize=12.5, color=GREEN)

    ax.text(xa, 1.25, "한 열이 세 열로", ha="left", va="center",
            fontsize=11, color=MUTED)
    ax.text(xb, 1.25, "한 행이 세 행으로", ha="left", va="center",
            fontsize=11, color=MUTED)
    ax.text(xc, 1.25, "실제로 복제하지는 않는다", ha="left", va="center",
            fontsize=11, color=MUTED)
    ax.text(0.2, 0.35, "진한 칸이 메모리에 실제로 있는 값이고, 점선 칸은 늘어난 것처럼 "
                       "취급되는 자리다.",
            ha="left", va="center", fontsize=11.5, color=INK)

    # --- 아랫줄 왼쪽: 맞춰지는 경우 ---
    axl = fig.add_subplot(gs[1, 0])
    blank_axes(axl, (0, 11.6), (0, 5.4))
    axl.set_title("뒤에서부터 축을 맞춘다", loc="left", fontsize=12.5,
                  color=INK, pad=6)
    CX = [3.4, 5.3]                 # 축 0, 축 1 칸의 왼쪽 x
    BW, BH = 1.55, 0.82

    def sbox(a, x, ycen, txt, color, face, dashed=False):
        a.add_patch(Rectangle((x, ycen - BH / 2), BW, BH, facecolor=face,
                              edgecolor=color, linewidth=1.4, zorder=3,
                              linestyle=(0, (3, 2)) if dashed else "-"))
        a.text(x + BW / 2, ycen, txt, ha="center", va="center", family=MONO,
               fontsize=12, color=color, zorder=4)

    axl.text(CX[0] + BW / 2, 4.55, "축 0", ha="center", va="center",
             fontsize=10.5, color=MUTED)
    axl.text(CX[1] + BW / 2, 4.55, "축 1", ha="center", va="center",
             fontsize=10.5, color=MUTED)

    axl.text(0.2, 3.55, "col", ha="left", va="center", family=MONO,
             fontsize=11.5, color=BLUE)
    axl.text(1.75, 3.55, "(3, 1)", ha="left", va="center", family=MONO,
             fontsize=11.5, color=BLUE)
    sbox(axl, CX[0], 3.55, "3", BLUE, BLUE_F)
    sbox(axl, CX[1], 3.55, "1", BLUE, BLUE_F)

    axl.text(0.2, 2.45, "row", ha="left", va="center", family=MONO,
             fontsize=11.5, color=ORANGE)
    axl.text(1.75, 2.45, "(3,)", ha="left", va="center", family=MONO,
             fontsize=11.5, color=ORANGE)
    sbox(axl, CX[0], 2.45, "1", MUTED, "white", dashed=True)
    sbox(axl, CX[1], 2.45, "3", ORANGE, ORANGE_F)
    axl.text(CX[1] + BW + 0.35, 2.45, "없는 축은 앞에 1 을 채운다", ha="left",
             va="center", fontsize=10, color=MUTED)

    axl.plot([CX[0] - 0.25, CX[1] + BW + 0.25], [1.32, 1.32], color=INK,
             linewidth=1.2)
    axl.text(0.2, 0.72, "결과", ha="left", va="center",
             fontsize=11.5, color=GREEN)
    axl.text(1.75, 0.72, "(3, 3)", ha="left", va="center", family=MONO,
             fontsize=11.5, color=GREEN)
    sbox(axl, CX[0], 0.72, "3", GREEN, GREEN_F)
    sbox(axl, CX[1], 0.72, "3", GREEN, GREEN_F)

    # --- 아랫줄 오른쪽: 맞지 않는 경우 ---
    axr = fig.add_subplot(gs[1, 1])
    blank_axes(axr, (0, 11.6), (0, 5.4))
    axr.set_title("맞지 않으면 오류가 난다", loc="left", fontsize=12.5,
                  color=RED, pad=6)
    axr.text(0.2, 4.55, "np.arange(5) * np.arange(3)", ha="left", va="center",
             family=MONO, fontsize=11.5, color=PURPLE)

    axr.text(0.2, 3.55, "a     (5,)", ha="left", va="center", family=MONO,
             fontsize=11.5, color=BLUE)
    sbox(axr, CX[1], 3.55, "5", BLUE, BLUE_F)
    axr.text(0.2, 2.45, "b     (3,)", ha="left", va="center", family=MONO,
             fontsize=11.5, color=ORANGE)
    sbox(axr, CX[1], 2.45, "3", ORANGE, ORANGE_F)
    axr.text(CX[1] + BW + 0.45, 3.0, "같지도 않고\n1 도 아니다", ha="left",
             va="center", fontsize=10.5, color=RED)

    axr.plot([CX[1] - 0.25, CX[1] + BW + 0.25], [1.72, 1.72], color=INK,
             linewidth=1.2)
    xc0, yc0, r = CX[1] + BW / 2, 1.02, 0.34
    axr.plot([xc0 - r, xc0 + r], [yc0 - r, yc0 + r], color=RED, linewidth=2.6)
    axr.plot([xc0 - r, xc0 + r], [yc0 + r, yc0 - r], color=RED, linewidth=2.6)
    axr.text(CX[1] + BW + 0.45, 1.02, "ValueError", ha="left", va="center",
             family=MONO, fontsize=11.5, color=RED)
    axr.text(0.2, 0.28, "5 x 3 을 얻으려면 축을 직접 끼워 넣는다:  a[:, None] * b",
             ha="left", va="center", fontsize=10.5, color=INK)

    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# === 그림 3. groupby 의 분할·적용·결합 ===
def fig_split_apply_combine(path=OUT + "groupby_split_apply_combine.png"):
    """같은 분할과 같은 집계에서 agg 와 transform 이 갈라지는 자리를 그린다."""
    fig, ax = plt.subplots(figsize=(14.2, 6.6))
    blank_axes(ax, (0, 28.6), (0, 13.2))

    A_F, B_F = BLUE_F, ORANGE_F
    RH = 0.82

    # (1) 원래 자료 — d = pd.DataFrame({"g": list("aabbb"), "x": [1.,3.,2.,8.,5.]})
    rows = [["a", "1.0"], ["a", "3.0"], ["b", "2.0"], ["b", "8.0"], ["b", "5.0"]]
    faces = [A_F, A_F, B_F, B_F, B_F]
    w0, h0 = draw_table(ax, 0.6, 9.15, ["g", "x"], rows, [1.35, 1.65],
                        rowh=RH, row_faces=faces, title="원래 자료 — 5행")
    x_right = 0.6 + w0

    # (2) 분할
    jx = 4.9
    ax.plot([x_right + 0.3, jx], [6.4, 6.4], color=INK, linewidth=1.6, zorder=5)
    arrow(ax, (jx, 6.4), (8.5, 8.9), color=BLUE)
    arrow(ax, (jx, 6.4), (8.5, 4.1), color=ORANGE)
    ax.text(4.6, 7.15, "분할", ha="center", va="bottom", fontsize=12, color=INK)

    wa, ha_ = draw_table(ax, 8.9, 10.35, ["g", "x"], rows[:2], [1.35, 1.65],
                         rowh=RH, row_faces=faces[:2],
                         title="그룹  g = a", title_color=BLUE, title_fs=10.5)
    wb, hb = draw_table(ax, 8.9, 5.35, ["g", "x"], rows[2:], [1.35, 1.65],
                        rowh=RH, row_faces=faces[2:],
                        title="그룹  g = b", title_color=ORANGE, title_fs=10.5)

    # (3) 적용 — d.groupby("g")["x"].mean() 이 준 값
    arrow(ax, (8.9 + wa + 0.35, 8.95), (14.4, 8.95), color=BLUE)
    arrow(ax, (8.9 + wb + 0.35, 3.95), (14.4, 3.95), color=ORANGE)
    ax.text(13.1, 9.25, "적용", ha="center", va="bottom", fontsize=12, color=INK)
    ax.text(13.1, 4.25, "적용", ha="center", va="bottom", fontsize=12, color=INK)
    ax.text(13.1, 8.35, "mean", ha="center", va="top", fontsize=10.5,
            color=MUTED, family=MONO)
    ax.text(13.1, 3.35, "mean", ha="center", va="top", fontsize=10.5,
            color=MUTED, family=MONO)

    for ycen, txt, col, face in [(8.95, "2.0", BLUE, A_F),
                                 (3.95, "5.0", ORANGE, B_F)]:
        ax.add_patch(FancyBboxPatch((14.7, ycen - 0.48), 1.8, 0.96,
                                    boxstyle="round,pad=0,rounding_size=0.16",
                                    facecolor=face, edgecolor=col,
                                    linewidth=1.6, zorder=4))
        ax.text(15.6, ycen, txt, ha="center", va="center", family=MONO,
                fontsize=12.5, color=INK, zorder=5)

    # (4) 결합 — 두 갈래
    kx = 18.1
    ax.plot([16.6, kx], [8.95, 8.95], color=BLUE, linewidth=1.6, zorder=5)
    ax.plot([16.6, kx], [3.95, 3.95], color=ORANGE, linewidth=1.6, zorder=5)
    ax.plot([kx, kx], [3.95, 8.95], color=MUTED, linewidth=1.4, zorder=5)
    arrow(ax, (kx, 7.6), (20.1, 11.1), color=INK)
    arrow(ax, (kx, 5.3), (20.1, 4.2), color=INK)
    ax.text(18.45, 9.9, "결합", ha="left", va="center", fontsize=12, color=INK)
    ax.text(18.45, 2.9, "결합", ha="left", va="center", fontsize=12, color=INK)

    draw_table(ax, 20.5, 12.4, ["g", "x"], [["a", "2.0"], ["b", "5.0"]],
               [1.35, 1.65], rowh=RH, row_faces=[A_F, B_F],
               title="agg — 그룹당 한 행 (5행 -> 2행)", title_fs=11.5)
    ax.text(20.5, 12.4 - 3 * RH - 0.35,
            'd.groupby("g")["x"].mean()', ha="left", va="top",
            fontsize=10.5, color=PURPLE, family=MONO)

    trows = [["a", "1.0", "2.0"], ["a", "3.0", "2.0"], ["b", "2.0", "5.0"],
             ["b", "8.0", "5.0"], ["b", "5.0", "5.0"]]
    draw_table(ax, 20.5, 6.6, ["g", "x", "그룹평균"], trows,
               [1.35, 1.65, 2.5], rowh=RH, row_faces=faces,
               title="transform — 원래 행 수 유지 (5행 -> 5행)", title_fs=11.5)
    ax.text(20.5, 6.6 - 6 * RH - 0.35,
            'd.groupby("g")["x"].transform("mean")', ha="left", va="top",
            fontsize=10.5, color=PURPLE, family=MONO)

    ax.text(0.2, 0.55, "분할과 적용은 똑같다. 갈라지는 곳은 결합뿐이다 — "
                       "줄여서 돌려줄 것인가, 원래 행에 되돌려 붙일 것인가.",
            ha="left", va="center", fontsize=12, color=INK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# === 시연 / 진입점 ===
if __name__ == "__main__":
    fig_name_vs_object()
    fig_broadcasting()
    fig_split_apply_combine()
