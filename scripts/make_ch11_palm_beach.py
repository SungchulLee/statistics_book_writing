r"""11.6절 「그림으로 보는 통계학」 — 스튜던트화 잔차 24.

$2000$ 년 미국 대통령 선거에서 플로리다의 $67$ 개 군이 낸 표를 군마다 하나의 점으로
놓는다. 가로축은 부시가 받은 표, 세로축은 **팻 뷰캐넌**이 받은 표다. 팜비치 군만
혼자 위로 솟아 있다. 그 군은 후보 이름이 좌우 양쪽에 번갈아 적힌 「나비 투표지」를
썼다.

이 스크립트가 재는 값:

  원 눈금  Buchanan ~ Bush ($67$ 군)
    팜비치 — 스튜던트화 잔차 24.08, 지렛값 0.0709, 쿡 거리 2.23
    나머지 $66$ 군의 최대 — |t| 3.28, 쿡 거리 1.981
    기울기 0.004917 -> (팜비치를 빼면) 0.003482   곧 29% 가 그 한 점이었다
    팜비치를 뺀 적합이 예측하는 팜비치 = 598표 (실제 3407표)

  로그-로그  log(Buchanan) ~ log(Bush), 팜비치 제외 적합 $R^2 = 0.8658$
    팜비치 예측 592표, 95% 예측구간 251 ~ 1399표
    실제 3407표 — 구간 위쪽 끝의 2.4배, 초과 2815표

부시가 플로리다에서 이긴 표차는 $537$ 표였다.

자료는 R 의 `Stat2Data::PalmBeach`($67$ 군의 뷰캐넌 표와 부시 표)를 그대로 적어
넣었다. 그 꾸러미가 밝힌 출처는 플로리다 주 선거국의 결과 보관소다.

만드는 파일:

  ch11/diagnostics/img/palm_beach.png

실행:  python3 scripts/make_ch13_palm_beach.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/diagnostics/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

# 2000 년 플로리다 67 개 군 — 뷰캐넌 표와 부시 표 (Stat2Data::PalmBeach)
BUCHANAN = np.array([262, 73, 248, 65, 570, 789, 90, 182, 270, 186,
                     122, 89, 561, 36, 29, 650, 504, 83, 33, 39, 29,
                     9, 71, 24, 30, 22, 242, 99, 836, 76, 105, 102,
                     29, 10, 289, 305, 282, 67, 39, 29, 272, 563,
                     108, 47, 90, 267, 43, 446, 145, 3407, 570, 1010,
                     538, 147, 229, 124, 311, 305, 194, 114, 108, 27,
                     26, 396, 46, 120, 88])
BUSH = np.array([34062, 5610, 38637, 5413, 115185, 177279, 2873,
                 35419, 29744, 41745, 60426, 10964, 289456, 4256,
                 2698, 152082, 73029, 12608, 2448, 4750, 3300, 1840,
                 3546, 2153, 3764, 4743, 30646, 20196, 176967, 4985,
                 28627, 9138, 2481, 1669, 49963, 106141, 39053, 6860,
                 1316, 3038, 57948, 55135, 33864, 16059, 16404,
                 52043, 5058, 134476, 26216, 152846, 68581, 184312,
                 90101, 13439, 39497, 34705, 36248, 83100, 75293,
                 12126, 8014, 4051, 2326, 82214, 4511, 12176, 4983])
PB = 49            # 팜비치 군
MARGIN = 537       # 부시의 플로리다 표차


def diag(x, y):
    """계수, 잔차, 지렛값, 외부 스튜던트화 잔차, 쿡 거리."""
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    h = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    n, p = X.shape
    s2 = (r ** 2).sum() / (n - p)
    s2_i = ((r ** 2).sum() - r ** 2 / (1 - h)) / (n - p - 1)   # i 를 뺀 분산
    t = r / np.sqrt(s2_i * (1 - h))
    D = r ** 2 * h / (p * s2 * (1 - h) ** 2)
    return b, r, h, t, D


def fit_wo(x, y, drop):
    m = np.arange(len(x)) != drop
    X = np.column_stack([np.ones(m.sum()), x[m]])
    b = np.linalg.lstsq(X, y[m], rcond=None)[0]
    res = y[m] - X @ b
    s = np.sqrt((res ** 2).sum() / (m.sum() - 2))
    r2 = 1 - (res ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    return b, s, X, m, float(r2)


def main():
    keep = np.arange(len(BUSH)) != PB

    # ── 원 눈금 ──
    b, r, h, t, D = diag(BUSH, BUCHANAN)
    b_wo, _, _, _, _ = fit_wo(BUSH, BUCHANAN, PB)
    pred_raw = b_wo[0] + b_wo[1] * BUSH[PB]
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print("원 눈금  Buchanan ~ Bush (67 군)")
    print(f"  팜비치 — 스튜던트화 잔차 {t[PB]:.2f}  지렛값 {h[PB]:.4f}  "
          f"쿡 거리 {D[PB]:.2f}")
    print(f"  나머지 66 군의 최대 — |t| {np.abs(t[keep]).max():.2f}  "
          f"쿡 거리 {D[keep].max():.3f}")
    print(f"  기울기 {b[1]:.6f} -> 팜비치를 빼면 {b_wo[1]:.6f}  "
          f"({100 * (b[1] - b_wo[1]) / b[1]:.0f}% 가 그 한 점이었다)")
    print(f"  팜비치를 뺀 적합의 팜비치 예측 = {pred_raw:.0f} 표 "
          f"(실제 {BUCHANAN[PB]} 표)")

    # ── 로그-로그 ──
    lx, ly = np.log(BUSH), np.log(BUCHANAN)
    lb, s, X, m, r2 = fit_wo(lx, ly, PB)
    XtXi = np.linalg.inv(X.T @ X)
    tc = stats.t.ppf(0.975, m.sum() - 2)

    def band(xx):
        """팜비치를 뺀 적합의 95% 예측구간 (로그 눈금에서 계산해 되돌린다)."""
        c = lb[0] + lb[1] * xx
        v = s * np.sqrt(1 + np.array([[1, u] @ XtXi @ [1, u] for u in xx]))
        return np.exp(c), np.exp(c - tc * v), np.exp(c + tc * v)

    fit_pb, lo_pb, hi_pb = (float(z[0]) for z in band(np.array([lx[PB]])))
    print(f"\n로그-로그  팜비치 제외 적합 R^2 {r2:.4f}  기울기 {lb[1]:.4f}")
    print(f"  팜비치 예측 {fit_pb:.0f} 표  95% 예측구간 {lo_pb:.0f} ~ {hi_pb:.0f} 표")
    print(f"  실제 {BUCHANAN[PB]} 표 — 구간 위쪽 끝의 "
          f"{BUCHANAN[PB] / hi_pb:.1f} 배, 초과 {BUCHANAN[PB] - fit_pb:.0f} 표")
    print(f"  (부시의 플로리다 표차는 {MARGIN} 표였다)")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)

    # ── ① 원 눈금 — 한 점이 기울기를 끌어올린다 ──
    ax = axes[0]
    g = np.linspace(0, 300000, 200)
    ax.plot(BUSH[keep], BUCHANAN[keep], "o", ms=7, color=MUTED, alpha=0.85,
            label="플로리다의 다른 $66$ 개 군")
    ax.plot(BUSH[PB], BUCHANAN[PB], "X", ms=18, color=RED, mec="white",
            mew=1.6, zorder=6, label="팜비치 군")
    ax.plot(g, b[0] + b[1] * g, lw=2.4, color=AMBER,
            label=f"$67$ 군 전부 — 기울기 ${b[1]:.5f}$")
    ax.plot(g, b_wo[0] + b_wo[1] * g, "--", lw=2.4, color=TEAL,
            label=f"팜비치를 빼고 — 기울기 ${b_wo[1]:.5f}$")
    ax.plot([BUSH[PB]], [pred_raw], "o", ms=10, color=TEAL, mec="white",
            mew=1.3, zorder=6)
    ax.annotate(f"뺀 적합이 말하는 팜비치\n${pred_raw:.0f}$ 표",
                xy=(BUSH[PB], pred_raw), xytext=(188000, 1250),
                fontsize=11, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.annotate(f"실제 ${BUCHANAN[PB]}$ 표\n스튜던트화 잔차 ${t[PB]:.1f}$",
                xy=(BUSH[PB], BUCHANAN[PB]), xytext=(55000, 2850),
                fontsize=11.4, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.set_xlim(-8000, 305000)
    ax.set_ylim(-150, 3800)
    ax.set_xlabel("부시가 받은 표", fontsize=11.5)
    ax.set_ylabel("뷰캐넌이 받은 표", fontsize=11.5)
    ax.set_title("① 점 하나가 기울기의 $29\\%$ 였다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9.8, framealpha=0.94, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 로그-로그 — 예측구간 밖 ──
    ax = axes[1]
    gl = np.linspace(np.log(1100), np.log(320000), 200)
    f, lo, hi = band(gl)
    ax.fill_between(np.exp(gl), lo, hi, color=TEAL, alpha=0.16,
                    label="팜비치를 뺀 적합의 $95\\%$ 예측구간")
    ax.plot(np.exp(gl), f, lw=2.4, color=TEAL)
    ax.plot(BUSH[keep], BUCHANAN[keep], "o", ms=7, color=MUTED, alpha=0.9,
            label="다른 $66$ 개 군")
    ax.plot(BUSH[PB], BUCHANAN[PB], "X", ms=18, color=RED, mec="white",
            mew=1.6, zorder=6, label="팜비치 군")
    ax.annotate(f"구간은 ${lo_pb:.0f}$–${hi_pb:.0f}$ 표인데\n"
                f"실제는 ${BUCHANAN[PB]}$ 표다",
                xy=(BUSH[PB], BUCHANAN[PB]), xytext=(17000, 13),
                fontsize=11.4, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1100, 340000)
    ax.set_ylim(5, 6000)
    ax.set_xlabel("부시가 받은 표 (로그 눈금)", fontsize=11.5)
    ax.set_ylabel("뷰캐넌이 받은 표 (로그 눈금)", fontsize=11.5)
    ax.set_title(f"② 초과 ${BUCHANAN[PB] - fit_pb:.0f}$ 표 — 표차는 "
                 f"${MARGIN}$ 표였다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=9.8, framealpha=0.94, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("뺄 것인가 말 것인가 — 스튜던트화 잔차 $24$", fontsize=14.5)
    fig.text(0.5, -0.04,
             "$2000$ 년 플로리다 $67$ 개 군의 득표다"
             "(R 의 Stat2Data 꾸러미가 싣고 있는 그 표). "
             "기울기·잔차·예측구간은 모두 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "palm_beach.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
