r"""6.2절 포획-재포획 — 가능도를 눈으로 보는 그림 셋.

  cr_likelihood.png     달아 둔 보기(M=50, n=40, m=10)의 가능도를 점으로 찍고,
                        그 옆에 이웃한 값의 비 L(N)/L(N-1) 을 놓는다. 비가 1 을
                        건너는 자리가 곧 봉우리이고, 그 자리가 Mn/m = 200 이다.
  cr_flatness.png       같은 봉우리를 상대가능도로 다시 그린다. 가능도구간과
                        왈드구간을 겹쳐 놓아 대칭 구간이 왜 못 쓰는지를 보인다.
  cr_m_sharpens.png     Mn/m = 200 을 고정한 채 m 만 2, 5, 10, 20 으로 올린다.
                        봉우리를 뾰족하게 만드는 것이 n 이 아니라 m 임을 보인다.

가능도는 초기하분포다.

    L(N) = C(M, m) C(N-M, n-m) / C(N, n)

정수 모수라 미분할 수 없으므로 곡선이 아니라 점으로 찍는다. 그것이 이 그림의
요점이기도 하다.

만드는 파일:

  ch06/mle/img/cr_likelihood.png
  ch06/mle/img/cr_flatness.png
  ch06/mle/img/cr_m_sharpens.png

실행:  python3 scripts/make_ch06_capture_recapture.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
난수를 쓰지 않는다. 모든 수는 조합으로 정확히 계산한다.
"""

from math import comb

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["NanumGothic", "Apple SD Gothic Neo", "Malgun Gothic"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch06/mle/img/"

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
MUTED = "#90A4AE"
RED = "#D32F2F"

# 본문이 내내 끌고 가는 보기: 50 마리 표지, 40 마리 재포획, 그중 10 마리가 표지.
M, N_CAP, M_REC = 50, 40, 10


# === 가능도 ================================================================

def like(N, M=M, n=N_CAP, m=M_REC):
    """초기하 가능도. 관측이 불가능한 N 에서는 0 을 준다."""
    if N < M + n - m:
        return 0.0
    return comb(M, m) * comb(N - M, n - m) / comb(N, n)


def grid(M=M, n=N_CAP, m=M_REC, upper=600):
    """관측이 가능한 가장 작은 N 부터 upper 까지."""
    lo = M + n - m
    Ns = np.arange(lo, upper + 1)
    return Ns, np.array([like(int(N), M, n, m) for N in Ns])


# === 그림 1. 가능도와 이웃한 값의 비 ========================================

def fig_likelihood():
    Ns, L = grid(upper=600)
    hat = M * N_CAP // M_REC                      # 200
    peak = L.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.6))

    # --- 왼쪽: 가능도 자체 ---
    ax1.vlines(Ns, 0, L, color=BLUE_F, lw=0.9)
    ax1.plot(Ns, L, "o", ms=2.6, color=BLUE, label="$L(N)$")
    i = int(np.where(Ns == hat)[0][0])
    ax1.plot([hat], [L[i]], "o", ms=8, color=RED, zorder=5)
    ax1.annotate(f"$\\hat N = Mn/m = {hat}$", xy=(hat, L[i]),
                 xytext=(hat + 95, L[i] * 0.97), fontsize=11.5, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    # 왼쪽은 가파르고 오른쪽은 길게 끌린다 — 그 비대칭을 글자로 짚어 둔다.
    ax1.annotate("가파르게 오른다", xy=(112, peak * 0.82), fontsize=10.5,
                 color=MUTED, ha="left")
    ax1.annotate("길게 끌린다", xy=(430, peak * 0.42), fontsize=10.5,
                 color=MUTED, ha="center")

    ax1.set_xlabel("개체수 $N$", fontsize=11.5)
    ax1.set_ylabel("가능도 $L(N)$", fontsize=11.5)
    ax1.set_title("가능도는 점으로만 뜻이 있다", fontsize=13, color=INK, pad=10)
    ax1.set_xlim(70, 610)
    ax1.set_ylim(0, peak * 1.18)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=10.5, loc="upper right", frameon=False)

    # --- 오른쪽: 이웃한 값의 비 ---
    ratio = L[1:] / L[:-1]
    Nr = Ns[1:]
    up = ratio >= 1.0
    ax2.axhline(1.0, color=INK, lw=1.2, ls="--")
    ax2.plot(Nr[up], ratio[up], "o", ms=3.2, color=GREEN,
             label="$L(N)/L(N-1) \\geq 1$ — 아직 오른다")
    ax2.plot(Nr[~up], ratio[~up], "o", ms=3.2, color=ORANGE,
             label="$< 1$ — 이미 내린다")
    ax2.axvline(hat, color=RED, lw=1.1, ls=":")
    ax2.annotate(f"색이 바뀌는 자리 = {hat}", xy=(hat, 1.0),
                 xytext=(hat + 70, 1.06), fontsize=11, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    ax2.set_xlabel("개체수 $N$", fontsize=11.5)
    ax2.set_ylabel("이웃한 가능도의 비", fontsize=11.5)
    ax2.set_title("비가 1 을 건너는 자리가 봉우리다", fontsize=13, color=INK, pad=10)
    ax2.set_xlim(70, 610)
    ax2.set_ylim(0.95, 1.15)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=10, loc="upper right", frameon=False)

    fig.suptitle(f"$M = {M}$ 마리를 표지하고 $n = {N_CAP}$ 마리를 다시 잡아 "
                 f"$m = {M_REC}$ 마리가 표지였다", fontsize=14.5, y=1.02)
    fig.tight_layout()
    path = OUT + "cr_likelihood.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)

    # 본문이 인용하는 수를 함께 찍어 둔다.
    for N in (120, 150, 200, 300, 400, 600):
        print(f"  L({N})/L(200) = {like(N) / like(200):.3f}")


# === 그림 2. 봉우리의 평평함과 두 구간 ======================================

def fig_flatness():
    Ns, L = grid(upper=900)
    hat = M * N_CAP // M_REC
    R = L / L.max()                                # 상대가능도

    # 가능도구간: R(N) >= 1/8 인 N 들 (표준적인 기준 하나)
    inside = Ns[R >= 1 / 8]
    lo_lik, hi_lik = int(inside.min()), int(inside.max())

    # 왈드구간: delta 방법의 Var = M^2 n (n-m) / m^3
    var = M ** 2 * N_CAP * (N_CAP - M_REC) / M_REC ** 3
    se = np.sqrt(var)
    lo_wald, hi_wald = hat - 1.96 * se, hat + 1.96 * se
    floor = M + N_CAP - M_REC                      # 논리적 하한 80

    fig, ax = plt.subplots(figsize=(13.4, 4.8))

    ax.vlines(Ns, 0, R, color=BLUE_F, lw=0.8)
    ax.plot(Ns, R, "o", ms=2.4, color=BLUE)
    ax.axhline(1 / 8, color=GREEN, lw=1.3, ls="--")
    ax.text(880, 1 / 8 + 0.018, "상대가능도 $1/8$", fontsize=10.5,
            color=GREEN, ha="right")

    # 두 구간을 아래쪽에 막대로 겹쳐 놓는다.
    y_lik, y_wald = -0.085, -0.155
    ax.plot([lo_lik, hi_lik], [y_lik, y_lik], lw=5, color=GREEN,
            solid_capstyle="butt")
    ax.plot([lo_wald, hi_wald], [y_wald, y_wald], lw=5, color=ORANGE,
            solid_capstyle="butt")
    ax.plot([hat], [y_lik], "|", ms=14, color=INK, mew=1.6)
    ax.plot([hat], [y_wald], "|", ms=14, color=INK, mew=1.6)
    ax.text(hi_lik + 14, y_lik, f"가능도구간  ({lo_lik}, {hi_lik})",
            fontsize=10.5, color=GREEN, va="center")
    ax.text(hi_wald + 14, y_wald,
            f"왈드구간  ({lo_wald:.0f}, {hi_wald:.0f})",
            fontsize=10.5, color=ORANGE, va="center")

    # 논리적으로 불가능한 영역 — 띠가 좁으므로 글자를 세워 안에 넣는다.
    ax.axvspan(0, floor, color="#FFCDD2", alpha=0.55, lw=0)
    ax.text((40 + floor) / 2, 0.55,
            f"$N < M + n - m = {floor}$ — 관측과 모순",
            fontsize=10, color=RED, ha="center", va="center", rotation=90)

    ax.axvline(hat, color=RED, lw=1.1, ls=":")
    ax.text(hat + 8, 1.02, f"$\\hat N = {hat}$", fontsize=11.5, color=RED)

    ax.set_xlabel("개체수 $N$", fontsize=11.5)
    ax.set_ylabel("상대가능도 $L(N)/L(\\hat N)$", fontsize=11.5)
    ax.set_title("봉우리가 평평하면 대칭 구간은 쓸 수 없다",
                 fontsize=14, color=INK, pad=12)
    ax.set_xlim(40, 900)
    ax.set_ylim(-0.21, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = OUT + "cr_flatness.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    print(f"  가능도구간 = ({lo_lik}, {hi_lik}),  "
          f"왈드구간 = ({lo_wald:.1f}, {hi_wald:.1f}),  SE = {se:.1f}")
    print(f"  왈드 하한이 논리적 하한 {floor} 보다 "
          f"{'작다' if lo_wald < floor else '크다'}")


# === 그림 3. 봉우리를 뾰족하게 만드는 것은 m 이다 ===========================

def fig_m_sharpens():
    # Mn/m = 200 을 정확히 고정한 채 m 만 키운다.
    settings = [(20, 20, 2), (50, 20, 5), (50, 40, 10), (100, 40, 20)]
    colors = ["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"]

    fig, ax = plt.subplots(figsize=(13.4, 4.8))

    # m = 2 의 구간은 1200 을 넘으므로 네 곡선이 모두 닫히도록 넉넉히 잡는다.
    UPPER = 1300

    for (Mi, ni, mi), c in zip(settings, colors):
        Ns, L = grid(Mi, ni, mi, upper=UPPER)
        R = L / L.max()
        ax.plot(Ns, R, lw=1.9, color=c,
                label=f"$m = {mi}$   ($M = {Mi}$, $n = {ni}$)")
        inside = Ns[R >= 1 / 8]
        lo, hi = int(inside.min()), int(inside.max())
        print(f"  m = {mi:2d}:  가능도구간 = ({lo}, {hi}),  폭 = {hi - lo}")

    ax.axhline(1 / 8, color=MUTED, lw=1.2, ls="--")
    ax.text(UPPER - 10, 1 / 8 + 0.02, "상대가능도 $1/8$", fontsize=10.5,
            color=MUTED, ha="right")
    ax.axvline(200, color=RED, lw=1.1, ls=":")
    ax.text(206, 1.02, "$\\hat N = 200$ — 네 경우 모두 같다",
            fontsize=11.5, color=RED)

    ax.set_xlabel("개체수 $N$", fontsize=11.5)
    ax.set_ylabel("상대가능도 $L(N)/L(\\hat N)$", fontsize=11.5)
    ax.set_title("추정값은 그대로인데 봉우리만 뾰족해진다 — $m$ 이 하는 일",
                 fontsize=14, color=INK, pad=12)
    ax.set_xlim(40, UPPER)
    ax.set_ylim(0, 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10.5, loc="upper right", frameon=False)

    fig.tight_layout()
    path = OUT + "cr_m_sharpens.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# === 실행 ==================================================================

if __name__ == "__main__":
    print("그림 1 — 가능도와 이웃한 값의 비")
    fig_likelihood()
    print("그림 2 — 봉우리의 평평함과 두 구간")
    fig_flatness()
    print("그림 3 — m 이 봉우리를 뾰족하게 만든다")
    fig_m_sharpens()
    print("완료")
