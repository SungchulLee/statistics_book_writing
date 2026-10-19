r"""12.1절 「그림으로 보는 통계학」 — 순위상관은 무엇을 재려고 만들어졌나.

이 쪽은 스피어만의 순위상관을 「이상점에 로버스트하고 단조변환에 불변인
도구」로 소개한다. 맞는 말인데, 그가 그것을 만든 까닭은 로버스트성이
아니었다.

1904 년에 스피어만은 『아메리칸 저널 오브 사이콜로지』 같은 권에 두 편을
실었다. 앞의 것이 순위상관을 내놓은 「두 가지 사이의 연관을 증명하고
재기」이고, 뒤의 것이 「객관적으로 정해지고 측정된 '일반지능'」이다.
순위상관은 뒤의 물음을 풀려고 만든 **연장**이었다.

그가 본 것은 이것이다. 시골 학교 아이들의 성적을 과목끼리 상관지으면
**전부 양수**가 나온다. 고전어와 음높이 변별처럼 아무 상관이 없어 보이는
짝까지. 이 「양의 다발」을 그는 공통 원인 하나로 설명하려 했다 — g.

그 설명에는 검증할 수 있는 예측이 딸려 있다. 공통요인이 하나뿐이라면

  r_ij = a_i a_j   (i != j)

이고, 그러면 어떤 넷을 골라도 **사원차**가 0 이어야 한다.

  r_ij r_kl - r_il r_kj = 0

이 그림은 그 예측을 재 본다. 널리 재수록되는 상관표에서 한 요인 모형을
맞추면 비대각 잔차가 최대 0.056, 사원차 45 개가 최대 0.11 로 아이가
서른 명일 때의 표본오차 0.14 안에 모두 들어온다. 예측이 버틴다.

버틴다고 해서 머릿속에 g 라는 것이 하나 있다는 뜻은 아니다. 서스톤이
1930–40 년대에 보였듯 같은 상관표는 여러 요인 모형으로도 똑같이 잘
맞는다. 상관의 구조가 하나라는 것과 원인이 하나라는 것은 다른 말이다 —
이 장이 내내 하는 이야기다.

만드는 파일:

  ch12/correlation/img/spearman_g.png

실행:  python3 scripts/make_ch12_spearman_g.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import itertools

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch12/correlation/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

LAB = ["고전어", "프랑스어", "영어", "수학", "음높이 변별", "음악"]

# 스피어만(1904)의 상관표 — 여러 교과서에 널리 재수록되는 형태
R = np.array([
    [1.00, 0.83, 0.78, 0.70, 0.66, 0.63],
    [0.83, 1.00, 0.67, 0.67, 0.65, 0.57],
    [0.78, 0.67, 1.00, 0.64, 0.54, 0.51],
    [0.70, 0.67, 0.64, 1.00, 0.45, 0.51],
    [0.66, 0.65, 0.54, 0.45, 1.00, 0.40],
    [0.63, 0.57, 0.51, 0.51, 0.40, 1.00]])

NKIDS = 30          # 아이 수 — 표본오차를 어림하는 데만 쓴다


def one_factor(R, iters=500):
    """비대각만 보고 r_ij = a_i a_j 를 맞춘다 (교대 최소제곱)."""
    k = len(R)
    a = np.sqrt(np.clip(np.diag(R @ R) / (k - 1), 1e-9, None))
    for _ in range(iters):
        for i in range(k):
            idx = [j for j in range(k) if j != i]
            a[i] = sum(R[i, j] * a[j] for j in idx) / sum(a[j] ** 2 for j in idx)
    return a


def tetrads(R):
    k = len(R)
    out = []
    for i, j, m, l in itertools.combinations(range(k), 4):
        out += [R[i, j] * R[m, l] - R[i, l] * R[m, j],
                R[i, m] * R[j, l] - R[i, l] * R[j, m],
                R[i, j] * R[m, l] - R[i, m] * R[j, l]]
    return np.array(out)


def main():
    k = len(LAB)
    off = ~np.eye(k, dtype=bool)
    a = one_factor(R)
    fit = np.outer(a, a)
    res = (R - fit)[off]
    tet = tetrads(R)

    iu = np.triu_indices(k, 1)
    obs_v, fit_v = R[iu], fit[iu]

    rbar = R[off].mean()
    se_r = (1 - rbar ** 2) / np.sqrt(NKIDS - 1)
    se_t = 2 * rbar * se_r

    print(f"    양의 다발 — 비대각 최솟값 {R[off].min():.2f} (모두 양수: "
          f"{bool((R[off] > 0).all())})")
    print("\n    한 요인 모형의 적재값")
    for n, v in zip(LAB, a):
        print(f"      {n:<12} a = {v:.3f}   공통요인이 설명하는 몫 {v**2*100:4.1f}%")
    print(f"\n    비대각 잔차  최대 {np.abs(res).max():.4f}   "
          f"제곱평균제곱근 {np.sqrt((res**2).mean()):.4f}")
    print(f"    사원차 {len(tet)} 개  최대 {np.abs(tet).max():.4f}   "
          f"평균 {np.abs(tet).mean():.4f}")
    print(f"    아이 {NKIDS} 명일 때 상관의 표준오차 ~ {se_r:.3f}, "
          f"사원차의 표준오차 ~ {se_t:.3f}")
    print(f"    표본오차 안에 든 사원차 {int((np.abs(tet) <= se_t).sum())}/{len(tet)}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)

    # ── 왼쪽: 양의 다발 ──
    ax = axes[0]
    M = R.copy()
    np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, cmap="YlOrBr", vmin=0.35, vmax=0.90)
    for i in range(k):
        for j in range(k):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=13,
                        color=GREY)
            else:
                ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center",
                        fontsize=11.4,
                        color="white" if R[i, j] > 0.72 else INK)
    ax.set_xticks(range(k)); ax.set_xticklabels(LAB, fontsize=10.6, rotation=30,
                                                ha="right")
    ax.set_yticks(range(k)); ax.set_yticklabels(LAB, fontsize=10.6)
    ax.set_title(f"과목 쌍 {k*(k-1)//2} 개 가운데 음수가 하나도 없다 "
                 f"— 가장 작은 값이 {R[off].min():.2f}",
                 fontsize=12.0, color=INK, pad=8)
    fig.colorbar(im, ax=ax, shrink=0.82, label="상관계수")

    # ── 오른쪽: 한 요인 모형의 예측과 견주기 ──
    ax = axes[1]
    xs = np.linspace(0.30, 0.95, 50)
    ax.fill_between(xs, xs - se_r, xs + se_r, color=TEAL, alpha=0.16,
                    label=f"아이 {NKIDS} 명일 때의 표본오차 $\\pm{se_r:.2f}$")
    ax.plot(xs, xs, lw=2.0, color=INK, ls="--")
    ax.scatter(fit_v, obs_v, s=95, color=RED, zorder=5, edgecolor="white",
               linewidth=1.2, label="과목 쌍 $15$ 개")
    worst = np.argmax(np.abs(obs_v - fit_v))
    i0, j0 = iu[0][worst], iu[1][worst]
    ax.annotate(f"가장 많이 어긋난 쌍\n{LAB[i0]}–{LAB[j0]}\n"
                f"벗어난 폭 {abs(obs_v[worst]-fit_v[worst]):.3f}",
                xy=(fit_v[worst], obs_v[worst]), xytext=(0.78, 0.44),
                fontsize=10.4, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(0.33, 0.88,
            f"사원차 ${len(tet)}$ 개\n"
            f"최대 ${np.abs(tet).max():.3f}$  (표본오차 ${se_t:.3f}$)\n"
            f"표본오차 안에 든 것 ${int((np.abs(tet)<=se_t).sum())}/{len(tet)}$",
            fontsize=10.6, color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=GREY, alpha=0.95))
    ax.set_xlabel("공통요인 하나가 내놓는 예측  $a_i a_j$", fontsize=11.5)
    ax.set_ylabel("실제로 관측된 상관  $r_{ij}$", fontsize=11.5)
    ax.set_title("예측이 버틴다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0.30, 0.95); ax.set_ylim(0.30, 0.95)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("순위상관은 지능을 재려고 만들어졌다", fontsize=14.5)
    fig.text(0.5, -0.06,
             "공통요인이 하나뿐이면 $r_{ij} = a_i a_j$ 여야 하고, 그러면 어떤 넷을 "
             "골라도 사원차 $r_{ij}r_{kl} - r_{il}r_{kj}$ 가 $0$ 이어야 한다.\n"
             "맞아떨어진다고 해서 머릿속에 그런 것이 하나 있다는 뜻은 아니다 — "
             "같은 표는 여러 요인 모형으로도 똑같이 잘 맞는다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "spearman_g.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
