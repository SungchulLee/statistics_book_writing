r"""11장 진단 절의 '가정 위반의 처리' 쪽 개념 그림을 생성한다.

만드는 파일:
  ch11/diagnostics/img/remedy_matrix.png   위반의 종류와 처방이 맞아야 고쳐진다

실행:  python3 scripts/make_ch11_diagnostics_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch11/diagnostics/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def welch_p(groups):
    n = np.array([len(g) for g in groups], float)
    m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    k = len(n)
    w = n / v
    W = w.sum()
    m_t = (w * m).sum() / W
    tmp = np.sum((1 - w / W) ** 2 / (n - 1))
    F = ((w * (m - m_t) ** 2).sum() / (k - 1)) / (
        1 + 2 * (k - 2) / (k * k - 1) * tmp)
    return stats.f.sf(F, k - 1, (k * k - 1) / (3 * tmp))


def perm_p(groups, rng, B=199):
    """집단 표시를 뒤섞어 얻는 F 검정의 순열 p-값. SSB 가 F 의 단조함수다."""
    sizes = np.array([len(g) for g in groups])
    x = np.concatenate(groups)
    N = len(x)
    cuts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    tot = x.sum()

    def ssb(v):
        s = np.add.reduceat(v, cuts, axis=-1)
        return (s ** 2 / sizes).sum(axis=-1) - tot ** 2 / N

    obs = ssb(x)
    P = rng.permuted(np.tile(x, (B, 1)), axis=1)
    null = ssb(P)
    return (1 + np.sum(null >= obs)) / (B + 1)


def fig_remedy_matrix():
    rng = np.random.default_rng(1111)
    M = 2_000

    scen = [
        ("가정 모두 성립\n정규, $\\sigma$ = (1, 1, 1), $n$ = (15, 15, 15)",
         lambda: [20 + rng.standard_normal(15) for _ in range(3)]),
        ("정규성 위반\n대수정규, $n$ = (15, 15, 15)",
         lambda: [rng.lognormal(0, 1, 15) for _ in range(3)]),
        ("등분산 위반\n$\\sigma$ = (1, 1, 3), $n$ = (15, 15, 15)",
         lambda: [20 + rng.normal(0, s, 15) for s in (1, 1, 3)]),
        ("등분산 위반 + 불균형\n$\\sigma$ = (1, 1, 3), $n$ = (25, 25, 8)",
         lambda: [20 + rng.normal(0, s, n)
                  for s, n in zip((1, 1, 3), (25, 25, 8))]),
    ]
    methods = ["고전 $F$", "로그 변환 후 $F$", "크러스컬–월리스",
               "순열검정", "웰치 분산분석"]
    tab = np.zeros((len(scen), len(methods)))

    for i, (lb, gen) in enumerate(scen):
        hit = np.zeros(len(methods))
        for _ in range(M):
            g = gen()
            hit[0] += stats.f_oneway(*g).pvalue < 0.05
            lg = [np.log(x) for x in g]
            hit[1] += stats.f_oneway(*lg).pvalue < 0.05
            hit[2] += stats.kruskal(*g).pvalue < 0.05
            hit[3] += perm_p(g, rng) <= 0.05
            hit[4] += welch_p(g) < 0.05
        tab[i] = hit / M
        print(f"  {lb.splitlines()[0]:16s} " +
              "  ".join(f"{m}={v:.3f}" for m, v in
                        zip(["F", "logF", "KW", "perm", "Welch"], tab[i])))

    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    im = ax.imshow(tab, cmap="RdYlGn_r", vmin=0.0, vmax=0.22, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ok = abs(tab[i, j] - 0.05) <= 0.010
            ax.text(j, i, f"{tab[i, j]:.3f}", ha="center", va="center",
                    fontsize=12.5,
                    color="white" if tab[i, j] > 0.17 else INK,
                    fontweight="bold" if ok else "normal")
            if ok:
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.42), 0.94, 0.84,
                                           fill=False, edgecolor=INK, lw=2.2))
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10.5)
    ax.set_yticks(range(len(scen)))
    ax.set_yticklabels([s[0] for s in scen], fontsize=9.5)
    ax.tick_params(length=0, colors=INK)
    ax.set_title("평균이 모두 같을 때의 실제 기각률 — 굵은 테두리가 명목 0.05 를 지킨 칸",
                 fontsize=12, color=INK, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("실제 제1종 오류율", fontsize=10, color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK)
    save(fig, "remedy_matrix.png")


if __name__ == "__main__":
    fig_remedy_matrix()
