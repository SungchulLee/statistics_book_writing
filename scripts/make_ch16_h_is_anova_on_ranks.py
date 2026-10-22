r"""16.5 Kruskal-Wallis 쪽의 상자에 들어가는 그림을 만든다.

크러스컬과 월리스가 1987년에 스스로 적은 한 줄 — H 는 「순위에 보통의 분산분석을
적용한 통계량의 단순한 함수」 — 가 글자 그대로 맞음을 보인다.

만드는 파일:
  ch16/multi_group_nonparametric/img/h_is_anova_on_ranks.png

실행:  python3 scripts/make_ch16_h_is_anova_on_ranks.py   (저장소 최상위에서)
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
BLUE = "#1565C0"
ORANGE = "#E65100"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch16/multi_group_nonparametric/img/"
os.makedirs(OUT, exist_ok=True)


def h_and_f(groups):
    """같은 자료에서 (1) KW 의 H, (2) 순위에 그냥 돌린 일원분산분석의 F 를 돌려준다."""
    allv = np.concatenate(groups)
    N = allv.size
    k = len(groups)
    r = stats.rankdata(allv)
    parts, pos = [], 0
    for g in groups:
        parts.append(r[pos:pos + g.size])
        pos += g.size

    # (1) 교과서 공식
    H = 12.0 / (N * (N + 1)) * sum(p.sum() ** 2 / p.size for p in parts) - 3 * (N + 1)

    # (2) 순위를 그대로 분산분석에 넣는다
    gm = r.mean()
    SSB = sum(p.size * (p.mean() - gm) ** 2 for p in parts)
    SSW = sum(((p - p.mean()) ** 2).sum() for p in parts)
    SST = SSB + SSW
    F = (SSB / (k - 1)) / (SSW / (N - k))
    return H, F, (N - 1) * SSB / SST, N, k


# ------------------------------------------------- 항등식을 수로 확인한다
# 왼쪽 그림에서 점이 곡선에 **정확히** 얹히는 것을 보이려면 (k, N) 을 고정해야 한다.
DESIGNS = [(3, 6, RED, "$k=3$, 집단마다 $6$ 개 ($N=18$)"),
           (5, 8, BLUE, "$k=5$, 집단마다 $8$ 개 ($N=40$)")]

rng = np.random.default_rng(1952)
panel, worst = [], 0.0
for k, n, color, label in DESIGNS:
    Hs, Fs = [], []
    for _ in range(400):
        shift = rng.normal(0, 0.6, size=k)
        groups = [rng.normal(s, 1.0, size=n) for s in shift]
        H, F, H_from_ss, N, _ = h_and_f(groups)
        worst = max(worst, abs(H - H_from_ss))
        Hs.append(H)
        Fs.append(F)
    panel.append((np.array(Hs), np.array(Fs), k, k * n, color, label))

# p 값 비교는 설계를 섞어 모은다
rows = []
for _ in range(900):
    k = int(rng.integers(3, 7))
    ns = rng.integers(4, 13, size=k)
    shift = rng.normal(0, 0.55, size=k)
    groups = [rng.normal(s, 1.0, size=int(x)) for s, x in zip(shift, ns)]
    H, F, H_from_ss, N, k = h_and_f(groups)
    worst = max(worst, abs(H - H_from_ss))
    rows.append((stats.chi2.sf(H, k - 1), stats.f.sf(F, k - 1, N - k)))

pchi_v, pF_v = (np.array(c) for c in zip(*rows))
disagree = ((pchi_v < 0.05) != (pF_v < 0.05)).mean()
print(f"H 와 (N-1)SSB/SST 의 최대 어긋남 = {worst:.3e}")
print(f"0.05 를 두고 결론이 갈리는 비율 = {disagree:.4f}  (자료 {len(rows)} 벌)")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.4))

# ------------------------------------------------- 왼쪽: H 와 F 는 한 몸이다
for Hs, Fs, k, N, color, label in panel:
    hh = np.linspace(0.001, N - 1.2, 400)
    ff = (hh / (k - 1)) / ((N - 1 - hh) / (N - k))
    axL.plot(ff, hh, color=color, lw=3.4, alpha=0.30, zorder=2)
    axL.scatter(Fs, Hs, s=13, color=color, alpha=0.8, zorder=3,
                edgecolors="none", label=label)
axL.set_xlim(0, 10)
axL.set_ylim(0, 20)
axL.set_xlabel("순위를 그대로 넣은 일원분산분석의 $F$")
axL.set_ylabel("Kruskal-Wallis $H$")
axL.set_title("한 자료에서 잰 두 수 — 굵은 선이 공식이 말하는 자리",
              color=INK, fontsize=12, pad=9)
axL.legend(loc="lower right", fontsize=9, framealpha=0.95)
axL.text(0.33, 18.4, f"자료 $800$ 벌 모두 어긋남이 $10^{{-13}}$ 아래",
         fontsize=9, color=INK)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ------------------------------------------------- 오른쪽: 두 p 값
lo = 1e-6
axR.plot([lo, 1], [lo, 1], color=MUTED, lw=1.0, ls="--", zorder=1)
axR.scatter(np.clip(pF_v, lo, 1), np.clip(pchi_v, lo, 1), s=16, alpha=0.6,
            color=BLUE, edgecolors="none", zorder=3)
axR.set_xscale("log")
axR.set_yscale("log")
axR.set_xlim(lo, 1.4)
axR.set_ylim(lo, 1.4)
axR.axhline(0.05, color=GREEN, lw=0.9, ls=":", zorder=2)
axR.axvline(0.05, color=GREEN, lw=0.9, ls=":", zorder=2)
# 로그 눈금의 마이너스는 mathtext 라 unicode_minus 로 막지 못한다 — 평문으로 박는다
ticks = [1e-6, 1e-4, 1e-2, 1]
labels = ["0.000001", "0.0001", "0.01", "1"]
axR.set_xticks(ticks)
axR.set_xticklabels(labels, fontsize=9)
axR.set_yticks(ticks)
axR.set_yticklabels(labels, fontsize=9)
axR.minorticks_off()
axR.set_xlabel("$F_{k-1,\\,N-k}$ 로 읽은 $p$")
axR.set_ylabel("$\\chi^2_{k-1}$ 로 읽은 $p$")
axR.set_title(f"읽는 표만 다르다 — $0.05$ 에서 결론이 갈린 것은 "
              f"$900$ 벌 가운데 {int(round(disagree * 900))} 벌",
              color=INK, fontsize=12, pad=9)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "h_is_anova_on_ranks.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
