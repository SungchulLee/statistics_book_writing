r"""9.4 「대응 t 검정」 쪽의 상자에 들어가는 그림을 만든다.

마지막 관측값 이월(LOCF)이 왜 「보수적」으로 보였는지, 그리고 어디서 무너지는지.

  왼쪽  참 평균변화가 0 인 자료(H0 가 참)에 탈락을 넣고 대응 t 검정을 돌린다.
        탈락의 **수는 그대로 두고 이유만 바꾼다.** 무관한 탈락(MCAR)에서는
        LOCF 도 완전사례도 0.05 를 지킨다. 나빠진 사람이 떠나는 탈락에서는
        둘 다 함께 무너진다 — 메우는 방식이 아니라 **빠지는 이유**가 정한다.
  오른쪽 그 까닭. 떠난 사람의 참 변화는 왼쪽 꼬리에 있는데 LOCF 는 그들을
        변화 0 으로 적는다. 왼쪽 꼬리가 가운데로 옮겨 가 평균이 오른쪽으로 밀린다.

만드는 파일:
  ch09/paired_sample_tests/img/locf_dropout.png

실행:  python3 scripts/make_ch09_locf.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.

주의: BLAS 스레드를 묶어 돌릴 것. (M, N) 배열이 작아 단일 스레드가 더 빠르다.
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
RED, RED_F = "#C62828", "#FFCDD2"
GREY, MUTED = "#90A4AE", "#B0BEC5"
GREEN = "#2E7D32"

OUT = "docs/ch09/paired_sample_tests/img/"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20261023)
M, N, SD_D = 200_000, 40, 2.0          # 참 평균변화 0 이므로 H0 가 참이다
KS = [0, 4, 8, 12, 16, 20]
TCRIT = {n: stats.t.ppf(0.975, n - 1) for n in range(2, N + 1)}


def reject(d, n):
    t = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(n))
    return np.abs(t) > TCRIT[n]


def trial(k, informative):
    """탈락 k 명. informative 면 나빠진 사람이 떠난다."""
    d = rng.normal(0.0, SD_D, (M, N))
    full = reject(d, N).mean()
    if k == 0:
        return full, full, full, None
    # 떠날 사람을 고른다. 두 기전의 **탈락 수가 같도록** 분위수로 자른다.
    score = (-d + rng.normal(0, SD_D * 0.5, (M, N)) if informative
             else rng.random((M, N)))
    mask = np.zeros((M, N), bool)
    np.put_along_axis(mask, np.argpartition(-score, k - 1, axis=1)[:, :k], True, 1)
    cc = reject(d[~mask].reshape(M, N - k), N - k).mean()
    locf = reject(np.where(mask, 0.0, d), N).mean()
    return full, cc, locf, (d, mask)


res = {}
keep_example = None
for informative in (False, True):
    rows = []
    for k in KS:
        full, cc, locf, ex = trial(k, informative)
        rows.append((k, full, cc, locf))
        if informative and k == 8:
            keep_example = ex
    res[informative] = rows

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.5))

# ---------------------------------------------- 왼쪽: 이유만 바꾸면
q = np.array(KS) / N
axL.axhline(0.05, color=GREY, lw=1.4, ls="--", zorder=2)
axL.text(0.30, 0.085, "명목 $0.05$", fontsize=9.5, color=INK)
for informative, color, marker, label in (
        (True, RED, "o", "나빠진 사람이 떠난다"),
        (False, BLUE, "s", "무관한 탈락 (MCAR)")):
    locf = [r[3] for r in res[informative]]
    cc = [r[2] for r in res[informative]]
    # 두 선이 거의 포개진다 — 그것이 요점이므로 둘 다 보이게 그린다.
    axL.plot(q, locf, color=color, lw=4.0, alpha=0.35, zorder=4,
             solid_capstyle="round", label=f"{label} — LOCF")
    axL.plot(q, cc, marker=marker, color=color, lw=1.5, ls=(0, (5, 2)), ms=5,
             mfc="white", mew=1.4, zorder=5, label=f"{label} — 완전사례")
# 같은 탈락률에서 벌어지는 거리
i10 = KS.index(4)
axL.annotate("", xy=(q[i10], res[True][i10][3]), xytext=(q[i10], res[False][i10][3]),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
axL.text(q[i10] + 0.012, 0.135,
         f"같은 탈락률 $0.1$ 에서\n${res[False][i10][3]:.4f}$ 대 ${res[True][i10][3]:.4f}$",
         fontsize=9, color=INK)
axL.set_xlabel("탈락률")
axL.set_ylabel("귀무가설을 기각하는 비율")
axL.set_title("탈락의 수가 아니라 이유가 정한다", color=INK, fontsize=12, pad=9)
axL.set_ylim(-0.03, 1.02)
axL.set_xlim(-0.012, 0.52)
axL.legend(fontsize=8.5, loc="center right", framealpha=0.95)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ---------------------------------------------- 오른쪽: 그 까닭
d, mask = keep_example
d0, m0 = d[:4000].ravel(), mask[:4000].ravel()
bins = np.linspace(-7, 7, 61)
axR.hist(d0[~m0], bins=bins, color=BLUE_F, edgecolor=BLUE, lw=0.5,
         label="남은 사람의 참 변화", zorder=3)
axR.hist(d0[m0], bins=bins, color=RED_F, edgecolor=RED, lw=0.5,
         label="떠난 사람의 참 변화", zorder=4)
h = axR.get_ylim()[1]
axR.annotate("", xy=(0, h * 0.52), xytext=(-2.6, h * 0.52),
             arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2))
axR.text(-2.5, h * 0.57, "LOCF 는 이들을\n변화 $0$ 으로 적는다",
         fontsize=9.5, color=RED, ha="left")
axR.axvline(0, color=GREEN, lw=2.0, zorder=6)
axR.text(0.18, h * 0.90, "$0$", fontsize=11, color=GREEN)
axR.set_xlabel("한 사람의 참 변화 $d_i$")
axR.set_ylabel("사람 수")
axR.set_title("왼쪽 꼬리가 가운데로 옮겨 간다", color=INK, fontsize=12, pad=9)
axR.legend(fontsize=9, loc="upper right", framealpha=0.95)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "locf_dropout.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)

# 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
print(f"쌍 {N} 개, 반복 {M:,}, 참 평균변화 0, 유의수준 0.05")
for informative in (False, True):
    print("\n-- 나빠진 사람이 떠난다" if informative else "\n-- 무관한 탈락 (MCAR)")
    print("  탈락   탈락률   탈락 없음    완전사례      LOCF")
    for k, full, cc, locf in res[informative]:
        print(f"{k:6d} {k / N:8.2f} {full:10.4f} {cc:11.4f} {locf:9.4f}")
print("\nsaved", path)
