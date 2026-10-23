r"""15.4 「Bartlett 검정 비정규성 민감도」 쪽의 상자에 들어가는 그림을 만든다.

이 쪽이 바틀렛을 때리는 데 쓰는 분포가 **대수정규**다. 왜 하필 그것인가 —
효과가 더해지지 않고 **곱해지면** 나오는 분포이기 때문이다(지브라의 비례효과 법칙).

  왼쪽  모두 같은 크기에서 출발해 해마다 임의의 비율로 자라게 둔다. 걸음이
        쌓일수록 크기의 **로그**가 정규에 가까워지고, 곧 크기 자체가 대수정규가 된다.
  오른쪽 그 자료로 바틀렛을 돌린다. 세 집단을 **같은 과정**에서 뽑았으므로 등분산이고
        기각하면 모두 거짓 양성이다. 걸음이 쌓일수록 기각률이 올라간다 —
        분포가 대수정규에 가까워지는 만큼 검정이 무너진다.

만드는 파일:
  ch15/bartlett_test/img/gibrat_bartlett.png

실행:  python3 scripts/make_ch15_gibrat.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.

주의: BLAS 스레드를 묶어 돌릴 것.
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
RED = "#C62828"
GREY = "#90A4AE"
GREEN = "#2E7D32"

OUT = "docs/ch15/bartlett_test/img/"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(1931)
SIG = 0.35          # 한 걸음의 로그 성장률 표준편차
STEPS = [1, 2, 4, 8, 16]
B, K, N = 20000, 3, 20
ALPHA = 0.05


def grow(m, n, t):
    """모두 1 에서 출발해 t 걸음 동안 임의의 비율로 자란 크기."""
    g = rng.normal(0.0, SIG, (m, n, t))
    return np.exp(g.sum(axis=2))


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.5))

# ---------------------------------------------- 왼쪽: 걸음이 쌓이면 대수정규가 된다
for t, a in zip((1, 8), (0.75, 0.75)):
    x = grow(1, 40000, t)[0]
    x = x / x.mean()
    axL.hist(x, bins=np.linspace(0, 4, 90), density=True, alpha=a,
             color=(BLUE_F if t == 1 else "#FFE0B2"),
             edgecolor=(BLUE if t == 1 else "#E65100"), lw=0.6,
             label=f"{t} 걸음  (누적 로그 sd {SIG*np.sqrt(t):.2f})", zorder=3)
axL.set_xlim(0, 4)
axL.set_xlabel("크기 (평균으로 나눈 값)")
axL.set_ylabel("밀도")
axL.set_title("여덟 걸음이면 Lognormal(0,1) 이다", color=INK, fontsize=12, pad=9)
axL.legend(fontsize=9.5, framealpha=0.95)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ---------------------------------------------- 오른쪽: 그만큼 바틀렛이 무너진다
crit = stats.chi2.ppf(1 - ALPHA, K - 1)
sizes, sds = [], []
for t in STEPS:
    X = grow(B * K, N, t).reshape(B, K, N)
    s2 = X.var(axis=2, ddof=1)
    nu = N - 1
    sp2 = s2.mean(axis=1)
    M = K * nu * np.log(sp2) - nu * np.log(s2).sum(axis=1)
    C = 1 + (K + 1) / (3 * K * nu)
    sizes.append(float((M / C > crit).mean()))
    sds.append(SIG * np.sqrt(t))          # 누적 로그 표준편차 — 닫힌 꼴이라 흔들리지 않는다

axR.plot(sds, sizes, "o-", color=RED, lw=2.4, ms=8, zorder=5)
for sd, sz, t in zip(sds, sizes, STEPS):
    axR.annotate(f"{t} 걸음\n{sz:.3f}", (sd, sz), textcoords="offset points",
                 xytext=(10, -22 if t >= 8 else 8), fontsize=9, color=INK)
axR.axhline(ALPHA, color=GREEN, lw=1.8, ls="--", zorder=3)
axR.text(sds[0], ALPHA + 0.04, "명목 $0.05$", fontsize=10, color=GREEN)
# 이 쪽이 쓰는 설정과 만나는 자리
axR.axvline(1.0, color=GREY, lw=1.3, ls=":", zorder=2)
axR.text(1.02, 0.17, "누적 로그 sd $= 1$\n곧 $\\text{Lognormal}(0,1)$\n— 이 쪽의 설정",
         fontsize=9.5, color=INK)
axR.set_xlabel("누적 로그 표준편차  $\\sigma\\sqrt{t}$")
axR.set_ylabel("등분산 자료를 기각한 비율")
axR.set_title("곱이 쌓인 만큼 바틀렛이 무너진다", color=INK, fontsize=12, pad=9)
axR.set_ylim(0, 1.0)
for s_ in ("top", "right"):
    axR.spines[s_].set_visible(False)

fig.tight_layout()
path = OUT + "gibrat_bartlett.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)

print(f"집단 {K} 개, 집단당 n = {N}, 반복 {B:,}, 명목 {ALPHA}, 한 걸음 로그 sd = {SIG}")
# 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
print("\n  걸음   누적 로그sd   바틀렛 기각률")
for t, sd, sz in zip(STEPS, sds, sizes):
    print(f"{t:>6}{sd:>12.2f}{sz:>14.4f}")
print("\nsaved", path)
