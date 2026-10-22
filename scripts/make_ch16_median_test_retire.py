r"""16.5 Mood 중앙값검정 쪽의 상자에 들어가는 그림을 만든다.

프라이들린과 개스워스(2000)의 주장 — 중앙값검정은 **점근적으로 그것이 가장 강력한
이중지수분포에서조차** 다른 순위검정에 밀린다 — 을 직접 확인한다.

만드는 파일:
  ch16/multi_group_nonparametric/img/median_test_retire.png

실행:  python3 scripts/make_ch16_median_test_retire.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os
import time

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

OUT = "docs/ch16/multi_group_nonparametric/img/"
os.makedirs(OUT, exist_ok=True)

K = 3          # 집단 수 — 쪽의 설계와 맞춘다
ALPHA = 0.05
LAP_B = 1.0 / np.sqrt(2.0)     # 분산을 1 로 맞춘 이중지수(Laplace)


def draw(rng, reps, n, delta):
    """(reps, K, n) 꼴로 평균이 0, delta, 2*delta 인 Laplace 표본을 뽑는다."""
    x = rng.laplace(0.0, LAP_B, size=(reps, K, n))
    return x + (np.arange(K) * delta)[None, :, None]


def mood_reject(x):
    """Mood 중앙값검정 — 전체 중앙값 위/아래의 2 x K 표에 카이제곱."""
    reps, k, n = x.shape
    flat = x.reshape(reps, k * n)
    med = np.median(flat, axis=1)[:, None, None]
    above = (x > med).sum(axis=2).astype(float)      # (reps, k)
    A = above.sum(axis=1)[:, None]
    N = k * n
    Ea = n * A / N
    Eb = n * (N - A) / N
    below = n - above
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = ((above - Ea) ** 2 / Ea + (below - Eb) ** 2 / Eb).sum(axis=1)
    chi = np.nan_to_num(chi, nan=0.0, posinf=0.0)
    return stats.chi2.sf(chi, k - 1) < ALPHA


def kw_reject(x):
    """Kruskal-Wallis — 합친 순위의 집단간 변동."""
    reps, k, n = x.shape
    N = k * n
    r = stats.rankdata(x.reshape(reps, N), axis=1).reshape(reps, k, n)
    R = r.sum(axis=2)
    H = 12.0 / (N * (N + 1)) * (R ** 2 / n).sum(axis=1) - 3 * (N + 1)
    return stats.chi2.sf(H, k - 1) < ALPHA


def power(rng, reps, n, delta):
    x = draw(rng, reps, n, delta)
    return mood_reject(x).mean(), kw_reject(x).mean()


t0 = time.time()
rng = np.random.default_rng(2000)

# ----------------------------------------------- (다) 귀무가설에서의 실제 크기
REPS_C = 60000
size_ns = [8, 10, 12, 15, 20, 25, 30, 40, 55, 70, 90, 120, 160, 200]
sm, sk = [], []
print("귀무에서의 실제 크기 (이중지수)")
print("   n    Mood     KW")
for n in size_ns:
    a, b = power(rng, REPS_C, n, 0.0)
    sm.append(a)
    sk.append(b)
    print(f"{n:4d}  {a:6.4f}  {b:6.4f}")
print(f"Mood 크기의 범위 {min(sm):.4f} ~ {max(sm):.4f}, "
      f"KW 는 {min(sk):.4f} ~ {max(sk):.4f}\n")

# ----------------------------------------------- (가) n = 15 에서의 검정력 곡선
REPS_A = 20000
deltas = np.linspace(0.0, 1.0, 9)
pm, pk = [], []
print("집단당 n = 15, 이중지수분포")
print("delta    Mood     KW")
for d in deltas:
    a, b = power(rng, REPS_A, 15, d)
    pm.append(a)
    pk.append(b)
    print(f"{d:5.3f}  {a:6.3f}  {b:6.3f}")

# ----------------------------------------------- (나) 국소대립에서 n 을 키운다
# delta_n = C / sqrt(n) 으로 두면 두 곡선이 각자의 점근 검정력으로 수렴한다.
REPS_B = 12000
C = 1.45
ns = [10, 15, 25, 40, 70, 120, 200, 350, 600, 1000]
qm, qk = [], []
print(f"\n국소대립 delta = {C}/sqrt(n), 이중지수분포")
print("   n   delta    Mood     KW")
for n in ns:
    d = C / np.sqrt(n)
    reps = REPS_B if n <= 200 else REPS_B // 3
    a, b = power(rng, reps, n, d)
    qm.append(a)
    qk.append(b)
    print(f"{n:4d}  {d:6.3f}  {a:6.3f}  {b:6.3f}")

print(f"\n걸린 시간 {time.time() - t0:.1f} 초")

# ----------------------------------------------- 그림
fig, (axL, axR, axS) = plt.subplots(1, 3, figsize=(16.2, 4.3))

axL.plot(deltas, pk, "o-", color=BLUE, lw=2.2, ms=5, label="Kruskal-Wallis")
axL.plot(deltas, pm, "s--", color=ORANGE, lw=2.2, ms=5, label="Mood 중앙값검정")
axL.axhline(ALPHA, color=MUTED, lw=0.9, ls=":")
axL.text(0.03, 0.97, f"$\\delta=0$ 에서 크기는 ${pk[0]:.3f}$ 대 ${pm[0]:.3f}$ 로 맞아 있다\n"
                     f"$\\delta=0.625$ 에서 ${pk[5]:.3f}$ 대 ${pm[5]:.3f}$",
         fontsize=9, color=INK, va="top")
axL.set_xlabel("집단 간 이동폭 $\\delta$ (평균이 $0,\\ \\delta,\\ 2\\delta$)")
axL.set_ylabel("기각률")
axL.set_ylim(0, 1.02)
axL.set_title(f"집단당 $n=15$, 이중지수분포 — {REPS_A:,} 회",
              color=INK, fontsize=12, pad=9)
axL.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

axR.plot(ns, qk, "o-", color=BLUE, lw=2.2, ms=5, label="Kruskal-Wallis")
axR.plot(ns, qm, "s--", color=ORANGE, lw=2.2, ms=5, label="Mood 중앙값검정")
axR.set_xscale("log")
# 로그 눈금의 마이너스·지수는 mathtext 라 평문으로 박는다
axR.set_xticks(ns)
axR.set_xticklabels([str(v) for v in ns], fontsize=9)
axR.minorticks_off()
axR.set_xlabel("집단당 $n$ (가로축은 로그눈금)")
axR.set_ylabel("기각률")
axR.set_ylim(0, 1.02)
axR.set_title(f"이동폭을 $\\delta = {C}/\\sqrt{{n}}$ 로 묶어 놓고 $n$ 을 키우면",
              color=INK, fontsize=12, pad=9)
axR.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
axR.text(11, 0.33, "점근으로는 중앙값검정이 앞선다\n(이중지수에서 ARE $2$ 대 $1.5$)",
         fontsize=9, color=GREEN, va="top")
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

# ----------------------------------------------- (다) 크기가 흔들린다
axS.axhline(ALPHA, color=MUTED, lw=1.0, ls="--", zorder=1)
axS.plot(size_ns, sk, "o-", color=BLUE, lw=2.0, ms=4.5, zorder=3,
         label="Kruskal-Wallis")
axS.plot(size_ns, sm, "s--", color=ORANGE, lw=2.0, ms=4.5, zorder=3,
         label="Mood 중앙값검정")
axS.set_xscale("log")
show = [8, 15, 25, 40, 70, 120, 200]
axS.set_xticks(show)
axS.set_xticklabels([str(v) for v in show], fontsize=9)
axS.minorticks_off()
axS.set_xlabel("집단당 $n$ (가로축은 로그눈금)")
axS.set_ylabel("귀무가설에서의 실제 기각률")
axS.set_ylim(0, 0.085)
axS.set_title(f"$0.05$ 를 겨눈 검정의 실제 크기 — {REPS_C:,} 회",
              color=INK, fontsize=12, pad=9)
axS.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
axS.text(8.4, 0.0795, f"Mood 는 ${min(sm):.3f}$–${max(sm):.3f}$ 사이를 오르내린다\n"
                      f"(KW 는 ${min(sk):.3f}$–${max(sk):.3f}$)",
         fontsize=9, color=INK, va="top")
for s in ("top", "right"):
    axS.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "median_test_retire.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
