r"""균등 열둘을 더해 정규난수를 만들던 시절 — 하필 열둘인 까닭과 그 대가.

이 쪽은 균등분포에서 뽑은 평균이 종 모양이 되는 것을 본다. 계산기가 느리던 시절
그 성질을 **정규난수 생성기**로 썼다. $U_1 + \cdots + U_{12} - 6$ 이다.

① 하필 열둘인 까닭 — 분산이 $n/12$ 라 $n = 12$ 에서 정확히 $1$ 이 된다.
② 가운데는 거의 같은데 꼬리에서 갈린다. $\pm 6$ 너머는 **확률이 $0$** 이다.
③ 박스–뮐러는 균등 둘로 정확한 정규를 만든다 — 더할 필요가 없었다.

만드는 파일: docs/ch05/applications/img/twelve_uniforms.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)

# === ① 하필 열둘 =====================================================
print("① 균등 n 개를 더하면 분산이 n/12 다")
print(f"{'n':>5}{'이론 분산 n/12':>16}{'모의 분산':>12}{'첨도(초과)':>12}")
for n in (2, 6, 12, 24):
    s = RNG.random((200_000, n)).sum(1) - n / 2
    print(f"{n:>5}{n / 12:>16.4f}{s.var():>12.4f}"
          f"{stats.kurtosis(s):>12.4f}")
print("   n = 12 에서만 분산이 1 이라 **눈금을 다시 잡을 필요가 없다.** "
      "그래서 열둘이었다.")
print(f"   Irwin-Hall 의 초과첨도는 -6/(5n) 이므로 n = 12 에서 "
      f"{-6 / (5 * 12):+.4f} — 정규보다 살짝 **가볍다**.")

# === ② 꼬리에서 갈린다 ===============================================
R2 = 20_000_000
print(f"\n② 꼬리확률 P(Z > z) ({R2:,} 번)")
s12 = RNG.random((R2 // 10, 12)).sum(1) - 6
for _ in range(9):
    s12 = np.concatenate([s12, RNG.random((R2 // 10, 12)).sum(1) - 6])
ZS = [1, 2, 3, 4, 5, 6]
print(f"{'z':>4}{'정규':>14}{'균등 12 개':>16}{'비':>10}")
EMP, TRU = [], []
for z in ZS:
    e = float(np.mean(s12 > z))
    t = float(stats.norm.sf(z))
    EMP.append(e)
    TRU.append(t)
    print(f"{z:>4}{t:>14.3e}{e:>16.3e}{(e / t if t else np.nan):>10.4f}")
print(f"   z = 4 에서 벌써 {EMP[3] / TRU[3]:.2f} 배로 모자라고, "
      f"z >= 6 은 **원리적으로 나올 수 없다**(합의 최댓값이 6 이다).")
print(f"   정규라면 P(Z > 6) = {stats.norm.sf(6):.2e} 이고, "
      f"{R2:,} 번이면 {R2 * stats.norm.sf(6):.3f} 번쯤 나와야 한다.")

# === ③ 박스–뮐러 =====================================================
R3 = 20_000_000
u1 = RNG.random(R3)
u2 = RNG.random(R3)
bm = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
print(f"\n③ 박스–뮐러 — 균등 둘로 만든 정규 ({R3:,} 개)")
print(f"{'z':>4}{'정규':>14}{'박스-뮐러':>16}{'비':>10}")
BM = []
for z in ZS:
    e = float(np.mean(bm > z))
    BM.append(e)
    print(f"{z:>4}{stats.norm.sf(z):>14.3e}{e:>16.3e}"
          f"{e / stats.norm.sf(z):>10.4f}")
print(f"   z = 4 까지 비가 1 에 붙어 있다. 그 너머는 {R3:,} 번으로도 **재기에 "
      f"모자란다** — 정규라면 z > 5 가 {R3 * stats.norm.sf(5):.1f} 번, "
      f"z > 6 이 {R3 * stats.norm.sf(6):.2f} 번 나올 뿐이라 비가 흔들린다.")
print(f"   균등 12 개 쪽의 0 은 모자람이 아니라 **불가능**이다 — 이 둘을 "
      f"가르는 것이 ②의 점선이다.")
print(f"   관측된 최댓값 — 균등 12 개 {s12.max():.3f}, "
      f"박스–뮐러 {bm.max():.3f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_U, C_N, C_B = "#d62728", "#333333", "#1f77b4"

g = np.linspace(-4.2, 4.2, 400)
ax1.hist(s12[:2_000_000], bins=np.linspace(-4.2, 4.2, 160), density=True,
         color=C_U, alpha=0.55, label="$U_1+\\cdots+U_{12}-6$")
ax1.plot(g, stats.norm.pdf(g), color=C_N, lw=2.4, label="$N(0,1)$")
ax1.set_xlabel("값")
ax1.set_ylabel("밀도")
ax1.set_title("① 가운데는 거의 똑같다", fontsize=12)
ax1.legend(loc="upper right", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.semilogy(ZS, TRU, "-", color=C_N, lw=2.6, label="$N(0,1)$")
ax2.semilogy(ZS, np.maximum(EMP, 1e-9), "o--", color=C_U, lw=2.4, ms=9,
             label="균등 $12$ 개")
ax2.semilogy(ZS, np.maximum(BM, 1e-9), "^-", color=C_B, lw=2.0, ms=8,
             label="박스–뮐러")
ax2.axvline(6, color=C_U, lw=1.4, ls=":")
ax2.annotate("합의 최댓값이 $6$ 이라\n여기서 뚝 끊긴다", xy=(6, 2e-8),
             xytext=(4.0, 2.5e-7), fontsize=10.5, color=C_U,
             arrowprops=dict(arrowstyle="->", color=C_U, lw=1.1))
ax2.set_xlabel("$z$")
ax2.set_ylabel("$P(Z > z)$")
ax2.set_yticks([1e-9, 1e-7, 1e-5, 1e-3, 1e-1])
ax2.set_yticklabels(["$0$", "$10^{-7}$", "$10^{-5}$", "$10^{-3}$",
                     "$10^{-1}$"])
ax2.minorticks_off()
ax2.set_ylim(5e-10, 0.5)
ax2.set_title("② 꼬리에서 갈린다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10.5)
ax2.grid(alpha=0.25)

NSHOW = 4                       # z >= 5 는 2천만 번으로 잴 수 없다
rat_u = [e / t for e, t in zip(EMP[:NSHOW], TRU[:NSHOW])]
rat_b = [e / t for e, t in zip(BM[:NSHOW], TRU[:NSHOW])]
xs = np.arange(NSHOW)
w = 0.38
ax3.bar(xs - w / 2, rat_u, w, color=C_U, label="균등 $12$ 개")
ax3.bar(xs + w / 2, rat_b, w, color=C_B, label="박스–뮐러")
ax3.axhline(1.0, color=C_N, lw=1.8, ls="--")
for i in range(NSHOW):
    ax3.text(xs[i] - w / 2, rat_u[i] + 0.03, f"${rat_u[i]:.2f}$", ha="center",
             fontsize=10, color=C_U)
    ax3.text(xs[i] + w / 2, rat_b[i] + 0.03, f"${rat_b[i]:.2f}$", ha="center",
             fontsize=10, color=C_B)
ax3.annotate(f"$z$ 가 $5$ 이상인 곳은 {R3 // 10 ** 6} 백만 번으로도\n"
             f"잴 수 없어 뺐다", xy=(0.5, 1.28), fontsize=10,
             color="#555555", ha="center")
ax3.set_xticks(xs)
ax3.set_xticklabels([f"${z}$" for z in ZS[:NSHOW]])
ax3.set_xlabel("$z$")
ax3.set_ylabel("꼬리확률 / 참값")
ax3.set_ylim(0, 1.45)
ax3.set_title("③ 참값과의 비", fontsize=12)
ax3.legend(loc="lower left", fontsize=10.5)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch05/applications/img/twelve_uniforms.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
