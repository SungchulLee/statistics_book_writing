"""9.4절 대응표본 t 검정 — 이득의 원천을 보지 않고 이득을 챙기는 법.

들머리가 묻는 것을 그린다. 이 검정이 받는 것은 (n, d-bar, s_D) 세 수뿐이고
쌍 안의 상관은 받지 않는다. 그런데도 상관이 클수록 더 센 판정이 나온다 ---
상관이 **s_D 안에 이미 들어와 있기** 때문이다.

같은 수의 사람, 같은 시작 분포, 같은 끝 분포를 두고 상관만 0.5 와 0.9 로
바꾸었다. 두 산점도의 가장자리 분포는 똑같은데 차이의 흩어짐이 20.0 과
8.9 로 갈리고, 그 한 수가 t 를 -28 과 -63 으로 가른다.

쪽의 paired_reduction 은 대응 t 가 차이의 일표본 검정임을 보이고
pairing_efficiency 는 짝짓기의 이득과 대가를 곡선으로 그린다. 여기서는
상관이 식의 어디로 들어오는지만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/paired_sample_tests/img/"
N, MU0, SIG, DROP = 1306, 105.0, 20.0, 14.9
MU1 = MU0 * (1 - DROP / 100)
RHOS = (0.5, 0.9)
rng = np.random.default_rng(68)
z0, z1 = rng.standard_normal(N), rng.standard_normal(N)

fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5),
                         gridspec_kw={"width_ratios": [1, 1, 1.25]})
res = []
for ax, rho in zip(axes[:2], RHOS):
    w0 = MU0 + SIG * z0
    w1 = MU1 + SIG * (rho * z0 + np.sqrt(1 - rho ** 2) * z1)
    d = w1 - w0
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(N))
    res.append((rho, w0, w1, d, sd, t))
    ax.plot(w0, w1, ".", ms=2.6, color="tab:blue", alpha=0.45)
    ax.set_xlim(40, 175); ax.set_ylim(20, 160)
    ax.set_xlabel("시작할 때 (kg)", fontsize=11)
    ax.set_title(f"쌍 안의 상관 {rho}", fontsize=12.5)
    ax.grid(alpha=0.2)
    ax.text(0.04, 0.95, f"시작 평균 {w0.mean():.1f}  표준편차 {w0.std(ddof=1):.1f}\n"
            f"끝 평균 {w1.mean():.1f}  표준편차 {w1.std(ddof=1):.1f}",
            transform=ax.transAxes, va="top", fontsize=10,
            color="0.35")
axes[0].set_ylabel("68 주 뒤 (kg)", fontsize=11)
axes[1].set_yticklabels([])

ax = axes[2]
bins = np.linspace(-75, 45, 70)
for (rho, _, _, d, sd, t), c in zip(res, ("tab:orange", "tab:green")):
    ax.hist(d, bins=bins, histtype="step", lw=2.4, color=c,
            label=f"상관 {rho} — $s_D$ = {sd:.1f},  $t$ = {t:.1f}")
ax.axvline(0, color="0.15", lw=2.0)
ax.text(1.5, ax.get_ylim()[1] * 0.95, "변화 없음", fontsize=10.5,
        color="0.15")
ax.axvline(res[0][3].mean(), color="0.45", ls="--", lw=1.6)
ax.text(res[0][3].mean(), ax.get_ylim()[1] * 0.46,
        f"평균 {res[0][3].mean():.1f} kg\n(두 경우가 거의 같다)", ha="center",
        va="top", fontsize=10.5, color="0.35")
ax.set_xlabel("68 주 뒤 빼기 시작 (kg)", fontsize=11)
ax.set_ylabel("사람 수", fontsize=11)
ax.set_title("검정이 보는 것은 이 한 열뿐이다", fontsize=12.5)
ax.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
ax.grid(axis="y", alpha=0.2)

fig.suptitle("가장자리 분포는 똑같은데 차이의 흩어짐이 갈린다 — 상관은 "
             "$s_D$ 로 들어온다", fontsize=13.5, y=1.0)
plt.tight_layout()
path = OUT + "correlation_enters_sd.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
for rho, w0, w1, d, sd, t in res:
    print(f"  상관 {rho}: 시작 {w0.mean():.2f}±{w0.std(ddof=1):.2f} · "
          f"끝 {w1.mean():.2f}±{w1.std(ddof=1):.2f} · 차이 평균 {d.mean():.2f} "
          f"· s_D {sd:.2f} · t {t:.2f}")
print(f"  이론 s_D = sigma*sqrt(2(1-rho)): "
      f"{[round(SIG*np.sqrt(2*(1-r)), 2) for r in RHOS]}")
