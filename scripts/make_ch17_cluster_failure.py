r"""덩어리(cluster) 크기로 하는 추론이 왜 무너지는가 — 그리고 재표집은 왜 안 무너지는가.

뇌영상의 고전적 처방은 잡음 영상의 **공간 자기상관이 가우스 꼴**이라고 가정하고
덩어리 크기의 임계값을 계산한다. 실제 자기상관은 가까운 거리에서는 가우스와 거의
같지만 **먼 거리에서 더 높다**. 국소 매끄러움(= 원점에서의 곡률)이 같아도 꼬리가
두꺼우면 덩어리가 더 크게 자라고, 가우스 가정으로 잡은 임계값은 너무 낮다.

재표집은 그 가정을 아예 쓰지 않는다. 자료 자체의 최대 덩어리 분포에서 임계값을
읽으므로 자기상관이 어떤 꼴이든 수준을 지킨다.

① 두 자기상관함수 — 원점 곡률(국소 매끄러움)은 같고 꼬리만 다르다.
② 최대 덩어리 크기의 귀무분포와, 가우스 가정이 잡은 $95\%$ 임계값.
③ 실제 집단별오류율 — 덩어리 기준값(CDT)별로, 가정으로 잡을 때와 재표집으로 잡을 때.

만드는 파일: docs/ch17/permutation/img/cluster_failure.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import ndimage, stats  # noqa: E402

RNG = np.random.default_rng(20261020)
L = 160            # 2 차원 격자 한 변
S = 3.4            # 가우스 자기상관의 폭 (화소)
W, K = 0.75, 6.0   # 꼬리: 분산의 1-W 를 K 배 넓은 성분에 준다
N_CAL, N_EVAL = 4000, 4000
CDTS = [(2.3263, "0.01"), (3.0902, "0.001")]  # 덩어리 기준값 z 와 그 p

# --- 꼬리를 붙이되 원점 곡률은 그대로 ---------------------------------
# rho(r) = w exp(-r^2/2s1^2) + (1-w) exp(-r^2/2(k s1)^2) 의 원점 곡률은
# -(w/s1^2 + (1-w)/(k s1)^2) 이다. 순수 가우스 exp(-r^2/2S^2) 의 곡률
# -1/S^2 과 맞추려면 s1^2 = S^2 (w + (1-w)/k^2) 로 두면 된다.
S1 = S * np.sqrt(W + (1 - W) / K**2)


def spectrum(widths, weights):
    """자기공분산이 가우스들의 혼합일 때의 출력스펙트럼 (FFT 로 장을 만든다)."""
    fy = np.fft.fftfreq(L)[:, None]
    fx = np.fft.fftfreq(L)[None, :]
    q2 = (2 * np.pi) ** 2 * (fx**2 + fy**2)
    P = np.zeros((L, L))
    for s, w in zip(widths, weights):
        P += w * (2 * np.pi * s**2) * np.exp(-0.5 * q2 * s**2)
    return P


SPEC = {
    "gauss": spectrum([S], [1.0]),
    "heavy": spectrum([S1, K * S1], [W, 1 - W]),
}


def field(kind, rng):
    """주어진 자기상관을 갖고 화소별로 표준정규인 장을 하나 만든다."""
    z = np.fft.fft2(rng.standard_normal((L, L)))
    g = np.real(np.fft.ifft2(z * np.sqrt(SPEC[kind])))
    return g / g.std()


def max_cluster(g, cdt):
    """CDT 를 넘는 화소를 이어 붙여 가장 큰 덩어리의 화소 수를 센다."""
    lab, n = ndimage.label(g > cdt)
    if n == 0:
        return 0
    return int(np.bincount(lab.ravel())[1:].max())


def acf_1d(kind):
    """스펙트럼에서 되돌린 자기상관함수를 가로 방향으로 읽는다."""
    cov = np.real(np.fft.ifft2(SPEC[kind]))
    return (cov[0] / cov[0, 0])[: L // 2]


# === 모의실험 =======================================================
draws = {}
for kind in ("gauss", "heavy"):
    # 장 하나를 만들면 두 CDT 에서 모두 재는 것이 싸다
    out = [[], []]
    for _ in range(N_CAL + N_EVAL):
        g = field(kind, RNG)
        for j, (cdt, _) in enumerate(CDTS):
            out[j].append(max_cluster(g, cdt))
    for j, (cdt, _) in enumerate(CDTS):
        draws[(kind, cdt)] = np.array(out[j])
    print(f"  {kind} 끝")

rows = []
for cdt, ptxt in CDTS:
    cal_g = draws[("gauss", cdt)][:N_CAL]
    ev_h = draws[("heavy", cdt)][N_CAL:]
    cal_h = draws[("heavy", cdt)][:N_CAL]
    t_assume = np.quantile(cal_g, 0.95)   # 가우스 가정으로 잡은 임계값
    t_resamp = np.quantile(cal_h, 0.95)   # 자료 자체로 잡은 임계값
    rows.append({
        "p": ptxt, "cdt": cdt,
        "t_assume": t_assume, "t_resamp": t_resamp,
        "fwe_assume": np.mean(ev_h > t_assume) * 100,
        "fwe_resamp": np.mean(ev_h > t_resamp) * 100,
        "fwe_gauss": np.mean(draws[("gauss", cdt)][N_CAL:] > t_assume) * 100,
    })

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))

r = np.arange(L // 2)
for kind, c, lab in (("gauss", "#d62728", "가정한 가우스 꼴"),
                     ("heavy", "#1f77b4", "꼬리가 두꺼운 실제 꼴")):
    ax1.semilogy(r, np.clip(acf_1d(kind), 1e-5, None), color=c, lw=2.4, label=lab)
ax1.set_xlim(0, 28)
ax1.set_ylim(1e-4, 1.4)
ax1.set_xlabel("두 화소 사이의 거리 (화소)")
ax1.set_ylabel("공간 자기상관")
ax1.set_title("① 원점 곡률은 같고 꼬리만 다르다", fontsize=12)
ax1.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
ax1.set_yticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
ax1.minorticks_off()
ax1.annotate("여기가 같으면\n매끄러움이 같다고 본다",
             xy=(2.0, acf_1d("gauss")[2]), xytext=(7.5, 0.35), fontsize=9.5,
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax1.annotate("여기서 갈린다", xy=(20, acf_1d("heavy")[20]), xytext=(11, 6e-4),
             fontsize=9.5, color="#1f77b4",
             arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0))
ax1.legend(loc="upper right", fontsize=9.5)
ax1.grid(alpha=0.25, which="major")

cdt0 = CDTS[0][0]
bins = np.linspace(0, np.quantile(draws[("heavy", cdt0)], 0.995), 60)
for kind, c, lab in (("gauss", "#d62728", "가우스 꼴 (가정)"),
                     ("heavy", "#1f77b4", "꼬리가 두꺼운 꼴 (실제)")):
    ax2.hist(draws[(kind, cdt0)], bins=bins, density=True, color=c, alpha=0.5,
             edgecolor="none", label=lab)
t_a = rows[0]["t_assume"]
ax2.axvline(t_a, color="#d62728", lw=1.8, ls="--")
ax2.text(t_a + 6, ax2.get_ylim()[1] * 0.78,
         f"가정이 잡은\n$95\\%$ 임계값\n{t_a:.0f} 화소", fontsize=9.5, color="#d62728")
t_r = rows[0]["t_resamp"]
ax2.axvline(t_r, color="#1f77b4", lw=1.8, ls="--")
ax2.text(t_r + 6, ax2.get_ylim()[1] * 0.40,
         f"재표집이 잡은\n$95\\%$ 임계값\n{t_r:.0f} 화소", fontsize=9.5, color="#1f77b4")
ax2.set_xlabel(f"가장 큰 덩어리의 화소 수 (CDT $p = {CDTS[0][1]}$)")
ax2.set_ylabel("밀도")
ax2.set_title("② 같은 매끄러움, 더 크게 자라는 덩어리", fontsize=12)
ax2.legend(loc="upper right", fontsize=9.5)
ax2.grid(alpha=0.25)

w = 0.34
xs_ = np.arange(len(rows))
a = [r_["fwe_assume"] for r_ in rows]
b = [r_["fwe_resamp"] for r_ in rows]
ax3.bar(xs_ - w / 2, a, w, color="#d62728", label="가우스 가정으로 임계값을 잡으면")
ax3.bar(xs_ + w / 2, b, w, color="#1f77b4", label="재표집으로 임계값을 잡으면")
for i, (p_, q_) in enumerate(zip(a, b)):
    ax3.text(i - w / 2, p_ + 0.7, f"{p_:.1f}%", ha="center", fontsize=10)
    ax3.text(i + w / 2, q_ + 0.7, f"{q_:.1f}%", ha="center", fontsize=10)
ax3.axhline(5, color="black", lw=1.5, ls="--")
ax3.text(0.5, 5.6, "명목 $5\\%$", ha="center", fontsize=10)
ax3.set_xticks(xs_)
ax3.set_xticklabels([f"CDT $p = {r_['p']}$" for r_ in rows])
ax3.set_ylabel("실제 집단별오류율 ($\\%$)")
ax3.set_title("③ 기준값을 느슨히 잡을수록 더 무너진다", fontsize=12)
ax3.set_ylim(0, max(a) * 1.3)
ax3.legend(loc="upper right", fontsize=9.5)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch17/permutation/img/cluster_failure.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
ag, ah = acf_1d("gauss"), acf_1d("heavy")
print(f"\n① 원점 곡률로 잰 반치폭: "
      f"가우스 {2.3548 * S:.2f} 화소, 꼬리 쪽 {2.3548 * S1 / np.sqrt(W + (1 - W) / K**2):.2f} 화소")
for d in (2, 5, 10, 15, 20, 25):
    print(f"   거리 {d:>2} 화소 — 가우스 {ag[d]:.2e}, 꼬리 {ah[d]:.2e} "
          f"({ah[d] / ag[d]:,.0f} 배)")

print(f"\n②③ 모의실험 {N_CAL:,} + {N_EVAL:,} 회 (격자 {L}×{L})")
print(f"{'CDT':>10}{'가정 임계값':>12}{'재표집 임계값':>14}"
      f"{'가정 FWE':>11}{'재표집 FWE':>12}{'가우스 자료 FWE':>16}")
for r_ in rows:
    print(f"{'p=' + r_['p']:>10}{r_['t_assume']:>11.0f}화소{r_['t_resamp']:>13.0f}화소"
          f"{r_['fwe_assume']:>10.1f}%{r_['fwe_resamp']:>11.1f}%{r_['fwe_gauss']:>15.1f}%")
print(f"\n  임계값의 비 (재표집 / 가정): "
      + ", ".join(f"p={r_['p']} 에서 {r_['t_resamp'] / r_['t_assume']:.1f} 배"
                  for r_ in rows))
