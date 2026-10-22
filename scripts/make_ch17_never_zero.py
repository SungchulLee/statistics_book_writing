r"""17.5절 「그림으로 보는 통계학」 — 순열 $p$ 값은 결코 $0$ 이 아니다.

순열을 $m$ 번만 뽑고 $\hat p = K/m$ 을 보고하면, 그 검정의 크기는
$(\lfloor \alpha m \rfloor + 1)/(m+1)$ 로 **언제나 $1/(m+1)$ 아래로 내려가지 못한다.**
$1$ 을 더한 $p_u = (K+1)/(m+1)$ 은 그 아래로 내려간다. 차이는 한 번 검정할 때는
작지만 검정이 $50{,}000$ 개면 재앙이 된다.

① 검정의 크기 — 톱니는 $\alpha$ 위에서 놀고, $1$ 을 더한 쪽은 언제나 아래다.
② 전사체 $50{,}000$ 개에 본페로니 — 순진한 쪽은 거짓 발견을 **보장**한다.

만드는 파일: docs/ch17/comparison/img/never_zero.png
실행:  python3 scripts/make_ch17_never_zero.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch17/comparison/img/never_zero.png"
ALPHA, NTEST = 0.05, 50_000

# --- ① 검정의 크기 -----------------------------------------------------
m1 = np.arange(9, 2001)
size_hat = (np.floor(ALPHA * m1) + 1) / (m1 + 1)          # p̂ = K/m
size_u = np.floor(ALPHA * (m1 + 1)) / (m1 + 1)            # p_u = (K+1)/(m+1)
print(f"① α = {ALPHA}")
for m in (19, 20, 99, 100, 999, 1000):
    i = m - 9
    print(f"  m = {m:>5}:  p̂ 쪽 크기 {size_hat[i]:.4f}   1 을 더한 쪽 {size_u[i]:.4f}")
print(f"  m = 9..2000 에서 p̂ 쪽이 α 를 넘는 비율: "
      f"{np.mean(size_hat > ALPHA):.3f},  1 을 더한 쪽: {np.mean(size_u > ALPHA):.3f}")

# --- ② 검정이 50,000 개일 때 -------------------------------------------
m2 = np.unique(np.round(np.logspace(2, 7.2, 400)).astype(np.int64))
a_adj = ALPHA / NTEST                                     # 본페로니 문턱
p_hat_one = (np.floor(a_adj * m2) + 1) / (m2 + 1)
p_u_one = np.floor(a_adj * (m2 + 1)) / (m2 + 1)
fwer_hat = 1 - (1 - p_hat_one) ** NTEST
fwer_u = 1 - (1 - p_u_one) ** NTEST
print(f"\n② 검정 {NTEST:,} 개 · 본페로니 · α = {ALPHA}")
for m in (1_000, 10_000, 100_000, 1_000_000):
    i = int(np.argmin(np.abs(m2 - m)))
    print(f"  m = {m2[i]:>9,}:  p̂ 쪽 FWER {fwer_hat[i]:.4f}   "
          f"1 을 더한 쪽 {fwer_u[i]:.4f}   (p̂ = 0 인 검정 기댓값 "
          f"{NTEST / (m2[i] + 1):.1f} 개)")
need = m2[fwer_hat < ALPHA]
print(f"  p̂ 쪽 FWER 가 {ALPHA} 아래로 내려가는 데 드는 순열 횟수: "
      f"{need[0]:,} 번 이상")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
C1, C2 = "#1f6f8b", "#c0532b"

ax = axes[0]
ax.plot(m1, size_hat, color=C2, lw=1.4, label=r"$\hat p = K/B$ 를 쓸 때")
ax.plot(m1, size_u, color=C1, lw=1.4, label=r"$\tilde p = (K+1)/(B+1)$ 을 쓸 때")
ax.axhline(ALPHA, color="0.35", ls="--", lw=1.2)
ax.text(2000, ALPHA + 0.0012, r"약속한 $\alpha = 0.05$", color="0.3",
        fontsize=9, ha="right")
ax.set_xscale("log")
ax.set_xlabel("순열 횟수 B")
ax.set_ylabel(r"실제 제1종 오류율  $P(p \leq 0.05)$")
ax.set_title("① 약속한 크기와 실제 크기", fontsize=11)
ax.set_ylim(0.040, 0.062)
ax.legend(fontsize=9, loc="upper right")

ax = axes[1]
ax.plot(m2, fwer_hat, color=C2, lw=2, label=r"$\hat p = K/B$ 를 쓸 때")
ax.plot(m2, fwer_u, color=C1, lw=2, label=r"$\tilde p = (K+1)/(B+1)$ 을 쓸 때")
ax.axhline(ALPHA, color="0.35", ls="--", lw=1.2)
ax.axvline(NTEST / ALPHA, color="0.6", ls=":", lw=1.2)
ax.text(NTEST / ALPHA * 1.15, 0.45, "순열 백만 번", color="0.4", fontsize=9)
ax.text(1.3e2, ALPHA + 0.03, r"약속한 $\alpha = 0.05$", color="0.3", fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("순열 횟수 B")
ax.set_ylabel("적어도 하나를 잘못 기각할 확률")
ax.set_title(f"② 검정 {NTEST:,} 개에 본페로니를 씌우면", fontsize=11)
ax.set_ylim(-0.04, 1.06)
ax.legend(fontsize=9, loc="center left")

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
