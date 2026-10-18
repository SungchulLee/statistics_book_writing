r"""3.5절 「AI로 보는 통계학 1」 — 「n >= 30 이면 된다」를 수로 확인하기.

두 칸으로 나눈다.

  왼쪽  베리-에센이 실제로 재는 것 — 표준화합의 sup|F_n - Phi| 를 참 sigma
        로 재고, 정리의 상한과 함께 그린다. 상한은 한 번도 깨지지 않지만
        치우친 Exp(1) 에서 7 배, 대칭인 Uniform 에서 30 배 느슨하다.

  오른쪽 실무에서 정작 모자란 까닭 — 신뢰구간을 만들 때는 sigma 를 추정한다.
        참 sigma 를 쓰면 n = 5 에서도 포함률이 95% 에 붙지만, 추정하면
        81% 로 떨어진다. 이 손실은 왜도가 아니라 sigma 추정 탓이고
        베리-에센은 여기에 아무 말도 하지 않는다.

만드는 파일:

  ch03/limits/img/berry_esseen_check.png

실행:  python3 scripts/make_ch03_berry_esseen_check.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch03/limits/img/"

INK = "#37474F"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"
AMBER = "#B07C2C"

NS = [5, 10, 20, 30, 50, 100, 200, 400]
B = 60_000
C = 0.4748                      # 정리 1 에 쓰는 상수
SEED = 11


# === 모의실험 ======================================================
def measure(rng):
    """표준화합의 sup 거리와, 신뢰구간 포함률을 참/추정 sigma 두 가지로."""
    su = 1 / np.sqrt(12)                        # Uniform(0,1) 의 sigma
    rho_u = np.mean(np.abs(rng.uniform(0, 1, 400_000) - 0.5) ** 3)

    sup_e, sup_u, cov_true, cov_est, cov_t, cov_tu = [], [], [], [], [], []
    for n in NS:
        e = rng.exponential(1.0, (B, n))
        u = rng.uniform(0, 1, (B, n))

        sup_e.append(stats.kstest((e.mean(1) - 1.0) * np.sqrt(n), "norm").statistic)
        sup_u.append(stats.kstest((u.mean(1) - 0.5) / (su / np.sqrt(n)), "norm").statistic)

        cov_true.append(np.mean(np.abs(e.mean(1) - 1.0) <= 1.96 / np.sqrt(n)))
        cov_est.append(np.mean(np.abs(e.mean(1) - 1.0)
                               <= 1.96 * e.std(1, ddof=1) / np.sqrt(n)))
        tc = stats.t.ppf(0.975, n - 1)          # 제대로 된 95% 구간의 임계값
        cov_t.append(np.mean(np.abs(e.mean(1) - 1.0)
                             <= tc * e.std(1, ddof=1) / np.sqrt(n)))
        cov_tu.append(np.mean(np.abs(u.mean(1) - 0.5)
                              <= tc * u.std(1, ddof=1) / np.sqrt(n)))

    nn = np.array(NS, float)
    return (np.array(sup_e), np.array(sup_u),
            C * 2.0 / np.sqrt(nn),                  # Exp(1): rho=2, sigma=1
            C * rho_u / (su ** 3 * np.sqrt(nn)),
            np.array(cov_true), np.array(cov_est),
            np.array(cov_t), np.array(cov_tu))


# === 그림 =========================================================
def main():
    rng = np.random.default_rng(SEED)
    sup_e, sup_u, be_e, be_u, cov_true, cov_est, cov_t, cov_tu = measure(rng)
    nn = np.array(NS, float)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), constrained_layout=True)

    # ── 왼쪽: 정리가 재는 것 ──
    ax = axes[0]
    ax.plot(nn, be_e, marker="^", ms=7, lw=2.0, color=RED, ls="--",
            label="상한 — $\\mathrm{Exp}(1)$")
    ax.plot(nn, sup_e, marker="o", ms=7, lw=2.4, color=RED,
            label="실제 $\\sup|F_n-\\Phi|$ — $\\mathrm{Exp}(1)$")
    ax.plot(nn, be_u, marker="^", ms=7, lw=2.0, color=TEAL, ls="--",
            label="상한 — $\\mathrm{Uniform}(0,1)$")
    ax.plot(nn, sup_u, marker="o", ms=7, lw=2.4, color=TEAL,
            label="실제 — $\\mathrm{Uniform}(0,1)$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
    yt = [0.003, 0.01, 0.03, 0.1, 0.3]
    ax.set_yticks(yt); ax.set_yticklabels(["0.003", "0.01", "0.03", "0.1", "0.3"])
    ax.minorticks_off()
    ax.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
    ax.set_ylabel("표준화합과 정규의 최대 거리 (로그 눈금)", fontsize=11.5)
    ax.set_title("정리 1 이 재는 것 — 상한은 깨지지 않지만 느슨하다",
                 fontsize=12.5, color=INK, pad=8)
    ax.legend(fontsize=9.2, loc="lower left", framealpha=0.92)
    ax.grid(alpha=0.25, which="both")
    ax.text(0.97, 0.93, f"치우친 쪽이 {sup_e[3]/sup_u[3]:.0f} 배 멀다 (n = 30)",
            transform=ax.transAxes, ha="right", fontsize=10, color=INK)

    # ── 오른쪽: 실무에서 모자라는 까닭 ──
    ax = axes[1]
    ax.axhline(95, color=MUTED, lw=1.6, ls="--")
    ax.text(nn[-1], 95.25, "명목 95%", ha="right", fontsize=10, color=MUTED)
    ax.plot(nn, cov_true * 100, marker="o", ms=7, lw=2.4, color=TEAL,
            label="참 sigma + 1.96 — 왜도는 거의 상쇄된다")
    ax.plot(nn, cov_t * 100, marker="s", ms=7, lw=2.4, color=AMBER,
            label="추정 sigma + t — 제대로 된 95% 구간")
    ax.plot(nn, cov_tu * 100, marker="s", ms=6, lw=1.8, color=AMBER, ls=":",
            alpha=0.75, label="같은 것을 대칭 Uniform 에서")
    ax.plot(nn, cov_est * 100, marker="x", ms=7, lw=1.8, color=MUTED, ls="--",
            label="추정 sigma + 1.96 — 임계값을 잘못 쓴 것")
    ax.axvline(30, color=INK, lw=1.0, ls=":")
    ax.text(31, 82.5, "흔히 말하는 n = 30", fontsize=9.8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
    ax.set_ylabel("명목 95% 구간의 실제 포함률 (%)", fontsize=11.5)
    ax.set_title("임계값을 바로잡아도 치우친 쪽은 남는다 — Exp(1)",
                 fontsize=12.5, color=INK, pad=8)
    ax.set_ylim(78, 98.5)
    ax.legend(fontsize=8.6, loc="lower right", framealpha=0.92)
    ax.grid(alpha=0.25)

    fig.suptitle("「n 이 30 이면 된다」가 무엇을 묻는 말인지에 달려 있다",
                 fontsize=14.5)

    path = OUT + "berry_esseen_check.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    for i, n in enumerate(NS):
        print(f"    n={n:4d}  sup={sup_e[i]:.4f} 상한={be_e[i]:.4f} | "
              f"참σ+z {cov_true[i]*100:.2f}%  추정σ+t {cov_t[i]*100:.2f}%  "
              f"추정σ+z {cov_est[i]*100:.2f}%  (대칭 t {cov_tu[i]*100:.2f}%)")


if __name__ == "__main__":
    main()
