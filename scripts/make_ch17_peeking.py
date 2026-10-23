r"""17.1절 「그림으로 보는 통계학」 — 순열검정도 「언제 멈출지」는 고쳐 주지 않는다.

A/B 검정을 돌려 놓고 자꾸 들여다보면, 한 번도 효과가 없는데도 어느 순간 유의해
보이는 일이 생긴다. 이 스크립트는 그 값을 두 가지로 잰다.

  ① 귀무가설이 참인 자료에서, 중간에 $k$ 번 들여다보며 그때마다 $5\%$ 수준으로
     검정할 때 **한 번이라도 유의**가 될 확률. $k$ 를 $1,2,5,10,20,50$ 으로 둔다
  ② 그 일이 $t$ 검정만의 사정이 아님 — 같은 설정에서 **순열검정**으로 판정해도
     거짓양성률이 거의 같다. 이 쪽이 가르치는 방법이 정확한 검정이라는 사실과
     멈추는 규칙은 별개다

①은 분산을 안다고 두고 $Z_n=(\sum a-\sum b)/\sqrt{2n}$ 을 누적합으로 한 번에
계산한다(귀무에서 정확히 표준정규다). ②는 라벨을 실제로 섞어 순열 p 값을 구하므로
느리다 — 전체가 $1$–$2$ 분 걸린다.

만드는 파일:

  ch17/permutation/img/peeking.png

실행:  python3 scripts/make_ch17_peeking.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch17/permutation/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

ALPHA = 0.05
ZCRIT = stats.norm.ppf(1 - ALPHA / 2)
LOOKS = (1, 2, 5, 10, 20, 50)

REPS_Z, NMAX_Z = 50_000, 1000       # ① 큰 모의실험
REPS_P, NMAX_P, K_P, B_PERM = 300, 400, 10, 499   # ② 순열 확인


def peeking_rates(rng):
    """누적합으로 Z 경로를 만들고, 들여다본 횟수마다 거짓양성률을 센다."""
    a = rng.standard_normal((REPS_Z, NMAX_Z))
    b = rng.standard_normal((REPS_Z, NMAX_Z))
    z = (np.cumsum(a, 1) - np.cumsum(b, 1)) / np.sqrt(2.0 * np.arange(1, NMAX_Z + 1))
    out = {}
    for k in LOOKS:
        idx = np.linspace(NMAX_Z / k, NMAX_Z, k).astype(int) - 1
        out[k] = float((np.abs(z[:, idx]) > ZCRIT).any(1).mean())
    # 전체를 5% 로 지키려면 한 번당 얼마를 써야 하는가 (10 번 볼 때)
    idx10 = np.linspace(NMAX_Z / 10, NMAX_Z, 10).astype(int) - 1
    cmax = np.abs(z[:, idx10]).max(1)
    c = float(np.quantile(cmax, 1 - ALPHA))
    return out, c, 2 * (1 - stats.norm.cdf(c))


def permutation_check(rng):
    """같은 설정을 순열검정으로 판정한다 — 마지막 한 번만 볼 때와 열 번 볼 때."""
    looks = np.linspace(NMAX_P / K_P, NMAX_P, K_P).astype(int)
    ever_t = ever_p = last_t = last_p = 0
    for _ in range(REPS_P):
        a = rng.standard_normal(NMAX_P)
        b = rng.standard_normal(NMAX_P)
        st = sp = False
        for j, n in enumerate(looks):
            obs = abs(a[:n].mean() - b[:n].mean())
            sig_t = obs / np.sqrt(2.0 / n) > ZCRIT
            x = np.concatenate([a[:n], b[:n]])
            xs = x[np.argsort(rng.random((B_PERM, 2 * n)), axis=1)]
            d = np.abs(xs[:, :n].mean(1) - xs[:, n:].mean(1))
            sig_p = (1 + int((d >= obs).sum())) / (B_PERM + 1) < ALPHA
            st = st or sig_t
            sp = sp or sig_p
            if j == len(looks) - 1:
                last_t += sig_t
                last_p += sig_p
        ever_t += st
        ever_p += sp
    return (last_t / REPS_P, last_p / REPS_P, ever_t / REPS_P, ever_p / REPS_P)


def main():
    rng = np.random.default_rng(17)
    rates, crit, per_look = peeking_rates(rng)
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"① 귀무가설이 참인 자료 {REPS_Z} 번, 집단당 최대 {NMAX_Z} 명")
    print("  들여다본 횟수   한 번이라도 유의할 확률")
    for k in LOOKS:
        print(f"  {k:11d} {rates[k]:22.4f}")
    print(f"\n  10 번 볼 때 전체 {ALPHA:.0%} 를 지키는 임계값 {crit:.3f} "
          f"— 한 번당 알파 {per_look:.4f}")

    lt, lp, et, ep = permutation_check(np.random.default_rng(71))
    print(f"\n② 순열검정으로 다시 판정 ({REPS_P} 번, 집단당 최대 {NMAX_P} 명, "
          f"섞기 {B_PERM} 번)")
    print(f"  마지막에 한 번만 볼 때   t {lt:.3f}   순열 {lp:.3f}")
    print(f"  열 번 들여다볼 때        t {et:.3f}   순열 {ep:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)

    # ── ① 들여다본 횟수와 거짓양성률 ──
    ax = axes[0]
    ks = np.array(LOOKS, dtype=float)
    vs = np.array([rates[k] for k in LOOKS])
    ax.plot(ks, vs, "o-", ms=9, lw=2.6, color=RED)
    ax.axhline(ALPHA, color=INK, lw=1.6, ls="--")
    ax.text(50, ALPHA + 0.012, f"약속한 ${ALPHA:.2f}$", ha="right", fontsize=11,
            color=INK)
    for k, v in zip(ks, vs):
        ax.annotate(f"${v:.3f}$", xy=(k, v), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=10.4,
                    color=RED)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{int(k)}" for k in ks], fontsize=11)
    ax.set_xlim(0.85, 62)
    ax.set_ylim(0, 0.37)
    ax.set_xlabel("자료가 쌓이는 동안 들여다본 횟수", fontsize=11.5)
    ax.set_ylabel("한 번이라도 「유의」가 될 확률", fontsize=11.5)
    ax.set_title("① 효과가 하나도 없는 자료에서", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 순열검정도 같다 ──
    ax = axes[1]
    xs = np.arange(2)
    w = 0.36
    ax.bar(xs - w / 2, [lt, et], width=w, color=BLUE, alpha=0.85, label="$t$ 검정")
    ax.bar(xs + w / 2, [lp, ep], width=w, color=TEAL, alpha=0.85, label="순열검정")
    for x, v in zip(xs - w / 2, [lt, et]):
        ax.text(x, v + 0.008, f"${v:.3f}$", ha="center", fontsize=11.5, color=BLUE,
                fontweight="bold")
    for x, v in zip(xs + w / 2, [lp, ep]):
        ax.text(x, v + 0.008, f"${v:.3f}$", ha="center", fontsize=11.5, color=TEAL,
                fontweight="bold")
    ax.axhline(ALPHA, color=INK, lw=1.6, ls="--")
    ax.text(1.46, ALPHA + 0.008, f"약속한 ${ALPHA:.2f}$", ha="right", fontsize=11,
            color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(["마지막에 한 번만 본다", f"{K_P} 번 들여다본다"],
                       fontsize=11.5)
    ax.set_ylim(0, 0.29)
    ax.set_ylabel("거짓양성률", fontsize=11.5)
    ax.set_title("② 정확한 검정이어도 마찬가지다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10.6, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("무엇으로 검정하느냐가 아니라 언제 멈추느냐의 문제다", fontsize=14.5)
    # 설명 줄에는 마크다운이 먹지 않고 천 단위 쉼표도 수식으로 넣어야 한다
    fig.text(0.5, -0.05,
             f"두 판 모두 두 집단이 같은 분포인 자료다. 왼쪽은 "
             f"${REPS_Z // 10000}$ 만 번, 오른쪽은 ${REPS_P}$ 번 되풀이해 세었고, "
             f"오른쪽의 순열 p 값은 라벨을 ${B_PERM}$ 번 섞어 구했다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "peeking.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
