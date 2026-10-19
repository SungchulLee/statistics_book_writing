r"""20.4절 「그림으로 보는 통계학」 — 시험지를 누가 썼는가.

MNIST 는 NIST 의 특수자료 두 벌을 섞어 만든 것이다.

  SD-3  인구조사국 직원들이 썼다      — 반듯하다
  SD-1  고등학생들이 썼다              — 지저분하다

NIST 가 원래 나눠 놓은 방식은 SD-3 으로 배우고 SD-1 로 시험 보는
것이었다. 르쿤과 코르테스와 버지스는 SD-3 이 SD-1 보다 「훨씬 깨끗하고
알아보기 쉽다」고 적고, 두 벌을 **섞었다.**

  훈련  SD-3 에서 30,000 장 + SD-1 에서 30,000 장 = 60,000 장
  시험  SD-3 에서  5,000 장 + SD-1 에서  5,000 장 = 10,000 장

그리고 조건을 하나 더 걸었다 — **훈련에 글씨를 낸 사람과 시험에 글씨를
낸 사람이 겹치지 않게** 했다. 훈련 60,000 장이 약 250 명의 것이다.

두 가지를 각각 재 본다.

패널 1  다른 사람들에게 시험을 보이면
  공무원으로 배우고 공무원에게 시험  →  어디서나 1% 근처로 평평하다
  공무원으로 배우고 고등학생에게 시험 →  두 집단이 벌어질수록 17% 까지
  이 틈은 모형이 덜 배운 것이 아니라 쓴 사람이 다른 것이다.

패널 2  같은 사람들에게 시험을 보이면
  사람을 통째로 나눔  →  사람당 낱장이 몇 장이든 3% 근처로 평평하다
  낱장을 섞어 나눔    →  사람당 128 장이면 0.09% — 32 배 부풀었다
  독립 단위는 낱장이 아니라 글씨를 쓴 사람이다.

만드는 파일:

  ch20/case_studies/img/mnist_split.png

실행:  python3 scripts/make_ch20_mnist_split.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch20/case_studies/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

SEED = 1998
DELTAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
PERS = (4, 8, 16, 32, 64, 128)


def nn_err(Xtr, ytr, Xte, yte):
    """1-최근접이웃 오류율. 거리는 전개식으로 재어 큰 배열을 만들지 않는다."""
    D = (Xte ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None] - 2 * Xte @ Xtr.T
    return float((ytr[D.argmin(1)] != yte).mean())


def panel1(delta, rng, C=10, d=12, n=60, sig=1.0, reps=40):
    """집단 A 로 배운 모형을 A 와 B 에서 각각 평가한다. B 는 밀려 있고 더 지저분하다."""
    eA = eB = 0.0
    for _ in range(reps):
        mu = rng.normal(size=(C, d)) * 1.6
        sh = rng.normal(size=(1, d))
        sh /= np.linalg.norm(sh)

        def draw(m, grp):
            y = rng.integers(0, C, m)
            x = mu[y] + rng.normal(0, sig, (m, d))
            if grp == "B":
                x = x + delta * sh + rng.normal(0, 0.45 * delta, (m, d))
            return x, y

        X, y = draw(n * C, "A")
        M = np.stack([X[y == c].mean(0) for c in range(C)])       # 집단별 평균
        for grp, acc in (("A", 0), ("B", 1)):
            t, ty = draw(2000, grp)
            e = nn_err(M, np.arange(C), t, ty)
            if acc == 0:
                eA += e
            else:
                eB += e
    return eA / reps, eB / reps


def panel2(per, rng, W=60, C=10, d=10, tau=1.25, sig=0.55, reps=80):
    """사람마다의 버릇이 그 사람의 모든 낱장에 똑같이 실린다. 두 가지 나누기를 견준다."""
    eI = eW = 0.0
    for _ in range(reps):
        mu = rng.normal(size=(C, d)) * 2.4
        s = rng.normal(0, tau, (W, d))                 # 사람마다의 버릇
        w = np.repeat(np.arange(W), per)
        y = rng.integers(0, C, W * per)
        X = mu[y] + s[w] + rng.normal(0, sig, (W * per, d))

        p = rng.permutation(W * per)                   # 낱장을 섞어 나눈다
        k = int(0.8 * W * per)
        eI += nn_err(X[p[:k]], y[p[:k]], X[p[k:]], y[p[k:]])

        pw = rng.permutation(W)                        # 사람을 통째로 나눈다
        trw = set(pw[:int(0.8 * W)].tolist())
        m = np.array([ww in trw for ww in w])
        eW += nn_err(X[m], y[m], X[~m], y[~m])
    return eI / reps, eW / reps


def main():
    rng = np.random.default_rng(SEED)

    print("  패널 1 — 공무원으로 배우고 누구에게 시험 보이는가")
    print(f"{'밀림':>8}{'공무원에게':>14}{'고등학생에게':>16}{'배수':>10}")
    a1, b1 = [], []
    for dl in DELTAS:
        a, b = panel1(dl, rng)
        a1.append(a); b1.append(b)
        print(f"{dl:>8.1f}{a:>14.4f}{b:>16.4f}{b/max(a,1e-9):>10.1f}")

    print("\n  패널 2 — 같은 사람이 양쪽에 있는가")
    print(f"{'사람당 낱장':>12}{'낱장을 섞어':>14}{'사람을 통째로':>16}{'부풀린 배수':>14}")
    i2, w2 = [], []
    for per in PERS:
        i, w = panel2(per, rng)
        i2.append(i); w2.append(w)
        print(f"{per:>12}{i:>14.4f}{w:>16.4f}{w/max(i,1e-9):>14.1f}")

    a1, b1 = np.array(a1), np.array(b1)
    i2, w2 = np.array(i2), np.array(w2)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 다른 사람들에게 시험을 보이면 ──
    ax = axes[0]
    ax.fill_between(DELTAS, a1 * 100, b1 * 100, color=RED, alpha=0.13, zorder=1)
    ax.plot(DELTAS, b1 * 100, lw=3.4, color=RED, marker="o", ms=7,
            label="공무원으로 배우고 「고등학생」에게 시험", zorder=3)
    ax.plot(DELTAS, a1 * 100, lw=3.2, color=TEAL, marker="D", ms=7,
            label="공무원으로 배우고 「공무원」에게 시험", zorder=3)
    ax.annotate(f"이 틈은 모형이 덜 배운 것이 아니라\n"
                f"쓴 사람이 다른 것이다 — ${b1[-1]/a1[-1]:.0f}$ 배",
                xy=(2.6, (a1[-2] + b1[-2]) / 2 * 100), xytext=(1.12, 13.0),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("두 집단의 글씨가 서로 다른 정도", fontsize=11.5)
    ax.set_ylabel("시험 오류율", fontsize=11.5)
    ax.set_title("다른 사람들에게 시험을 보이면", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0, 19.5)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_yticklabels(["$0\\%$", "$5\\%$", "$10\\%$", "$15\\%$"])
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 같은 사람들에게 시험을 보이면 ──
    ax = axes[1]
    ax.fill_between(PERS, i2 * 100, w2 * 100, color=BLUE, alpha=0.12, zorder=1)
    ax.plot(PERS, w2 * 100, lw=3.4, color=INK, marker="o", ms=7,
            label="사람을 통째로 나눈다 — 정직한 수", zorder=3)
    ax.plot(PERS, i2 * 100, lw=3.2, color=BLUE, marker="D", ms=7, ls="--",
            label="낱장을 섞어 나눈다 — 같은 사람이 양쪽에", zorder=3)
    ax.annotate(f"같은 사람의 낱장이 늘수록\n수가 좋아 보인다 — "
                f"${w2[-1]/i2[-1]:.0f}$ 배 부풀었다",
                xy=(110, i2[-1] * 100), xytext=(46, 6.3), fontsize=10.8,
                color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax.annotate("같은 사람을 더 시켜도\n나아지지 않는다",
                xy=(20, w2[2] * 100), xytext=(9.0, 6.3), fontsize=10.8,
                color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(list(PERS))
    ax.set_xticklabels([str(p) for p in PERS])
    ax.set_yticks([0.1, 0.3, 1, 3, 10])
    ax.set_yticklabels(["$0.1\\%$", "$0.3\\%$", "$1\\%$", "$3\\%$", "$10\\%$"])
    ax.minorticks_off()
    ax.set_xlabel("한 사람이 쓴 낱장 수", fontsize=11.5)
    ax.set_ylabel("시험 오류율", fontsize=11.5)
    ax.set_title("같은 사람들에게 시험을 보이면", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0.055, 11)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("시험지를 누가 썼는가", fontsize=14.5)
    fig.text(0.5, -0.06,
             "MNIST 는 두 가지를 모두 막았다. 서로 다른 두 집단을 "
             "$30{,}000$ 장씩 섞어 훈련과 시험이 같은 모집단을 보게 했고, "
             "훈련에 글씨를 낸 사람과 시험에 글씨를 낸 사람이 겹치지 않게 "
             "했다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "mnist_split.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
