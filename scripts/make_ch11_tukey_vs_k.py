r"""11.5절 「그림으로 보는 통계학」 — 집단이 늘면 셋이 갈라진다.

이 쪽은 k = 3 에서 최대 |t| 의 귀무분포를 그려 보이고, 투키가 본페로니를
이기는 폭이 「k 가 커질수록 벌어진다」고 적어 두었다. 그 말을 k 를 2 에서
10 까지 올려 가며 수로 잰다.

  보정 없음  문턱이 거의 그대로이고 FWER 이 0.049 에서 0.608 로 치솟는다
  투키       문턱이 2.101 에서 3.244 로 오르고 FWER 은 어느 k 에서나 0.05
  본페로니    문턱이 투키보다 늘 조금 높고 FWER 이 0.035 까지 내려간다

k = 2 에서는 셋이 정확히 같다 — 쌍이 하나뿐이면 보정할 것이 없다.

투키의 문턱은 스튜던트화 범위분포의 분위수를 sqrt(2) 로 나눈 것이다.

만드는 파일:

  ch11/post_hoc/img/tukey_vs_k.png

실행:  python3 scripts/make_ch11_tukey_vs_k.py   (저장소 최상위에서)
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

OUT = "docs/ch11/post_hoc/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
TEAL = "#00838F"
BLUE = "#1565C0"
RED = "#D32F2F"

M, REPS, SEED = 10, 40_000, 1953
KS = (2, 3, 4, 5, 6, 8, 10)


def main():
    rng = np.random.default_rng(SEED)
    raw, tuk, bon = [], [], []
    f_raw, f_tuk, f_bon = [], [], []

    print(f"{'k':>3}{'쌍':>5}{'보정없음':>10}{'투키':>9}{'본페로니':>10}"
          f"{'FWER 없음':>11}{'FWER 투키':>11}{'FWER 본페':>11}")
    for k in KS:
        nu = k * (M - 1)
        m = k * (k - 1) // 2
        t_raw = float(stats.t.ppf(0.975, nu))
        t_bon = float(stats.t.ppf(1 - 0.05 / (2 * m), nu))
        q_tuk = float(stats.studentized_range.ppf(0.95, k, nu) / np.sqrt(2))

        y = rng.normal(0, 1, (REPS, k, M))            # 평균이 모두 같다
        gm = y.mean(2)
        mse = y.var(2, ddof=1).mean(1)
        i, j = np.triu_indices(k, 1)
        mx = (np.abs(gm[:, i] - gm[:, j]) / np.sqrt(mse[:, None] * 2 / M)).max(1)

        raw.append(t_raw); tuk.append(q_tuk); bon.append(t_bon)
        f_raw.append(float((mx > t_raw).mean()))
        f_tuk.append(float((mx > q_tuk).mean()))
        f_bon.append(float((mx > t_bon).mean()))
        print(f"{k:>3}{m:>5}{t_raw:>10.3f}{q_tuk:>9.3f}{t_bon:>10.3f}"
              f"{f_raw[-1]:>11.4f}{f_tuk[-1]:>11.4f}{f_bon[-1]:>11.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)

    ax = axes[0]
    ax.axhline(5, color=INK, lw=2.0, ls="--")
    ax.plot(KS, np.array(f_raw) * 100, marker="o", ms=7, lw=2.6, color=GREY,
            label="보정 없음")
    ax.plot(KS, np.array(f_bon) * 100, marker="s", ms=7, lw=2.6, color=BLUE,
            label="본페로니")
    ax.plot(KS, np.array(f_tuk) * 100, marker="D", ms=7, lw=2.8, color=TEAL,
            label="투키 HSD")
    ax.annotate(f"{f_raw[-1]*100:.1f}%\n열 번에 여섯 번",
                xy=(10, f_raw[-1] * 100), xytext=(7.0, f_raw[-1] * 100 - 4),
                fontsize=11, color=GREY, ha="center")
    ax.annotate(f"{f_bon[-1]*100:.1f}%", xy=(10, f_bon[-1] * 100),
                xytext=(8.3, 13), fontsize=10.8, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.annotate(f"어느 $k$ 에서나 {min(f_tuk)*100:.1f}–{max(f_tuk)*100:.1f}%",
                xy=(5, f_tuk[3] * 100), xytext=(3.3, 22), fontsize=10.8,
                color=TEAL, arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3))
    ax.set_xlabel("집단의 수 $k$", fontsize=11.5)
    ax.set_ylabel("가족단위 오류율 (%)", fontsize=11.5)
    ax.set_title("약속은 셋 다 5% 였다", fontsize=12.8, color=INK, pad=8)
    ax.set_xticks(KS)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(KS, raw, marker="o", ms=7, lw=2.6, color=GREY, label="보정 없음")
    ax.plot(KS, bon, marker="s", ms=7, lw=2.6, color=BLUE, label="본페로니")
    ax.plot(KS, tuk, marker="D", ms=7, lw=2.8, color=TEAL, label="투키 HSD")
    ax.fill_between(KS, tuk, bon, color=RED, alpha=0.18)
    ax.annotate("이 틈이 투키가 버는 검정력이다",
                xy=(8, (tuk[5] + bon[5]) / 2), xytext=(4.3, 3.42),
                fontsize=10.8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate("$k = 2$ 면 셋이 같다", xy=(2, raw[0]), xytext=(2.3, 1.78),
                fontsize=10.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.set_xlabel("집단의 수 $k$", fontsize=11.5)
    ax.set_ylabel("쌍별 $|t|$ 에 걸 문턱", fontsize=11.5)
    ax.set_title("투키의 문턱만 분포에서 나온다", fontsize=12.8, color=INK, pad=8)
    ax.set_xticks(KS)
    ax.set_ylim(1.7, 3.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("집단이 늘어날수록 세 방법이 갈라진다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"집단마다 $n = {M}$ 이고 평균을 모두 같게 두었다"
             f"({REPS//1000} 천 번). 본페로니가 안전한 쪽으로 틀리는 것은 "
             "맞지만, 그 안전함은 쓰지 않은 검정력으로 치르는 값이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "tukey_vs_k.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
