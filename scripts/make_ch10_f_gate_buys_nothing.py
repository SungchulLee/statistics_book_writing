r"""10.3절 투키 정직유의차 §1 — 전역 F 를 관문으로 두면 무엇을 얻고 무엇을 잃는가.

「분산분석이 유의했을 때 사후검정을 한다」는 문장이 교과서마다 적혀 있다.
투키가 스스로 FWER 을 통제하므로 그 관문이 필요 없다는 말도 함께 적혀 있는데,
어느 쪽도 수로 재어 보이지는 않는다. 둘을 같은 설정에서 잰다.

집단당 10 개, 표준편차 1, alpha = 0.05, 6 만 번.

  왼쪽  평균이 모두 같을 때의 가족오류율
        보정 없음은 실제로 k = 5 에서 0.273 이다 — 독립을 가정한 0.401 이 아니다
        투키만 쓰면 어느 k 에서나 0.048-0.051 로 명목 수준에 앉는다
        F 관문을 앞에 두면 0.043 에서 0.032 로 내려간다 — 보호를 더 사는 것이
        아니라 이미 있던 보호 위에 하나를 더 얹는 것이다

  오른쪽 한 집단만 1.5 만큼 떨어뜨렸을 때 그 쌍을 잡아낼 확률
        투키만은 k 와 무관하게 0.91 쯤이다
        F 관문을 앞에 두면 0.906 에서 0.867 로 내려간다
        관문이 깎는 몫이 k = 3 에서 0.004, k = 10 에서 0.046 으로 열 배가 된다
        집단이 늘수록 한 쌍의 차이가 전역 F 안에서 묽어지기 때문이다

그러니 관문은 **사지 않은 보호에 값을 치르는** 장치다.

만드는 파일:

  ch10/post_hoc/img/f_gate_buys_nothing.png

실행:  python3 scripts/make_ch10_f_gate_buys_nothing.py   (저장소 최상위에서)
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

OUT = "docs/ch10/post_hoc/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
TEAL = "#00838F"
BLUE = "#1565C0"
RED = "#D32F2F"
AMBER = "#EF6C00"

M, REPS, SEED, DELTA = 10, 60_000, 1953, 1.5
KS = (3, 4, 5, 6, 7, 8, 9, 10)


def one_k(rng, k):
    """집단 k 개에서 네 양을 잰다 — 귀무의 FWER 셋과 대립의 검정력 둘."""
    nu = k * (M - 1)
    q_tuk = stats.studentized_range.ppf(0.95, k, nu) / np.sqrt(2)
    t_raw = stats.t.ppf(0.975, nu)
    f_crit = stats.f.ppf(0.95, k - 1, nu)
    i, j = np.triu_indices(k, 1)

    # 완전 귀무 — 평균이 모두 같다
    y = rng.normal(0, 1, (REPS, k, M))
    gm, mse = y.mean(2), y.var(2, ddof=1).mean(1)
    F = M * gm.var(1, ddof=1) / mse
    mx = (np.abs(gm[:, i] - gm[:, j]) / np.sqrt(mse[:, None] * 2 / M)).max(1)
    gate = F > f_crit

    # 대립 — 첫 집단만 DELTA 만큼 떨어뜨리고 그 집단이 낀 쌍만 센다
    mu = np.zeros(k)
    mu[0] = DELTA
    z = rng.normal(0, 1, (REPS, k, M)) + mu[None, :, None]
    gm2, mse2 = z.mean(2), z.var(2, ddof=1).mean(1)
    F2 = M * gm2.var(1, ddof=1) / mse2
    hit = (np.abs(gm2[:, 0:1] - gm2[:, 1:]) / np.sqrt(mse2[:, None] * 2 / M) > q_tuk).any(1)

    return dict(
        raw=float((mx > t_raw).mean()),
        indep=1 - 0.95 ** (k * (k - 1) // 2),
        tukey=float((mx > q_tuk).mean()),
        gated=float((gate & (mx > q_tuk)).mean()),
        pw_tukey=float(hit.mean()),
        pw_gated=float((( F2 > f_crit) & hit).mean()),
    )


def main():
    rng = np.random.default_rng(SEED)
    r = {k: one_k(rng, k) for k in KS}

    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print("      완전 귀무의 FWER                    참 쌍 하나의 검정력")
    print("  k   보정없음  독립가정   투키만  F관문+투키    투키만  F관문+투키   깎인 몫")
    for k in KS:
        d = r[k]
        print(f"{k:3d}   {d['raw']:7.4f}  {d['indep']:7.4f}  {d['tukey']:7.4f}"
              f"  {d['gated']:9.4f}   {d['pw_tukey']:7.4f}  {d['pw_gated']:9.4f}"
              f"  {d['pw_tukey'] - d['pw_gated']:8.4f}")

    ks = np.array(KS)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # ---------- 왼쪽 : 관문은 보호를 더 사 주지 않는다 ----------
    axL.axhline(0.05, color=MUTED, lw=1, ls=":", zorder=1)
    axL.text(2.78, 0.062, "0.05", color=GREY, fontsize=9, va="bottom", ha="left")

    axL.plot(ks, [r[k]["indep"] for k in KS], color=MUTED, lw=1.8, ls="--",
             marker="o", ms=4, label="보정 없음 — 독립이라 치면")
    axL.plot(ks, [r[k]["raw"] for k in KS], color=RED, lw=2.4, marker="o", ms=5,
             label="보정 없음 — 실제")
    axL.plot(ks, [r[k]["tukey"] for k in KS], color=TEAL, lw=2.4, marker="s", ms=5,
             label="투키만")
    axL.plot(ks, [r[k]["gated"] for k in KS], color=BLUE, lw=2.4, marker="^", ms=5,
             label="F 관문을 통과한 뒤 투키")

    axL.annotate("", xy=(5, r[5]["raw"]), xytext=(5, r[5]["indep"]),
                 arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
    axL.annotate(f"k = 5 에서 독립 가정은 {r[5]['indep']:.3f} 인데\n"
                 f"실제로는 {r[5]['raw']:.3f} 다 — 쌍들이 같은\n집단평균을 돌려 쓰기 때문이다",
                 xy=(5.05, (r[5]["raw"] + r[5]["indep"]) / 2), xytext=(3.15, 0.545),
                 fontsize=9.5, color=INK, va="top",
                 arrowprops=dict(arrowstyle="->", color=GREY, lw=1))

    axL.set_ylim(0, 1.0)
    axL.set_xlabel("집단 수 k", fontsize=11, color=INK)
    axL.set_ylabel("평균이 모두 같을 때 한 쌍이라도 기각할 확률", fontsize=11, color=INK)
    axL.set_title("관문은 없던 보호를 사 주지 않는다", fontsize=13, color=INK, pad=10)
    axL.legend(fontsize=9.5, loc="upper left", frameon=False)

    # 아래 두 선은 본 축에서 눌려 보이지 않는다 — 그 띠만 떼어 확대한다
    axI = axL.inset_axes([0.54, 0.075, 0.43, 0.27], zorder=6)
    axI.patch.set_alpha(1.0)
    axI.axhline(0.05, color=MUTED, lw=1, ls=":")
    axI.plot(ks, [r[k]["tukey"] for k in KS], color=TEAL, lw=2, marker="s", ms=4)
    axI.plot(ks, [r[k]["gated"] for k in KS], color=BLUE, lw=2, marker="^", ms=4)
    axI.set_ylim(0.025, 0.060)
    axI.set_xticks(ks)
    axI.set_yticks([0.03, 0.04, 0.05])
    axI.tick_params(colors=GREY, labelsize=8)
    axI.set_title("아래 두 선만 확대", fontsize=9, color=GREY, pad=4)
    for s in ("top", "right"):
        axI.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axI.spines[s].set_color(MUTED)
    axI.set_facecolor("white")
    for s in ("left", "bottom"):
        axI.spines[s].set_zorder(7)

    # ---------- 오른쪽 : 관문은 검정력만 깎는다 ----------
    pw_t = np.array([r[k]["pw_tukey"] for k in KS])
    pw_g = np.array([r[k]["pw_gated"] for k in KS])

    axR.fill_between(ks, pw_g, pw_t, color=AMBER, alpha=0.22, zorder=1,
                     label="관문이 깎은 몫")
    axR.plot(ks, pw_t, color=TEAL, lw=2.4, marker="s", ms=5, zorder=3, label="투키만")
    axR.plot(ks, pw_g, color=BLUE, lw=2.4, marker="^", ms=5, zorder=3,
             label="F 관문을 통과한 뒤 투키")

    g3 = r[3]["pw_tukey"] - r[3]["pw_gated"]
    g10 = r[10]["pw_tukey"] - r[10]["pw_gated"]
    axR.annotate(f"k = 3 에서 {g3:.3f}", xy=(3, (r[3]["pw_tukey"] + r[3]["pw_gated"]) / 2),
                 xytext=(3.35, 0.9265), fontsize=10, color=AMBER,
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=1))
    axR.annotate(f"k = 10 에서 {g10:.3f}\n열 배다",
                 xy=(9.9, (r[10]["pw_tukey"] + r[10]["pw_gated"]) / 2),
                 xytext=(7.5, 0.8565), fontsize=10, color=AMBER, ha="left",
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=1))

    axR.set_ylim(0.850, 0.936)
    axR.set_xlabel("집단 수 k  (한 집단만 1.5 만큼 떨어져 있다)", fontsize=11, color=INK)
    axR.set_ylabel("그 쌍을 잡아낼 확률", fontsize=11, color=INK)
    axR.set_title("관문은 검정력만 깎는다 — 그 몫이 k 와 함께 열 배가 된다",
                  fontsize=13, color=INK, pad=10)
    axR.legend(fontsize=9.5, loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False)

    for ax in (axL, axR):
        ax.set_xticks(ks)
        ax.tick_params(colors=GREY, labelsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)

    fig.tight_layout()
    fig.savefig(OUT + "f_gate_buys_nothing.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("\n저장: " + OUT + "f_gate_buys_nothing.png")


if __name__ == "__main__":
    main()
