r"""10.3절 사후비교 방법의 비교와 검정력 §4 — 쌍이 적으면 홀름이 잃는 것이 거의 없다.

이 절의 표는 임계값이 하나로 떨어지는 네 방법만 담는다. 홀름은 단계적 하강이라
고정 임계값이 없어 그 표에 들어가지 못하고, 그래서 연습문제로만 다루어져 왔다.
투키와 맞대어 두 양을 잰다 — 완전 귀무의 가족오류율과 참 쌍 하나의 검정력.

집단당 10 개, 표준편차 1, alpha = 0.05, 6 만 번. 쌍별 p 값은 합동 MSE 와
자유도 N - k 로 만든 t 검정에서 나온 것이고, 홀름은 그 p 값들에만 작용한다.

  왼쪽  투키는 어느 k 에서나 0.049-0.051 로 명목 수준에 앉는다. 쌍별 비교의
        상관을 알고 쓰기 때문이다. 홀름은 합집합 한계만 쓰므로 그 상관을
        버리고 0.042 에서 0.035 로 내려간다 — 보수적이다

  오른쪽 그 보수성의 대가가 검정력이다. 다만 k = 3 에서 1.4 퍼센트포인트뿐이고
        k = 8 에서도 2.6 이다. 쌍이 적으면 잃는 것이 거의 없다

홀름을 고를 이유는 검정력이 아니다. 균형도 등분산도 스튜던트화 범위표도
필요 없고 p 값 목록만 있으면 선다는 것, 그리고 어떤 종속 구조에서도
타당하다는 것이다. 대신 동시신뢰구간이 나오지 않는다.

만드는 파일:

  ch10/post_hoc/img/holm_vs_tukey.png

실행:  python3 scripts/make_ch10_holm_vs_tukey.py   (저장소 최상위에서)
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
PURPLE = "#6A1B9A"
AMBER = "#EF6C00"

M, REPS, SEED, DELTA = 10, 60_000, 7, 1.5
KS = (3, 4, 5, 6, 7, 8)


def holm_rejects(p, alpha=0.05):
    """단계적 하강 — 오름차순 p 를 alpha/(m-i) 와 차례로 보고 처음 실패에서 멈춘다.

    멈춘 자리의 가설과 그 뒤의 가설은 모두 남긴다. 되돌려 준 배열은
    원래 쌍 차례에 맞춘 기각 여부다.
    """
    m = p.shape[1]
    order = np.argsort(p, 1)
    ps = np.take_along_axis(p, order, 1)
    ok = ps < alpha / (m - np.arange(m))
    stop = np.where(ok.all(1), m, ok.argmin(1))        # 기각한 개수
    rej = np.zeros_like(ok)
    np.put_along_axis(rej, order, np.arange(m)[None, :] < stop[:, None], 1)
    return rej


def one_k(rng, k, shift):
    nu = k * (M - 1)
    q = stats.studentized_range.ppf(0.95, k, nu) / np.sqrt(2)
    i, j = np.triu_indices(k, 1)

    mu = np.zeros(k)
    mu[0] = shift
    y = rng.normal(0, 1, (REPS, k, M)) + mu[None, :, None]
    gm, mse = y.mean(2), y.var(2, ddof=1).mean(1)
    t = np.abs(gm[:, i] - gm[:, j]) / np.sqrt(mse[:, None] * 2 / M)
    rej_h = holm_rejects(2 * stats.t.sf(t, nu))
    rej_t = t > q

    if shift == 0:                                     # 완전 귀무 — 하나라도 기각하면 오류
        return float(rej_t.any(1).mean()), float(rej_h.any(1).mean())
    true_pair = i == 0                                 # 0 번 집단이 낀 쌍만 참이다
    return float(rej_t[:, true_pair].any(1).mean()), float(rej_h[:, true_pair].any(1).mean())


def main():
    rng = np.random.default_rng(SEED)
    fw_t, fw_h, pw_t, pw_h = [], [], [], []

    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print("        완전 귀무의 FWER        참 쌍 하나의 검정력")
    print("  k   m    투키    홀름        투키    홀름       차")
    for k in KS:
        a, b = one_k(rng, k, 0.0)
        c, d = one_k(rng, k, DELTA)
        fw_t.append(a); fw_h.append(b); pw_t.append(c); pw_h.append(d)
        print(f"{k:3d} {k*(k-1)//2:3d}  {a:.4f}  {b:.4f}      "
              f"{c:.4f}  {d:.4f}  {c-d:+.4f}")

    ks = np.array(KS)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # ---------- 왼쪽 : 투키는 명목에 앉고 홀름은 그 아래로 ----------
    axL.axhline(0.05, color=MUTED, lw=1, ls=":")
    axL.text(7.75, 0.0505, "명목 0.05", color=GREY, fontsize=9, ha="right", va="bottom")
    axL.plot(ks, fw_t, color=TEAL, lw=2.4, marker="s", ms=6, label="투키 HSD")
    axL.plot(ks, fw_h, color=PURPLE, lw=2.4, marker="D", ms=5.5, label="쌍별 t + 홀름")
    axL.fill_between(ks, fw_h, fw_t, color=PURPLE, alpha=0.10)
    axL.annotate("홀름은 쌍들의 상관을 버리고 합집합\n한계만 쓰므로 명목 아래로 내려간다",
                 xy=(6.05, fw_h[3]), xytext=(4.25, 0.0462),
                 fontsize=9.5, color=INK, va="top",
                 arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    axL.set_ylim(0.030, 0.058)
    axL.set_ylabel("평균이 모두 같을 때 한 쌍이라도 기각할 확률", fontsize=11, color=INK)
    axL.set_title("투키는 상관을 알고 쓴다 — 홀름은 묻지 않는다",
                  fontsize=13, color=INK, pad=10)
    axL.legend(fontsize=10, loc="lower left", frameon=False)

    # ---------- 오른쪽 : 그 대가는 쌍이 적을수록 작다 ----------
    # 두 검정력을 겹쳐 그리면 축을 좁게 잡아야 보이고, 그러면 1 퍼센트포인트
    # 차이가 거대해 보인다. 차이 자체를 0 부터 그려 크기를 정직하게 보인다.
    gap = np.array(pw_t) - np.array(pw_h)
    axR.bar(ks, gap, width=0.52, color=AMBER, alpha=0.75, zorder=2)
    for k, g in zip(ks, gap):
        axR.text(k, g + 0.0012, f"{g:.3f}", ha="center", va="bottom",
                 fontsize=10, color=AMBER)
    axR.axhline(0, color=MUTED, lw=1)
    axR.text(3, 0.0455, f"k = 3 에서는 투키 {pw_t[0]:.3f} 대 홀름 {pw_h[0]:.3f} —\n"
                        f"쌍이 셋뿐이면 홀름이 잃는 것이 1.4 퍼센트포인트다.\n"
                        f"k = 8 ({KS[-1]*(KS[-1]-1)//2} 쌍) 에서도 "
                        f"{pw_t[-1]:.3f} 대 {pw_h[-1]:.3f} 로 {gap[-1]:.3f} 다.",
             fontsize=10, color=INK, va="top", ha="left")
    axR.set_ylim(0, 0.050)
    axR.set_ylabel("투키의 검정력 빼기 홀름의 검정력", fontsize=11, color=INK)
    axR.set_title("쌍이 적으면 잃는 것이 거의 없다", fontsize=13, color=INK, pad=10)

    for ax in (axL, axR):
        ax.set_xticks(ks)
        ax.set_xlabel("집단 수 k   (쌍은 k(k-1)/2 개)", fontsize=11, color=INK)
        ax.tick_params(colors=GREY, labelsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)

    fig.tight_layout()
    fig.savefig(OUT + "holm_vs_tukey.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("\n저장: " + OUT + "holm_vs_tukey.png")


if __name__ == "__main__":
    main()
