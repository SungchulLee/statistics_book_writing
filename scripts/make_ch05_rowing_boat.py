r"""5.9절 「그림으로 보는 통계학」 — 날씨를 보러 보낸 배가 먼저 뒤집힌다.

정리 1 은 정규성에 **두 번** 기댄다. 분자와 분모에서 한 번씩이다. 평균을
비교하는 t 는 중심극한정리가 받쳐 주어 모집단이 정규가 아니어도 버티지만,
분산비는 받쳐 줄 것이 없다.

같은 모집단 · 같은 n 에서 명목 5% 의 실제 수준을 센다.

  F 검정 (분산비)   첨도가 커질수록 5% 를 한참 넘어선다
  t 검정 (평균차)   첨도와 거의 무관하게 5% 근처에 머문다

따라서 「t 를 쓰기 전에 분산이 같은지 F 로 먼저 보자」는 절차는, 더 튼튼한
검정을 지키겠다고 더 약한 검정을 앞세우는 셈이 된다.

모집단은 모두 평균 0 · 분산 1 로 맞춘다(수준 비교이므로 척도는 무관하지만
그림에서 첨도만 변수로 남기기 위해서다).

만드는 파일:

  ch05/applications/img/rowing_boat_level.png

실행:  python3 scripts/make_ch05_rowing_boat.py   (저장소 최상위에서)
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

OUT = "docs/ch05/applications/img/"

INK = "#37474F"
RED = "#D32F2F"
TEAL = "#00838F"
MUTED = "#90A4AE"

N = 20
REPS = 200_000
SEED = 11
ALPHA = 0.05


def draw(kind, rng, size):
    """평균 0 · 분산 1 로 맞춘 표본."""
    if kind == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size)
    if kind == "normal":
        return rng.normal(0.0, 1.0, size)
    if kind == "laplace":
        return rng.laplace(0.0, 1 / np.sqrt(2), size)
    df = int(kind[1:])                       # t7, t5 …
    return rng.standard_t(df, size) / np.sqrt(df / (df - 2))


# t 분포의 초과첨도는 6/(df-4) 이다. t(5) 는 초과첨도 6 이지만 4 차적률이
# 간신히 유한해서 n = 20 에서 들쭉날쭉하므로(재 보면 t(7)·라플라스 사이로
# 내려앉는다) 눈금을 정직하게 두려고 뺐다.
POPS = [("uniform", "균등", -1.2), ("normal", "정규", 0.0),
        ("t10", "t(10)", 1.0), ("t7", "t(7)", 2.0),
        ("laplace", "라플라스", 3.0)]


def levels(kind):
    rng = np.random.default_rng(SEED)
    x = draw(kind, rng, (REPS, N))
    y = draw(kind, rng, (REPS, N))

    s1, s2 = x.var(axis=1, ddof=1), y.var(axis=1, ddof=1)
    d = N - 1
    lo, hi = stats.f.ppf(ALPHA / 2, d, d), stats.f.ppf(1 - ALPHA / 2, d, d)
    f_rej = np.mean((s1 / s2 < lo) | (s1 / s2 > hi))

    sp = np.sqrt((s1 + s2) / 2)
    t = (x.mean(axis=1) - y.mean(axis=1)) / (sp * np.sqrt(2 / N))
    t_rej = np.mean(np.abs(t) > stats.t.ppf(1 - ALPHA / 2, 2 * N - 2))
    return f_rej * 100, t_rej * 100


def main():
    ks, fs, ts = [], [], []
    for kind, _, ex in POPS:
        a, b = levels(kind)
        ks.append(ex)
        fs.append(a)
        ts.append(b)
        print(f"    {kind:8} 초과첨도 {ex:+.1f}   F {a:5.2f}%   t {b:5.2f}%")

    fig, ax = plt.subplots(figsize=(10.4, 5.4), constrained_layout=True)

    ax.axhline(5, color=MUTED, lw=1.8, ls="--", zorder=2)
    ax.text(3.75, 5.6, "약속한 5%", ha="right", fontsize=10.5, color=MUTED)

    ax.plot(ks, fs, marker="o", ms=10, lw=2.8, color=RED, zorder=5,
            label="F 검정 — 분산이 같은지 먼저 보는 쪽")
    ax.plot(ks, ts, marker="s", ms=9, lw=2.8, color=TEAL, zorder=4,
            label="t 검정 — 지켜 주려던 쪽")

    for ex, (_, nm, _) in zip(ks, POPS):
        ax.text(ex, -1.4, nm, ha="center", fontsize=10.4, color=INK)

    ax.annotate(f"{fs[-1]:.1f}% — 스무 번에 한 번이라더니\n다섯 번에 한 번꼴이다",
                xy=(ks[-1], fs[-1]), xytext=(ks[-1] - 0.25, fs[-1] - 1.2),
                fontsize=10.6, color=RED, ha="right")
    ax.annotate(f"{fs[0]:.2f}% — 꼬리가 가벼우면 반대쪽으로 틀린다",
                xy=(ks[0], fs[0]), xytext=(ks[0] + 0.18, fs[0] + 1.6),
                fontsize=10.4, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))

    ax.set_xlabel("모집단의 초과첨도 — 꼬리가 무거울수록 오른쪽", fontsize=11.5)
    ax.set_ylabel("명목 5% 검정이 실제로 기각하는 비율 (%)", fontsize=11.5)
    ax.set_title(f"두 모분산이 정말 같을 때 잰 것  (n = {N} 씩, {REPS//1000} 천 번)",
                 fontsize=13.5, color=INK, pad=10)
    ax.set_xlim(-1.9, 3.8)
    ax.set_ylim(-2.6, max(fs) + 3)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.6, loc="upper left", framealpha=0.93)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, -0.045,
             "두 집단이 같은 모집단에서 나왔으므로 두 검정 모두 5% 를 지켜야 한다. "
             "평균 쪽은 지키고 분산 쪽은 무너진다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "rowing_boat_level.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
