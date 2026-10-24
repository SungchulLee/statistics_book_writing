r"""11.7절 「그림으로 보는 통계학」 — 절단점 하나가 $10$ 만큼 움직였을 때.

계단함수는 연속인 $x$ 를 구간으로 잘라 구간마다 상수 하나를 준다. 그러니 절단점을
어디에 두느냐가 누가 어느 칸에 드는지를 정한다. $2017$ 년에 미국심장학회·심장협회가
고혈압의 절단점을 수축기 $140$ 에서 $130$, 이완기 $90$ 에서 $80$ 으로 옮겼다.
먼트너 외가 같은 NHANES 자료를 두 지침으로 각각 세어 본 수가 이 그림이다.

  유병률        $31.9\%$  ->  $45.6\%$   ($+13.7$ 퍼센트포인트)
  약물 권고     $34.3\%$  ->  $36.2\%$   ($+1.9$ 퍼센트포인트)

**자료는 한 줄도 바뀌지 않았다.** 바뀐 것은 절단점뿐이다. 그리고 라벨은 크게
움직였는데 처방 권고는 거의 그대로다.

그리는 것:

  ① 두 지침의 라벨 함수 — 수축기 혈압에 대한 계단함수 둘. 뛰는 자리가 $140$ 과
     $130$ 으로 다르고, 그 사이 $10$ mmHg 띠에 든 사람들이 라벨이 갈린다
  ② 그 절단점이 바꾼 것 — 유병률과 약물 권고를 $95\%$ 신뢰구간과 함께 나란히

만드는 파일:

  ch11/splines_gams/img/one_cut_moved.png

실행:  python3 scripts/make_ch13_one_cut_moved.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/splines_gams/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

# 먼트너 외가 보고한 수 (퍼센트, 95% 신뢰구간)
PREV = {"JNC7": (31.9, 30.1, 33.7), "2017": (45.6, 43.6, 47.6)}
MEDS = {"JNC7": (34.3, 32.5, 36.2), "2017": (36.2, 34.2, 38.2)}
CUT = {"JNC7": 140.0, "2017": 130.0}


def main():
    d_prev = PREV["2017"][0] - PREV["JNC7"][0]
    d_meds = MEDS["2017"][0] - MEDS["JNC7"][0]
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print("            JNC7            2017 ACC/AHA       차이")
    print(f"유병률     {PREV['JNC7'][0]:5.1f} "
          f"({PREV['JNC7'][1]:.1f}, {PREV['JNC7'][2]:.1f})   "
          f"{PREV['2017'][0]:5.1f} ({PREV['2017'][1]:.1f}, {PREV['2017'][2]:.1f})   "
          f"{d_prev:+5.1f} pp")
    print(f"약물 권고  {MEDS['JNC7'][0]:5.1f} "
          f"({MEDS['JNC7'][1]:.1f}, {MEDS['JNC7'][2]:.1f})   "
          f"{MEDS['2017'][0]:5.1f} ({MEDS['2017'][1]:.1f}, {MEDS['2017'][2]:.1f})   "
          f"{d_meds:+5.1f} pp")
    print(f"\n라벨은 {d_prev / d_meds:.1f} 배 더 움직였다 "
          f"({d_prev:+.1f} pp 대 {d_meds:+.1f} pp)")
    # 보고된 103.3 백만 명이 45.6% 라는 데서 성인 수를 되짚고, 13.7 pp 를 곱한다
    adults = 103.3 / (PREV["2017"][0] / 100)
    print(f"미국 성인 {adults:.0f} 백만 명으로 치면 "
          f"{adults * d_prev / 100:.1f} 백만 명이 칸을 옮겼다")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), constrained_layout=True)

    # ── ① 두 지침의 라벨 함수 ──
    ax = axes[0]
    x = np.linspace(105, 165, 1200)
    ax.axvspan(130, 140, color=AMBER, alpha=0.16)
    for name, col, off, lab in [("JNC7", BLUE, 0.0, "옛 기준 — $140$ 에서 뛴다"),
                                ("2017", RED, 0.06, "$2017$ 년 기준 — $130$ 에서 뛴다")]:
        c = CUT[name]
        ax.step(x, (x >= c).astype(float) + off, where="post", lw=3.0, color=col,
                label=lab)
        ax.plot([c], [off], "o", ms=9, mfc="white", mec=col, mew=2.2, zorder=4)
    ax.text(135, 0.47, "이 $10$ mmHg 띠에 든 사람은\n한 기준에서는 환자가 아니고\n"
                       "다른 기준에서는 환자다",
            ha="center", va="center", fontsize=11.2, color=AMBER,
            fontweight="bold")
    ax.set_xlim(105, 165)
    ax.set_ylim(-0.18, 1.32)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["정상", "고혈압"], fontsize=11.5)
    ax.set_xticks([110, 120, 130, 140, 150, 160])
    ax.set_xlabel("수축기 혈압 (mmHg)", fontsize=11.5)
    ax.set_title("① 같은 혈압, 다른 라벨", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="x")
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 그 절단점이 바꾼 것 ──
    ax = axes[1]
    xs = np.array([0.0, 1.0])
    w = 0.33
    for i, (name, col, lab) in enumerate([("JNC7", BLUE, "옛 기준 (JNC7)"),
                                          ("2017", RED, "$2017$ 년 기준")]):
        vals = np.array([PREV[name][0], MEDS[name][0]])
        lo = vals - np.array([PREV[name][1], MEDS[name][1]])
        hi = np.array([PREV[name][2], MEDS[name][2]]) - vals
        pos = xs + (i - 0.5) * w
        ax.bar(pos, vals, width=w * 0.92, color=col, alpha=0.82, label=lab)
        ax.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor=INK,
                    elinewidth=1.6, capsize=5)
        for p, v in zip(pos, vals):     # 막대 안에 적는다 — 위에 두면 쪽지와 겹친다
            ax.text(p, v - 3.4, f"${v:.1f}$", ha="center", fontsize=11.5,
                    color="white", fontweight="bold")

    # 막대 위에 띄운다 — 막대 안에 두면 가려진다
    for xc, yc, d, col in [(0.0, 50.8, d_prev, AMBER), (1.0, 41.2, d_meds, TEAL)]:
        ax.text(xc, yc, f"${d:+.1f}$ 퍼센트포인트", ha="center", fontsize=12.5,
                color=col, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=col,
                          alpha=0.95))
    ax.set_xticks(xs)
    ax.set_xticklabels(["고혈압이라 불린 비율", "약을 권고받은 비율"], fontsize=11.8)
    ax.set_ylabel("미국 성인 가운데 ($\\%$)", fontsize=11.5)
    ax.set_ylim(0, 58)
    ax.set_title("② 라벨은 크게, 처방은 조금", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("절단점 하나를 $10$ 만큼 옮기자 삼천만 명이 칸을 옮겼다",
                 fontsize=14.5)
    fig.text(0.5, -0.05,
             "두 막대 모두 같은 NHANES 자료를 두 지침으로 각각 센 것이다. "
             "수염은 $95\\%$ 신뢰구간이고, 수는 모두 먼트너 외가 보고한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "one_cut_moved.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
