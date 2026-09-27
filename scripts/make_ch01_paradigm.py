r"""1장 1.5(두 패러다임 대비)와 1.6(현대적 주제)의 그림들을 생성한다.

1장은 "자료가 어떻게 만들어지는가"를 다루는 개념 장이므로, 여기서 만드는
그림도 모식도이거나 아주 작은 모의실험이다. 1장에 이미 있는 그림들
(표본평균의 표집분포, 출판편향 깔때기, 폭격기 생존자편향, 복잡도에 따른
훈련·시험오차, 로봇청소기 지도)과 주제가 겹치지 않게 골랐다.

만드는 파일:

  ch01/contrast/img/design_data_stages.png
      설계 → 수집 → 분석의 각 단계에서 무엇을 살 수 있고 무엇이
      되돌릴 수 없는지를 그린 모식도 (모의실험 아님)

  ch01/contrast/img/analyze_available_effective_n.png
      이미 있는 자료로 시작할 때의 대가. 왼쪽: 신뢰구간이 좁아지면서
      참값에서 멀어진다. 오른쪽: 편향이 있으면 유효 표본 크기가
      자료 크기와 무관하게 포화한다.

  ch01/modern/img/models_vs_algorithms_extrapolation.png
      같은 자료에 통계 모형(직선)과 학습 알고리즘(랜덤 포레스트)을
      얹었을 때, 훈련 구간 밖에서 무엇이 달라지는가.

실행:  python3 scripts/make_ch01_paradigm.py   (저장소 최상위에서)
필요:  numpy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.

주의:  한글은 수식 $...$ 바깥에만 쓴다. mathtext 에는 한글 글리프가 없다.
       로그축 눈금도 mathtext 를 거치므로 평문 라벨로 바꿔 둔다.
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from sklearn.ensemble import RandomForestRegressor

# === 공통 설정 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

CONTRAST = "docs/ch01/contrast/img/"
MODERN = "docs/ch01/modern/img/"


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# =====================================================================
# 그림 1. 단계별로 무엇을 살 수 있는가 (design_data.md)
# =====================================================================

def fig_design_stages(path=CONTRAST + "design_data_stages.png"):
    fig, ax = plt.subplots(figsize=(10.6, 6.5))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.9)
    ax.axis("off")

    # --- 위쪽: 다섯 단계 상자 ---
    stages = [("질문", 0.25), ("설계", 2.05), ("수집", 3.85),
              ("분석", 6.95), ("결론", 8.75)]
    w, h, y = 1.45, 0.75, 5.30
    for name, x0 in stages:
        left = name in ("질문", "설계", "수집")
        ax.add_patch(FancyBboxPatch(
            (x0, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=GREEN_F if left else BLUE_F,
            edgecolor=GREEN if left else BLUE, linewidth=1.6))
        ax.text(x0 + w / 2, y + h / 2, name, ha="center", va="center",
                fontsize=14, color=INK)

    for a, b in [(1.70, 2.05), (3.50, 3.85), (5.30, 6.95), (8.40, 8.75)]:
        ax.add_patch(FancyArrowPatch(
            (a + 0.04, y + h / 2), (b - 0.04, y + h / 2),
            arrowstyle="-|>", mutation_scale=15,
            linewidth=1.6, color=INK, shrinkA=0, shrinkB=0))

    # --- 되돌릴 수 없는 벽 ---
    ax.plot([6.05, 6.05], [2.45, 6.18], color=RED, linewidth=2.2,
            linestyle=(0, (5, 3)))
    ax.text(6.05, 6.30, "되돌릴 수 없는 지점", ha="center", va="bottom",
            fontsize=11.5, color=RED)

    # --- 왼쪽 패널: 설계 단계에서만 살 수 있는 것 ---
    ax.add_patch(FancyBboxPatch(
        (0.25, 2.45), 5.35, 2.45,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor="#F3F8EF", edgecolor=GREEN, linewidth=1.5))
    ax.text(0.50, 4.68, "설계 단계에서만 살 수 있는 것",
            fontsize=12, color=GREEN, va="top")
    bought = [("무작위 배정", "인과를 말할 자격"),
              ("무작위 선택", "모집단으로 일반화"),
              ("짝짓기·층화", "같은 표본에 더 큰 검정력"),
              ("사전등록", "해석 가능한 $p$ 값"),
              ("검정력 계산", "필요한 표본 크기")]
    for k, (item, gain) in enumerate(bought):
        yy = 4.22 - 0.40 * k
        ax.text(0.55, yy, "·  " + item, fontsize=10.5, color=INK, va="center")
        ax.text(2.55, yy, "→", fontsize=10.5, color=MUTED, va="center")
        ax.text(2.90, yy, gain, fontsize=10.5, color=GREEN, va="center")

    # --- 오른쪽 패널: 분석 단계에서 되돌릴 수 있는 것 ---
    ax.add_patch(FancyBboxPatch(
        (6.50, 2.45), 3.70, 2.45,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor="#F2F7FC", edgecolor=BLUE, linewidth=1.5))
    ax.text(6.72, 4.68, "분석 단계에서 되돌릴 수 있는 것",
            fontsize=12, color=BLUE, va="top")
    for k, item in enumerate(["모형 명세", "변수 변환", "공변량 보정",
                              "다중성 보정", "추정량 교체"]):
        ax.text(6.80, 4.22 - 0.40 * k, "·  " + item,
                fontsize=10.5, color=INK, va="center")
        ax.text(8.55, 4.22 - 0.40 * k, "→", fontsize=10.5, color=MUTED,
                va="center")
        ax.text(8.90, 4.22 - 0.40 * k, "다시 하면 된다", fontsize=10.5,
                color=BLUE, va="center")

    # --- 아래: 분석이 고치지 못하는 것 ---
    ax.add_patch(FancyBboxPatch(
        (0.25, 0.35), 9.95, 1.72,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor="#FDF3F3", edgecolor=RED, linewidth=1.5))
    ax.text(5.22, 1.87, "수집에서 결정되어 분석이 고쳐 주지 못하는 것 — GIGO",
            fontsize=12, color=RED, ha="center", va="center")
    lost = [("뽑히지 않은 사람", "나중에 뽑을 수 없다"),
            ("묻지 않은 질문", "나중에 답을 얻을 수 없다"),
            ("기록하지 않은 변수", "나중에 보정할 수 없다")]
    for k, (item, why) in enumerate(lost):
        x0 = 0.55 + 3.20 * k
        ax.add_patch(FancyBboxPatch(
            (x0, 0.58), 3.00, 0.95,
            boxstyle="round,pad=0,rounding_size=0.10",
            facecolor="white", edgecolor=RED, linewidth=1.0))
        ax.text(x0 + 1.50, 1.28, "×  " + item, fontsize=11, color=RED,
                ha="center", va="center")
        ax.text(x0 + 1.50, 0.86, why, fontsize=10, color=INK,
                ha="center", va="center")

    save(fig, path)


# =====================================================================
# 그림 2. 이미 있는 자료로 시작할 때의 대가 (analyze_available.md)
# =====================================================================

def fig_available_cost(path=CONTRAST + "analyze_available_effective_n.png"):
    rng = np.random.default_rng(7)
    p_true, p_found = 0.50, 0.53         # 선택 기제가 만드는 3 퍼센트포인트 편향
    var = p_true * (1 - p_true)          # 0.25
    B = 4000
    ns = np.array([30, 100, 300, 1_000, 3_000, 10_000,
                   30_000, 100_000, 300_000, 1_000_000])

    rows = []
    for n in ns:
        out = {}
        for tag, p in (("found", p_found), ("srs", p_true)):
            ph = rng.binomial(n, p, B) / n
            se = np.sqrt(ph * (1 - ph) / n)
            lo, hi = ph - 1.96 * se, ph + 1.96 * se
            out[tag] = (ph.mean(), lo.mean(), hi.mean(),
                        float(((lo <= p_true) & (p_true <= hi)).mean()))
        rows.append(out)

    print("\n[그림 2] 가진 자료(편향 0.03) 대 무작위 표본")
    print(f"{'n':>9}{'추정값':>10}{'95% 구간':>22}{'포함률':>9}"
          f"{'무작위 포함률':>14}")
    for n, r in zip(ns, rows):
        m, lo, hi, cov = r["found"]
        print(f"{n:>9}{m:>10.4f}   [{lo:.4f}, {hi:.4f}]{cov:>11.3f}"
              f"{r['srs'][3]:>14.3f}")

    bias = p_found - p_true
    n_eff_cap = var / bias ** 2
    print(f"편향 {bias:.2f} 의 유효 표본 크기 상한 = {n_eff_cap:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

    # --- (a) 구간이 좁아지면서 참값에서 멀어진다 ---
    ax = axes[0]
    found = np.array([[r["found"][0], r["found"][1], r["found"][2]]
                      for r in rows])
    srs = np.array([[r["srs"][0], r["srs"][1], r["srs"][2]] for r in rows])
    ax.fill_between(ns, found[:, 1], found[:, 2], color=ORANGE_F, alpha=0.9)
    ax.plot(ns, found[:, 0], "-", color=ORANGE, linewidth=2.0)
    ax.fill_between(ns, srs[:, 1], srs[:, 2], color=BLUE_F, alpha=0.75)
    ax.plot(ns, srs[:, 0], "-", color=BLUE, linewidth=1.6)
    ax.axhline(p_true, color=INK, linestyle=":", linewidth=1.4)

    for n, r in zip(ns, rows):
        if n in (1_000, 3_000, 10_000):
            ax.annotate(f"{r['found'][3] * 100:.0f}%",
                        xy=(n, r["found"][2]), xytext=(n, 0.5775),
                        fontsize=10, color=ORANGE, ha="center", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=ORANGE,
                                        linewidth=0.9, shrinkA=2, shrinkB=1))
    ax.text(2.3e4, 0.5775, "←  참값을 담은 비율", fontsize=9.5, color=ORANGE,
            ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(650, 1.5e6)
    ax.set_ylim(0.449, 0.593)
    ax.set_xticks([1e3, 1e4, 1e5, 1e6])
    ax.set_xticklabels(["1천", "1만", "10만", "100만"])
    ax.set_xlabel("가진 자료의 크기")
    ax.set_ylabel("추정한 비율")
    ax.set_title("(a) 구간은 좁아지고, 참값은 밖으로 나간다", fontsize=11)
    ax.legend(handles=[
        Patch(facecolor=ORANGE_F, edgecolor=ORANGE,
              label="가진 자료 — 추정값과 95% 구간"),
        Patch(facecolor=BLUE_F, edgecolor=BLUE,
              label="무작위 표본 — 추정값과 95% 구간"),
        Line2D([], [], color=INK, linestyle=":", linewidth=1.4,
               label="참값 0.50")],
        fontsize=8.5, loc="lower right", framealpha=0.97)

    # --- (b) 유효 표본 크기의 포화 ---
    ax = axes[1]
    grid = np.logspace(1, 7, 400)
    ax.plot(grid, grid, linestyle="--", color=INK, linewidth=1.4,
            label="편향이 없을 때")
    for b, color in ((0.001, GREEN), (0.01, PURPLE), (0.03, ORANGE)):
        cap = var / b ** 2
        ax.plot(grid, grid / (1 + grid * b ** 2 / var), color=color,
                linewidth=2.0, label=f"편향 {b:g}")
        ax.axhline(cap, color=color, linestyle=":", linewidth=1.0)
        ax.text(1.3e7, cap, f"{cap:,.0f}", fontsize=9.5, color=color,
                ha="left", va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 1e7)
    ax.set_ylim(5, 5e6)
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(["10", "100", "1천", "1만", "10만", "100만", "1천만"])
    ax.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ax.set_yticklabels(["10", "100", "1천", "1만", "10만", "100만"])
    ax.set_xlabel("가진 자료의 크기")
    ax.set_ylabel("같은 오차를 내는 무작위 표본의 크기")
    ax.set_title("(b) 편향이 있으면 유효 표본 크기가 멈춘다", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)

    fig.tight_layout()
    save(fig, path)


# =====================================================================
# 그림 3. 모형과 알고리즘은 훈련 구간 밖에서 갈린다 (models_vs_algorithms.md)
# =====================================================================

def fig_models_vs_algorithms(
        path=MODERN + "models_vs_algorithms_extrapolation.png"):
    rng = np.random.default_rng(11)
    n, sigma = 150, 1.0
    x_lo, x_hi, x_end = 0.0, 6.0, 12.0

    cases = [("(a) 참 관계가 직선이면", lambda z: 1.0 + 1.6 * z),
             ("(b) 참 관계가 휘면", lambda z: 6.0 * np.log1p(z))]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    summary = []

    for ax, (title, f) in zip(axes, cases):
        x = np.sort(rng.uniform(x_lo, x_hi, n))
        y = f(x) + rng.normal(0, sigma, n)

        # 통계 모형: 계수 두 개짜리 직선
        X = np.column_stack([np.ones(n), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        s2 = resid @ resid / (n - 2)
        se_slope = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])

        # 학습 알고리즘: 랜덤 포레스트
        forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                       random_state=0).fit(x[:, None], y)

        grid = np.linspace(x_lo, x_end, 700)
        lin_hat = beta[0] + beta[1] * grid
        rf_hat = forest.predict(grid[:, None])
        truth = f(grid)

        inside = grid <= x_hi
        rmse = lambda a, m: float(np.sqrt(np.mean((a[m] - truth[m]) ** 2)))
        summary.append(dict(
            title=title, slope=beta[1], se=se_slope,
            lin_in=rmse(lin_hat, inside), rf_in=rmse(rf_hat, inside),
            lin_out=rmse(lin_hat, ~inside), rf_out=rmse(rf_hat, ~inside)))

        ax.axvspan(x_lo, x_hi, color="#ECEFF1", alpha=0.85, zorder=0)
        ax.axvspan(x_hi, x_end, color="#FDF3F3", alpha=0.85, zorder=0)
        ax.scatter(x, y, s=14, color=MUTED, alpha=0.8, zorder=2,
                   label="훈련 자료")
        ax.plot(grid, truth, color=GREEN, linewidth=3.4, zorder=3,
                label="참 관계")
        ax.plot(grid, lin_hat, color=BLUE, linewidth=1.8, zorder=4,
                label="통계 모형 (직선)")
        ax.plot(grid, rf_hat, color=ORANGE, linewidth=2.0, zorder=4,
                label="학습 알고리즘 (포레스트)")
        ax.axvline(x_hi, color=RED, linestyle="--", linewidth=1.2, zorder=5)

        ax.set_xlim(x_lo, x_end)
        ax.set_ylim(-3.5, 27)
        ax.text(3.0, 25.6, "훈련 구간", fontsize=10, color=INK, ha="center",
                va="top")
        ax.text(9.0, 25.6, "자료가 없는 구간", fontsize=10, color=RED,
                ha="center", va="top")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)

        s = summary[-1]
        ax.text(0.25, 21.6,
                f"훈련 구간 RMSE   직선 {s['lin_in']:.2f}"
                f" / 포레스트 {s['rf_in']:.2f}\n"
                f"자료 없는 구간   직선 {s['lin_out']:.2f}"
                f" / 포레스트 {s['rf_out']:.2f}",
                fontsize=9.5, color=INK, va="top", linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=MUTED, linewidth=0.9))

    print("\n[그림 3] 같은 자료, 두 가지 적합")
    for s in summary:
        lo, hi = s["slope"] - 1.96 * s["se"], s["slope"] + 1.96 * s["se"]
        print(f"{s['title']}")
        print(f"  직선의 기울기 {s['slope']:.4f}  95% 구간 "
              f"[{lo:.4f}, {hi:.4f}]")
        print(f"  훈련 구간 RMSE  직선 {s['lin_in']:.4f}"
              f"  포레스트 {s['rf_in']:.4f}")
        print(f"  자료 없는 구간  직선 {s['lin_out']:.4f}"
              f"  포레스트 {s['rf_out']:.4f}")

    fig.tight_layout()
    save(fig, path)


if __name__ == "__main__":
    fig_design_stages()
    fig_available_cost()
    fig_models_vs_algorithms()
