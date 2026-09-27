r"""1.4절(편향) 세 쪽에 들어가는 그림을 생성한다.

세 그림 모두 "자료가 어떻게 만들어지는가"를 눈으로 보여 주는 것이 목적이다.
모식도와 작은 모의실험을 섞어 쓰며, 본문에 인용된 수치는 모두 이 스크립트가
실제로 출력한 값이다(실행하면 콘솔에 다시 찍힌다).

만드는 파일:

  ch01/classical/img/nonresponse_bias.png     무응답 편향은 표본을 키워도 줄지 않는다
  ch01/classical/img/length_bias_geometry.png 길이 편향의 기하 — 긴 구간이 더 많이 뽑힌다
  ch01/classical/img/feedback_loop.png        예측이 자료를 만드는 고리와 그 궤적

실행:  python3 scripts/make_ch01_bias.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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

OUT = "docs/ch01/classical/img/"


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# =====================================================================
# 그림 1. 무응답 편향 — 표본을 키워도 줄지 않는다
# =====================================================================
def fig_nonresponse_bias():
    """왼쪽: 편향 = (1-r) x 응답자·무응답자 차이.  오른쪽: n 을 키워도 그대로."""
    print("[1] nonresponse_bias")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    # --- 왼쪽: 편향의 공식 ---
    r = np.linspace(0.05, 1.0, 200)
    for delta, color, lw in ((20, RED, 2.2), (10, BLUE, 2.6), (5, GREEN, 2.2)):
        axL.plot(100 * r, (1 - r) * delta, color=color, linewidth=lw,
                 label=f"차이 {delta}%p", zorder=3)

    # 오늘날 전화조사가 놓인 자리
    axL.axvspan(5, 10, color=ORANGE_F, alpha=0.55, zorder=1)
    axL.annotate("오늘날 전화조사 — 응답률 10% 이하",
                 xy=(10.2, 20.8), xytext=(13.5, 20.8), fontsize=9.5,
                 color=ORANGE, ha="left", va="center", zorder=5,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.1,
                                 shrinkA=0, shrinkB=1))

    # 본문에서 쓰는 기준점
    axL.plot([25], [7.5], "o", color=BLUE, markersize=8,
             markeredgecolor="white", markeredgewidth=1.2, zorder=6)
    axL.annotate("응답률 25%, 차이 10%p\n→ 편향 7.5%p",
                 xy=(25, 7.5), xytext=(46, 12.6), fontsize=9.5, color=INK,
                 ha="left", va="center", zorder=6,
                 arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.1,
                                 shrinkA=0, shrinkB=4))

    axL.set_xlim(0, 100)
    axL.set_ylim(0, 22)
    axL.set_yticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
    axL.set_xlabel("응답률 (%)", fontsize=10.5, color=INK)
    axL.set_ylabel("추정값의 편향 (%p)", fontsize=10.5, color=INK)
    axL.set_title("편향 = (1 - 응답률) x 응답자·무응답자의 차이",
                  fontsize=11.5, color=INK, pad=9)
    axL.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    axL.grid(alpha=0.25, linewidth=0.7)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # --- 오른쪽: 표본을 키워도 편향은 그대로 ---
    rng = np.random.default_rng(11)
    p_true, resp_rate, delta = 0.50, 0.25, 0.10
    p_resp = p_true + (1 - resp_rate) * delta      # 응답자의 참 비율
    p_non = p_true - resp_rate * delta             # 무응답자의 참 비율
    reps = 4000
    ns = [100, 1_000, 10_000, 100_000, 1_000_000]

    print(f"    참 비율 {p_true:.3f}  응답자 {p_resp:.3f}  무응답자 {p_non:.3f}"
          f"  응답률 {resp_rate:.2f}  → 편향 {100 * (p_resp - p_true):.2f}%p")

    lo_hi = []
    for n in ns:
        m = rng.binomial(n, resp_rate, reps)             # 응답한 사람 수
        m = np.maximum(m, 1)
        est = rng.binomial(m, p_resp) / m                # 응답자만의 비율
        lo, hi = np.quantile(est, [0.025, 0.975])
        lo_hi.append((lo, hi))
        print(f"    n = {n:>9,}  응답자 {m.mean():>9.0f}명"
              f"  추정 평균 {100 * est.mean():6.2f}%"
              f"  95% 구간 [{100 * lo:.2f}, {100 * hi:.2f}]"
              f"  폭 {100 * (hi - lo):5.2f}%p")

    x = np.arange(len(ns))
    axR.axhline(100 * p_true, color=GREEN, linewidth=2.0, zorder=2)
    axR.axhline(100 * p_resp, color=RED, linewidth=1.6, linestyle="--", zorder=2)
    axR.text(len(ns) - 0.55, 100 * p_true - 1.1, "참값 50%", fontsize=9.5,
             color=GREEN, ha="right", va="top")
    axR.text(len(ns) - 0.55, 60.6, "응답자만 보면 57.5%",
             fontsize=9.5, color=RED, ha="right", va="center")

    for xi, (lo, hi) in zip(x, lo_hi):
        axR.vlines(xi, 100 * lo, 100 * hi, color=BLUE, linewidth=7,
                   alpha=0.45, zorder=3)
        axR.plot([xi], [100 * (lo + hi) / 2], "o", color=BLUE, markersize=7,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=4)

    axR.annotate("표집오차는 줄어드는데", xy=(2.0, 59.7), xytext=(0.55, 64.3),
                 fontsize=9.5, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.1,
                                 connectionstyle="arc3,rad=-0.18"))
    axR.annotate("편향 7.5%p 는 그대로", xy=(4.0, 53.6), xytext=(2.25, 46.3),
                 fontsize=9.5, color=RED, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.1,
                                 connectionstyle="arc3,rad=0.2"))

    axR.set_xticks(x)
    axR.set_xticklabels([f"{n:,}" for n in ns], fontsize=9)
    axR.set_xlim(-0.6, len(ns) - 0.4)
    axR.set_ylim(43, 66)
    axR.set_xlabel("표본 크기 (조사를 보낸 사람 수)", fontsize=10.5, color=INK)
    axR.set_ylabel("추정된 지지율 (%)", fontsize=10.5, color=INK)
    axR.set_title("표본을 1만 배 키워도 중심은 움직이지 않는다",
                  fontsize=11.5, color=INK, pad=9)
    axR.grid(axis="y", alpha=0.25, linewidth=0.7)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    fig.tight_layout()
    save(fig, "nonresponse_bias.png")


# =====================================================================
# 그림 2. 길이 편향의 기하
# =====================================================================
def fig_length_bias_geometry():
    """위: 시간축 위의 간격과 승객.  아래: 간격 분포와 승객이 겪는 분포."""
    print("[2] length_bias_geometry")

    rng = np.random.default_rng(5)
    mean_gap = 10.0

    # --- 위 패널에 쓸 작은 그림용 표본 ---
    gaps = rng.exponential(mean_gap, 14)
    arrivals = np.concatenate([[0.0], np.cumsum(gaps)])
    horizon = arrivals[-1]
    n_riders = 90
    riders = np.sort(rng.uniform(0, horizon, n_riders))
    idx = np.searchsorted(arrivals, riders) - 1
    counts = np.array([(idx == k).sum() for k in range(len(gaps))])

    longest, shortest = int(np.argmax(gaps)), int(np.argmin(gaps))
    print(f"    윗 그림: 버스 {len(gaps)}대, 관측 구간 {horizon:.1f}분, 승객 {n_riders}명")
    print(f"      가장 긴 간격 {gaps[longest]:.1f}분 → 승객 {counts[longest]}명")
    print(f"      가장 짧은 간격 {gaps[shortest]:.1f}분 → 승객 {counts[shortest]}명")

    fig = plt.figure(figsize=(11.4, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.42)
    axT = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # --- 위: 기하 ---
    bar_base, bar_scale = 0.0, 1.0
    for k, (a, b) in enumerate(zip(arrivals[:-1], arrivals[1:])):
        face = ORANGE_F if k == longest else (GREEN_F if k == shortest else BLUE_F)
        edge = ORANGE if k == longest else (GREEN if k == shortest else BLUE)
        axT.add_patch(Rectangle((a, bar_base), b - a, counts[k] * bar_scale,
                                facecolor=face, edgecolor=edge, linewidth=1.1,
                                zorder=2))
    axT.plot(riders, np.full(n_riders, -1.3), "o", color=INK, markersize=3.4,
             alpha=0.75, zorder=4)
    axT.hlines(-4.0, 0, horizon, color=INK, linewidth=1.4, zorder=3)
    axT.vlines(arrivals, -4.8, -3.2, color=INK, linewidth=1.4, zorder=3)

    axT.text(-2.0, -1.3, "승객", fontsize=9.5, color=INK, ha="right", va="center")
    axT.text(-2.0, -4.0, "버스", fontsize=9.5, color=INK, ha="right", va="center")

    x_long = (arrivals[longest] + arrivals[longest + 1]) / 2
    x_short = (arrivals[shortest] + arrivals[shortest + 1]) / 2
    axT.annotate(f"가장 긴 간격 {gaps[longest]:.0f}분\n→ 승객 {counts[longest]}명",
                 xy=(x_long, counts[longest] + 0.4),
                 xytext=(x_long, counts[longest] + 5.2),
                 fontsize=9.5, color=ORANGE, ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.1))
    axT.annotate(f"가장 짧은 간격 {gaps[shortest]:.1f}분\n→ 승객 {counts[shortest]}명",
                 xy=(x_short, 0.3), xytext=(x_short, 12.0),
                 fontsize=9.5, color=GREEN, ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.1))

    axT.set_xlim(-14, horizon + 2)
    axT.set_ylim(-5.6, max(counts) + 9.0)
    axT.set_yticks([])
    axT.set_xlabel("시간 (분)", fontsize=10.5, color=INK)
    axT.set_ylabel("그 간격에 떨어진 승객 수", fontsize=10.5, color=INK, labelpad=26)
    axT.set_title("승객은 시간축 위에 고르게 떨어진다 — 그래서 넓은 구간이 더 많이 받는다",
                  fontsize=11.5, color=INK, pad=9)
    for s in ("top", "right", "left"):
        axT.spines[s].set_visible(False)

    # --- 아래: 분포 두 개 ---
    big_gaps = rng.exponential(mean_gap, 400_000)
    big_arr = np.cumsum(big_gaps)
    big_riders = rng.uniform(0, big_arr[-1], 400_000)
    seen = big_gaps[np.searchsorted(big_arr, big_riders)]
    wait = big_arr[np.searchsorted(big_arr, big_riders)] - big_riders

    m_gap, m_seen, m_wait = big_gaps.mean(), seen.mean(), wait.mean()
    print(f"    아래 그림: 버스회사가 재는 평균 간격 {m_gap:.2f}분")
    print(f"      승객이 겪는 평균 간격 {m_seen:.2f}분  (이론 20.00)")
    print(f"      평균 대기시간 {m_wait:.2f}분")

    bins = np.linspace(0, 60, 61)
    axB.hist(big_gaps, bins=bins, density=True, color=BLUE_F,
             edgecolor=BLUE, linewidth=0.9, label="버스회사가 세는 간격", zorder=2)
    axB.hist(seen, bins=bins, density=True, histtype="step", color=ORANGE,
             linewidth=2.2, label="승객이 겪는 간격", zorder=3)

    axB.axvline(m_gap, color=BLUE, linewidth=1.8, linestyle="--", zorder=4)
    axB.axvline(m_seen, color=ORANGE, linewidth=1.8, linestyle="--", zorder=4)
    axB.text(m_gap + 0.9, 0.094, f"평균 {m_gap:.1f}분", fontsize=9.5,
             color=BLUE, ha="left", va="center")
    axB.text(m_seen + 0.9, 0.094, f"평균 {m_seen:.1f}분", fontsize=9.5,
             color=ORANGE, ha="left", va="center")
    axB.annotate("", xy=(m_seen, 0.083), xytext=(m_gap, 0.083),
                 arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.2))
    axB.text((m_gap + m_seen) / 2, 0.086, "두 배", fontsize=9.5, color=INK,
             ha="center", va="bottom")

    axB.set_xlim(0, 60)
    axB.set_ylim(0, 0.104)
    axB.set_xlabel("간격의 길이 (분)", fontsize=10.5, color=INK)
    axB.set_ylabel("밀도", fontsize=10.5, color=INK)
    axB.set_title("같은 버스 노선, 세는 단위만 다르다 — 간격 40만 개와 승객 40만 명",
                  fontsize=11.5, color=INK, pad=9)
    axB.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    axB.grid(axis="y", alpha=0.25, linewidth=0.7)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    save(fig, "length_bias_geometry.png")


# =====================================================================
# 그림 3. 피드백 루프
# =====================================================================
TRUE_RATE = np.array([0.10, 0.10])
PATROL_TOTAL = 100
POPULATION = 1000
ROUNDS = 30


def _run(rng, explore, rule):
    """본문 예제 1과 같은 고리를 돌리며 회차별 기록 비중을 남긴다."""
    records = np.array([102.0, 100.0])
    traj = [100 * records[0] / records.sum()]
    for _ in range(ROUNDS):
        if rule == "winner":
            hotspot = np.where(records == records.max(), 1.0, 0.0)
            share = (1 - explore) * hotspot / hotspot.sum() + explore * 0.5
        else:                                     # 기록에 비례해 배분
            share = records / records.sum()
        patrol = PATROL_TOTAL * share
        crimes = rng.binomial(POPULATION, TRUE_RATE)
        caught = rng.binomial(crimes, patrol / PATROL_TOTAL)
        records = records + caught
        traj.append(100 * records[0] / records.sum())
    return np.array(traj)


def _box(ax, x, y, w, h, text, face, edge):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.14",
                                facecolor=face, edgecolor=edge, linewidth=1.6,
                                zorder=3))
    ax.text(x, y, text, fontsize=10, color=INK, ha="center", va="center",
            zorder=4, linespacing=1.45)


def fig_feedback_loop():
    """왼쪽: 고리의 모식도.  오른쪽: 고리를 30번 돌린 궤적."""
    print("[3] feedback_loop")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.2, 4.9),
                                   gridspec_kw={"width_ratios": [1.0, 1.05]})

    # --- 왼쪽: 모식도 ---
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 8.4)
    axL.axis("off")
    axL.set_title("예측이 자료를 만들고, 그 자료가 예측을 키운다",
                  fontsize=11.5, color=INK, pad=9)

    bw, bh = 3.5, 1.25
    nodes = {
        "pred":  (2.8, 6.8, "모형: A구역이 위험하다", BLUE_F, BLUE),
        "act":   (7.0, 4.9, "순찰을 A구역에 집중한다", ORANGE_F, ORANGE),
        "obs":   (7.0, 2.6, "A구역에서만 검거가 기록된다", ORANGE_F, ORANGE),
        "learn": (2.8, 0.8, "그 기록이 내일의 훈련자료", BLUE_F, BLUE),
    }
    for x, y, t, f, e in nodes.values():
        _box(axL, x, y, bw, bh, t, f, e)

    def arrow(p, q, rad, color=INK):
        axL.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15,
                                      color=color, linewidth=1.7, zorder=2,
                                      shrinkA=2, shrinkB=2,
                                      connectionstyle=f"arc3,rad={rad}"))

    arrow((4.55, 6.8), (5.85, 5.35), -0.25)
    arrow((7.0, 4.28), (7.0, 3.23), 0.0)
    arrow((5.25, 2.6), (3.9, 1.32), 0.25)
    arrow((1.05, 0.8), (1.05, 6.8), -0.32)

    axL.text(4.9, 3.85, "고리가 한 바퀴 돌 때마다\n편향이 커진다", fontsize=10,
             color=RED, ha="center", va="center", linespacing=1.5, zorder=5)
    axL.text(1.85, 3.85, "되먹임", fontsize=10, color=INK, ha="center",
             va="center", zorder=5)

    # 고리를 끊는 장치
    axL.add_patch(FancyBboxPatch((6.3, 7.35), 3.3, 0.9,
                                 boxstyle="round,pad=0,rounding_size=0.14",
                                 facecolor=GREEN_F, edgecolor=GREEN,
                                 linewidth=1.6, zorder=3))
    axL.text(7.95, 7.8, "무작위 탐색", fontsize=10, color=GREEN,
             ha="center", va="center", zorder=4)
    axL.add_patch(FancyArrowPatch((7.95, 7.3), (7.55, 5.55),
                                  arrowstyle="-|>", mutation_scale=15,
                                  color=GREEN, linewidth=1.7, linestyle="--",
                                  zorder=2, shrinkA=2, shrinkB=2))

    # --- 오른쪽: 궤적 ---
    rng = np.random.default_rng(7)          # 본문 예제 1과 같은 씨앗·같은 순서
    t_w0 = _run(rng, 0.0, "winner")
    t_w20 = _run(rng, 0.2, "winner")
    t_prop = _run(np.random.default_rng(7), 0.0, "proportional")

    rounds = np.arange(ROUNDS + 1)
    axR.axhline(50, color=GREEN, linewidth=2.0, zorder=2)
    axR.text(ROUNDS - 0.4, 48.8, "참값 50% — 두 구역의 범죄율은 같다",
             fontsize=9.5, color=GREEN, ha="right", va="top")

    axR.plot(rounds, t_w0, color=RED, linewidth=2.4,
             label="상위 구역에 몰아주기, 탐색 0%", zorder=5)
    axR.plot(rounds, t_w20, color=ORANGE, linewidth=2.4,
             label="상위 구역에 몰아주기, 탐색 20%", zorder=4)
    axR.plot(rounds, t_prop, color=BLUE, linewidth=2.4,
             label="기록에 비례해 배분", zorder=3)

    for traj, color in ((t_w0, RED), (t_w20, ORANGE), (t_prop, BLUE)):
        axR.text(ROUNDS + 0.5, traj[-1], f"{traj[-1]:.1f}%", fontsize=10,
                 color=color, ha="left", va="center")

    print(f"    30회차 뒤 A구역 검거 기록 비중")
    print(f"      몰아주기 탐색 0%   {t_w0[-1]:.1f}%")
    print(f"      몰아주기 탐색 20%  {t_w20[-1]:.1f}%")
    print(f"      기록 비례 배분     {t_prop[-1]:.1f}%")

    axR.set_xlim(0, ROUNDS + 4.6)
    axR.set_ylim(44, 102)
    axR.set_xlabel("고리를 돈 횟수", fontsize=10.5, color=INK)
    axR.set_ylabel("A구역이 차지하는 검거 기록 비중 (%)", fontsize=10.5, color=INK)
    axR.set_title("출발점의 기록 차이는 102 대 100 뿐이었다",
                  fontsize=11.5, color=INK, pad=9)
    axR.legend(fontsize=9.5, loc="center right", framealpha=0.95)
    axR.grid(alpha=0.25, linewidth=0.7)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    fig.tight_layout()
    save(fig, "feedback_loop.png")


# =====================================================================
if __name__ == "__main__":
    fig_nonresponse_bias()
    fig_length_bias_geometry()
    fig_feedback_loop()
