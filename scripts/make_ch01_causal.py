r"""1장(자료가 어떻게 만들어지는가)의 인과 관련 그림 세 장을 생성한다.

세 그림은 각각 다음 주장을 떠받친다.

  1. 무작위 배정은 관측된 공변량이든 관측되지 않은 공변량이든 평균적으로
     균형 잡지만, 그 균형은 "평균적으로"일 뿐이며 표본이 작으면 한 번의
     배정에서 큰 불균형이 나올 확률이 높다.
  2. 관찰연구에서는 처리군과 대조군의 공변량 분포가 애초에 어긋나 있고,
     분포가 겹치지 않는 구간에서는 비교할 상대가 없다. 층화 보정은
     겹치는 구간 안에서만 작동한다.
  3. 심슨의 역설 — 1973년 버클리 대학원 입학. 전체에서는 남성 합격률이
     훨씬 높지만 학과별로 보면 방향이 뒤집힌다.

만드는 파일:

  docs/ch01/classical/img/randomization_balance.png      무작위 배정이 하는 일
  docs/ch01/classical/img/observational_overlap.png      공변량 겹침과 층화 보정
  docs/ch01/classical/img/simpsons_paradox_berkeley.png  심슨의 역설

실행:  python3 scripts/make_ch01_causal.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.

본문에 인용할 수치는 모두 stdout으로 함께 인쇄한다.
"""

from math import erf, sqrt

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def phi(x):
    """표준정규 누적분포함수 (scipy 없이)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def style(ax):
    """공통 축 장식: 위/오른쪽 테두리를 없애고 색을 통일한다."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def finish(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ===========================================================================
# 그림 1. 무작위 배정이 하는 일 — 통제실험
# ===========================================================================

def fig_randomization_balance():
    """무작위 배정 vs 자기선택 배정에서 공변량 불균형의 분포.

    왼쪽: 같은 모집단 200명을 (a) 무작위로 반씩 나누는 경우와
          (b) 나이가 많을수록 처리를 받기 쉬운 자기선택 배정의 경우.
          관측된 공변량(나이)과 관측되지 않은 공변량(잠재 위험)을 각각 본다.
    오른쪽: 한 군의 크기 n에 따라 "표준화 불균형이 0.2를 넘을 확률".
    """
    rng = np.random.default_rng(20260928)

    # --- 모집단 200명 ---
    N = 200
    n_half = N // 2
    age = rng.normal(60, 10, N)                      # 관측된 공변량
    zz = rng.normal(0, 1, N)
    # 관측되지 않은 공변량: 나이와 상관 0.5인 잠재 위험 점수
    age_std = (age - age.mean()) / age.std(ddof=1)
    risk = 0.5 * age_std + sqrt(1 - 0.25) * zz

    sd_age = age.std(ddof=1)
    sd_risk = risk.std(ddof=1)

    # 자기선택 배정: 나이가 많을수록 처리를 받을 확률이 높다
    p_self = 1.0 / (1.0 + np.exp(-0.2 * (age - age.mean())))

    B = 20000
    d_age_r = np.empty(B)
    d_risk_r = np.empty(B)
    d_age_s = np.empty(B)
    d_risk_s = np.empty(B)
    for b in range(B):
        perm = rng.permutation(N)
        t, c = perm[:n_half], perm[n_half:]
        d_age_r[b] = (age[t].mean() - age[c].mean()) / sd_age
        d_risk_r[b] = (risk[t].mean() - risk[c].mean()) / sd_risk

        u = rng.random(N) < p_self
        d_age_s[b] = (age[u].mean() - age[~u].mean()) / sd_age
        d_risk_s[b] = (risk[u].mean() - risk[~u].mean()) / sd_risk

    print("\n[그림 1] 무작위 배정 vs 자기선택 배정 (모집단 200명, 한 군 100명)")
    print(f"  무작위   나이(관측)     : 평균 {d_age_r.mean():+.4f}  "
          f"표준편차 {d_age_r.std():.4f}")
    print(f"  무작위   잠재위험(불관측): 평균 {d_risk_r.mean():+.4f}  "
          f"표준편차 {d_risk_r.std():.4f}")
    print(f"  이론값 sqrt(2/100) = {sqrt(2 / 100):.4f}")
    print(f"  자기선택 나이(관측)     : 평균 {d_age_s.mean():+.4f}  "
          f"표준편차 {d_age_s.std():.4f}")
    print(f"  자기선택 잠재위험(불관측): 평균 {d_risk_s.mean():+.4f}  "
          f"표준편차 {d_risk_s.std():.4f}")
    print(f"  무작위 배정에서 |표준화 차이| > 0.2 일 확률: "
          f"{np.mean(np.abs(d_age_r) > 0.2):.4f}")

    # --- 오른쪽 패널용 모의실험: 군 크기별로 큰 불균형이 날 확률 ---
    ns = np.array([10, 25, 50, 100, 200, 400])
    reps, chunk = 20000, 2000
    emp = []
    for m in ns:
        hit = 0
        done = 0
        while done < reps:
            k = min(chunk, reps - done)
            x = rng.normal(0, 1, (k, 2 * m))
            d = (x[:, :m].mean(1) - x[:, m:].mean(1)) / x.std(axis=1, ddof=1)
            hit += int(np.sum(np.abs(d) > 0.2))
            done += k
        emp.append(hit / reps)
    emp = np.array(emp)

    print("  군 크기별 P(|표준화 차이| > 0.2):")
    for m, e in zip(ns, emp):
        theory = 2 * (1 - phi(0.2 * sqrt(m / 2)))
        print(f"    n = {m:>4}:  모의 {e:.4f}   이론 {theory:.4f}")

    # --- 그리기 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3))

    bins = np.linspace(-0.70, 1.50, 89)
    ax1.hist(d_age_r, bins=bins, density=True, color=BLUE_F,
             edgecolor=BLUE, linewidth=1.0,
             label="무작위 배정 · 나이 (관측된 공변량)")
    ax1.hist(d_risk_r, bins=bins, density=True, histtype="step",
             color=GREEN, linewidth=1.9,
             label="무작위 배정 · 잠재 위험 (관측되지 않은 공변량)")
    ax1.hist(d_age_s, bins=bins, density=True, color=ORANGE_F,
             edgecolor=ORANGE, linewidth=1.0, alpha=0.85,
             label="자기선택 배정 · 나이")
    ax1.hist(d_risk_s, bins=bins, density=True, histtype="step",
             color=PURPLE, linewidth=1.9, linestyle="--",
             label="자기선택 배정 · 잠재 위험")

    ax1.axvline(0, color=INK, linewidth=1.1, zorder=0)
    ax1.set_xlim(-0.72, 1.52)
    ymax = ax1.get_ylim()[1]
    ax1.set_ylim(0, ymax * 1.42)
    ax1.annotate("무작위 배정:\n두 공변량 모두 0 주위",
                 xy=(-0.06, 2.9), xycoords="data",
                 xytext=(-0.68, ymax * 1.05), fontsize=9.5, color=BLUE,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.2,
                                 shrinkA=2, shrinkB=3))
    ax1.annotate("자기선택 배정:\n0에서 멀리 치우쳐 있다",
                 xy=(1.19, 5.9), xycoords="data",
                 xytext=(1.44, ymax * 1.10), fontsize=9.5, color=ORANGE,
                 ha="right", va="center",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.2,
                                 shrinkA=2, shrinkB=3))
    ax1.set_xlabel("표준화된 공변량 차이  (처리군 평균 - 대조군 평균, 모집단 표준편차 단위)",
                   fontsize=9.5, color=INK)
    ax1.set_ylabel("밀도", fontsize=10, color=INK)
    ax1.set_yticks([])
    ax1.set_title("(가) 배정 방식이 공변량 균형을 어떻게 바꾸는가",
                  fontsize=11.5, color=INK, pad=9)
    ax1.legend(fontsize=8.2, loc="upper center", frameon=False,
               bbox_to_anchor=(0.50, 0.80))
    style(ax1)

    grid = np.linspace(6, 430, 400)
    theory = np.array([2 * (1 - phi(0.2 * sqrt(m / 2))) for m in grid])
    ax2.plot(grid, 100 * theory, color=INK, linewidth=1.8,
             label="이론값")
    ax2.plot(ns, 100 * emp, "o", color=BLUE, markersize=7,
             markeredgecolor="white", markeredgewidth=1.2, zorder=5,
             label="모의실험 (각 20,000회)")
    for m, e in zip(ns, emp):
        if m in (25, 200):
            ax2.annotate(f"n = {m}\n{100 * e:.0f}%",
                         xy=(m, 100 * e), xytext=(m + 32, 100 * e + 9),
                         fontsize=9.5, color=BLUE, ha="left", va="bottom",
                         arrowprops=dict(arrowstyle="->", color=BLUE,
                                         linewidth=1.1, shrinkA=0, shrinkB=4))
    ax2.set_xlim(0, 440)
    ax2.set_ylim(0, 78)
    ax2.set_xlabel("한 군의 크기 n", fontsize=10, color=INK)
    ax2.set_ylabel("표준화 차이가 0.2를 넘을 확률 (%)", fontsize=10, color=INK,
                   labelpad=9)
    ax2.set_title("(나) 작은 연구는 '운 나쁜 배정'에 취약하다",
                  fontsize=11.5, color=INK, pad=9)
    ax2.legend(fontsize=9, loc="upper right", frameon=False)
    ax2.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.7)
    ax2.set_axisbelow(True)
    style(ax2)

    fig.tight_layout()
    finish(fig, "randomization_balance.png")


# ===========================================================================
# 그림 2. 관찰연구의 공변량 겹침과 층화 보정 — 관찰연구
# ===========================================================================

def fig_observational_overlap():
    """관찰연구에서 두 군의 공변량 분포가 어긋나 있는 모습과 층화 보정.

    처리(신약 복용)를 받을 확률이 나이와 함께 커지고, 나이는 결과(증상 점수)를
    직접 끌어올린다. 참 처리효과는 -5.0 (증상 점수를 5점 낮춘다).
    """
    rng = np.random.default_rng(7)
    N = 4000
    age = np.clip(rng.normal(55, 12, N), 25, 85)
    p_treat = 1.0 / (1.0 + np.exp(-0.15 * (age - 55)))
    T = rng.random(N) < p_treat

    tau = -5.0
    Y = 40 + 0.5 * age + tau * T + rng.normal(0, 6, N)

    naive = Y[T].mean() - Y[~T].mean()

    # --- 층화 보정: 5년 폭 나이 구간 안에서만 비교한다 ---
    edges = np.arange(25, 90, 5.0)
    idx = np.clip(np.digitize(age, edges) - 1, 0, len(edges) - 2)
    num = den = 0.0
    used = skipped = 0
    for k in range(len(edges) - 1):
        m = idx == k
        nt, nc = int((m & T).sum()), int((m & ~T).sum())
        if nt >= 10 and nc >= 10:
            num += m.sum() * (Y[m & T].mean() - Y[m & ~T].mean())
            den += m.sum()
            used += int(m.sum())
        else:
            skipped += int(m.sum())
    strat = num / den

    # --- 겹침 정도: 두 나이 분포의 겹침 계수 ---
    ob = np.arange(25, 86, 2.5)
    ft, _ = np.histogram(age[T], bins=ob, density=True)
    fc, _ = np.histogram(age[~T], bins=ob, density=True)
    overlap = float(np.minimum(ft, fc).sum() * 2.5)

    print("\n[그림 2] 관찰연구의 공변량 겹침 (n = 4,000, 참 효과 -5.0)")
    print(f"  처리군 {int(T.sum()):>4}명  평균 나이 {age[T].mean():.1f}세")
    print(f"  대조군 {int((~T).sum()):>4}명  평균 나이 {age[~T].mean():.1f}세")
    print(f"  평균 나이 차이 {age[T].mean() - age[~T].mean():+.1f}세")
    print(f"  두 나이 분포의 겹침 계수 {overlap:.3f}")
    print(f"  단순 차이     {naive:+.2f}   (부호가 뒤집혔다)")
    print(f"  나이 층화 보정 {strat:+.2f}")
    print(f"  참 효과       {tau:+.2f}")
    print(f"  층화에 쓰인 관측 {used}명, 상대가 없어 버린 관측 {skipped}명")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4),
                             gridspec_kw={"width_ratios": [1.18, 1.0, 1.05]})
    ax1, ax2, ax3 = axes

    # --- (가) 나이 분포 ---
    hb = np.arange(25, 86, 2.5)
    ax1.hist(age[~T], bins=hb, density=True, color=ORANGE_F,
             edgecolor=ORANGE, linewidth=1.0, label="대조군 (약 안 먹음)")
    ax1.hist(age[T], bins=hb, density=True, color=BLUE_F,
             edgecolor=BLUE, linewidth=1.0, alpha=0.78,
             label="처리군 (약 먹음)")
    ax1.axvline(age[~T].mean(), color=ORANGE, linewidth=1.6, linestyle="--")
    ax1.axvline(age[T].mean(), color=BLUE, linewidth=1.6, linestyle="--")
    y1 = ax1.get_ylim()[1]
    ax1.set_ylim(0, y1 * 1.22)
    ax1.annotate("", xy=(age[T].mean(), y1 * 1.06),
                 xytext=(age[~T].mean(), y1 * 1.06),
                 arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.3))
    ax1.text((age[T].mean() + age[~T].mean()) / 2, y1 * 1.10,
             f"{age[T].mean() - age[~T].mean():.1f}세",
             ha="center", va="bottom", fontsize=10, color=INK)
    ax1.set_xlabel("나이 (세)", fontsize=10, color=INK)
    ax1.set_ylabel("밀도", fontsize=10, color=INK)
    ax1.set_yticks([])
    ax1.set_title("(가) 두 군은 애초에 다른 사람들이다",
                  fontsize=11.5, color=INK, pad=9)
    ax1.legend(fontsize=9, loc="upper left", frameon=False)
    style(ax1)

    # --- (나) 층별 인원 ---
    be = np.arange(25, 90, 10.0)
    labels, ct, cc = [], [], []
    for k in range(len(be) - 1):
        m = (age >= be[k]) & (age < be[k + 1])
        labels.append(f"{int(be[k])}–{int(be[k + 1])}")
        ct.append(int((m & T).sum()))
        cc.append(int((m & ~T).sum()))
    xx = np.arange(len(labels))
    ax2.bar(xx - 0.2, cc, width=0.38, color=ORANGE_F, edgecolor=ORANGE,
            linewidth=1.1, label="대조군")
    ax2.bar(xx + 0.2, ct, width=0.38, color=BLUE_F, edgecolor=BLUE,
            linewidth=1.1, label="처리군")
    for i, (a, b) in enumerate(zip(cc, ct)):
        ax2.text(i - 0.2, a + 12, str(a), ha="center", va="bottom",
                 fontsize=8.5, color=ORANGE)
        ax2.text(i + 0.2, b + 12, str(b), ha="center", va="bottom",
                 fontsize=8.5, color=BLUE)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, max(max(ct), max(cc)) * 1.34)
    ax2.set_xlabel("나이 구간 (세)", fontsize=10, color=INK)
    ax2.set_ylabel("인원", fontsize=10, color=INK)
    ax2.set_title("(나) 양 끝에서는 비교할 상대가 없다",
                  fontsize=11.5, color=INK, pad=9)
    ax2.legend(fontsize=9, loc="upper left", frameon=False)
    ax2.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.7)
    ax2.set_axisbelow(True)
    style(ax2)

    # --- (다) 추정값 ---
    names = ["단순 차이\n(보정 안 함)", "나이 층화 보정", "참 처리효과"]
    vals = [naive, strat, tau]
    cols = [RED, GREEN, INK]
    yy = np.arange(3)[::-1]
    ax3.barh(yy, vals, height=0.52, color=cols, alpha=0.82,
             edgecolor=cols, linewidth=1.2)
    ax3.axvline(0, color=INK, linewidth=1.1)
    ax3.axvline(tau, color=MUTED, linewidth=1.3, linestyle="--")
    for y, v, c in zip(yy, vals, cols):
        off = 0.45 if v > 0 else -0.45
        ha = "left" if v > 0 else "right"
        ax3.text(v + off, y, f"{v:+.2f}", ha=ha, va="center",
                 fontsize=10.5, color=c)
    ax3.set_yticks(yy)
    ax3.set_yticklabels(names, fontsize=9.5)
    ax3.set_xlim(-9.4, 6.0)
    ax3.set_ylim(-0.62, 2.92)
    ax3.set_xlabel("추정된 처리효과 (증상 점수 변화)", fontsize=10, color=INK)
    ax3.set_title("(다) 층화는 겹치는 구간 안에서 작동한다",
                  fontsize=11.5, color=INK, pad=9)
    ax3.text(-9.1, 2.72, "◀ 약이 증상을 낮춘다", fontsize=9, color=GREEN,
             ha="left", va="center")
    ax3.text(5.7, 2.72, "약이 증상을 높인다 ▶", fontsize=9, color=RED,
             ha="right", va="center")
    ax3.grid(axis="x", color=MUTED, alpha=0.25, linewidth=0.7)
    ax3.set_axisbelow(True)
    style(ax3)

    fig.tight_layout()
    finish(fig, "observational_overlap.png")


# ===========================================================================
# 그림 3. 심슨의 역설 — 1973년 버클리 대학원 입학
# ===========================================================================

# Bickel, Hammel & O'Connell (1975), Science 187, 398-404.
# 학과, 남성 지원자, 남성 합격, 여성 지원자, 여성 합격
BERKELEY = [
    ("A", 825, 512, 108, 89),
    ("B", 560, 353, 25, 17),
    ("C", 325, 120, 593, 202),
    ("D", 417, 138, 375, 131),
    ("E", 191, 53, 393, 94),
    ("F", 373, 22, 341, 24),
]


def fig_simpsons_paradox_berkeley():
    """전체에서는 남성이 유리해 보이지만 학과별로는 뒤집힌다."""
    dept = [r[0] for r in BERKELEY]
    m_app = np.array([r[1] for r in BERKELEY], float)
    m_adm = np.array([r[2] for r in BERKELEY], float)
    w_app = np.array([r[3] for r in BERKELEY], float)
    w_adm = np.array([r[4] for r in BERKELEY], float)

    m_rate = 100 * m_adm / m_app
    w_rate = 100 * w_adm / w_app
    tot_app = m_app + w_app
    tot_rate = 100 * (m_adm + w_adm) / tot_app
    w_share = 100 * w_app / tot_app

    M = 100 * m_adm.sum() / m_app.sum()
    W = 100 * w_adm.sum() / w_app.sum()
    r = float(np.corrcoef(tot_rate, w_share)[0, 1])

    print("\n[그림 3] 1973년 버클리 대학원 입학 (상위 6개 학과)")
    print(f"  전체 남성 {int(m_adm.sum())}/{int(m_app.sum())} = {M:.1f}%")
    print(f"  전체 여성 {int(w_adm.sum())}/{int(w_app.sum())} = {W:.1f}%")
    print(f"  전체 격차 {M - W:+.1f} 퍼센트포인트 (남성 유리)")
    print("  학과  남성합격률  여성합격률  학과합격률  여성지원비율  지원자수")
    for i, d in enumerate(dept):
        print(f"    {d}   {m_rate[i]:>8.1f}%  {w_rate[i]:>8.1f}%  "
              f"{tot_rate[i]:>8.1f}%  {w_share[i]:>10.1f}%  {int(tot_app[i]):>6}")
    fav = sum(1 for i in range(6) if w_rate[i] > m_rate[i])
    print(f"  여성 합격률이 더 높은 학과 {fav}개 / 6개")
    print(f"  학과 합격률과 여성 지원 비율의 상관 r = {r:+.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})
    xx = np.arange(6)

    # --- (가) 학과별 합격률 대 전체 합격률 ---
    for i in xx:
        ax1.plot([i, i], [m_rate[i], w_rate[i]], color=MUTED, linewidth=1.4,
                 zorder=1)
    ax1.plot([-0.5, 5.3], [M, M], color=BLUE, linewidth=1.5, linestyle="--",
             zorder=0)
    ax1.plot([-0.5, 5.3], [W, W], color=ORANGE, linewidth=1.5, linestyle="--",
             zorder=0)
    ax1.plot(xx, m_rate, "o", markersize=9, color=BLUE, zorder=3,
             markeredgecolor="white", markeredgewidth=1.3, label="남성")
    ax1.plot(xx, w_rate, "s", markersize=9, color=ORANGE, zorder=3,
             markeredgecolor="white", markeredgewidth=1.3, label="여성")

    ax1.text(5.5, M, f"전체 남성 {M:.1f}%", fontsize=9.5, color=BLUE,
             ha="left", va="center")
    ax1.text(5.5, W, f"전체 여성 {W:.1f}%", fontsize=9.5, color=ORANGE,
             ha="left", va="center")
    ax1.text(5.5, (M + W) / 2, f"전체 격차 {M - W:.1f} 포인트",
             fontsize=9, color=INK, ha="left", va="center")
    ax1.set_xlim(-0.55, 8.7)
    ax1.set_ylim(0, 92)
    ax1.set_xticks(xx)
    ax1.set_xticklabels(dept, fontsize=10)
    ax1.set_xlabel("학과", fontsize=10, color=INK)
    ax1.set_ylabel("합격률 (%)", fontsize=10, color=INK)
    ax1.set_title("(가) 전체는 남성이 유리, 학과별로는 뒤집힌다",
                  fontsize=11.5, color=INK, pad=9)
    ax1.legend(fontsize=9.5, loc="lower left", frameon=False)
    ax1.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.7)
    ax1.set_axisbelow(True)
    style(ax1)

    # --- (나) 교란의 기제 ---
    sizes = 520 * tot_app / tot_app.max()
    ax2.scatter(tot_rate, w_share, s=sizes, color=GREEN_F, edgecolor=GREEN,
                linewidth=1.5, zorder=3)
    off = {"A": (22, 13), "B": (-20, -15), "C": (0, 21),
           "D": (24, -7), "E": (0, 21), "F": (0, -21)}
    for i, d in enumerate(dept):
        ax2.annotate(d, (tot_rate[i], w_share[i]),
                     textcoords="offset points", xytext=off[d],
                     ha="center", va="center", fontsize=11, color=GREEN,
                     fontweight="bold", zorder=5)
    b, a = np.polyfit(tot_rate, w_share, 1)
    gx = np.linspace(2, 70, 50)
    ax2.plot(gx, a + b * gx, color=MUTED, linewidth=1.4, linestyle="--",
             zorder=1)
    ax2.set_xlim(0, 74)
    ax2.set_ylim(-14, 90)
    ax2.set_xlabel("학과의 합격률 (%)  — 들어가기 쉬운 정도", fontsize=10,
                   color=INK)
    ax2.set_ylabel("지원자 가운데 여성의 비율 (%)", fontsize=10, color=INK)
    ax2.set_title("(나) 교란의 기제: 여성은 좁은 문에 지원했다",
                  fontsize=11.5, color=INK, pad=9)
    ax2.text(72, 85, f"상관 r = {r:+.2f}\n원의 크기는 지원자 수",
             fontsize=9.5, color=INK, ha="right", va="top")
    ax2.grid(color=MUTED, alpha=0.25, linewidth=0.7)
    ax2.set_axisbelow(True)
    style(ax2)

    fig.tight_layout()
    finish(fig, "simpsons_paradox_berkeley.png")


if __name__ == "__main__":
    print("1장 인과 관련 그림을 만든다.")
    fig_randomization_balance()
    fig_observational_overlap()
    fig_simpsons_paradox_berkeley()
    print("\n완료.")
