"""13장 들어가는 이야기 그림 — 둘째 묶음 여섯.

  1. effect_size            「확실하다」와 「크다」는 서로 독립이다
  2. validity_conditions    어겼을 때의 대처마다 치르는 값이 다르다
  3. cochran_q              적힌 n 과 쓰이는 n
  4. homogeneity            「셋이 같지 않다」에서 한 나라를 빼면
  5. independence           행과 열의 합만 있으면 표 하나가 정해진다
  6. independence_titanic   기각했다는 말 뒤에 적을 것 셋

만드는 파일:
  docs/ch13/practice/img/two_independent_sentences.png
  docs/ch13/practice/img/each_remedy_has_a_price.png
  docs/ch13/test/img/written_n_versus_used_n.png
  docs/ch13/test/img/which_country_carries_it.png
  docs/ch13/test/img/margins_decide_the_table.png
  docs/ch13/test/img/three_things_left_to_write.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from itertools import product
from scipy import stats

PRAC = "docs/ch13/practice/img/"
TEST = "docs/ch13/test/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def chi2_from_table(tab):
    tab = np.asarray(tab, float)
    exp = np.outer(tab.sum(1), tab.sum(0)) / tab.sum()
    return ((tab - exp) ** 2 / exp).sum()


# 성별 × 주로 쓰는 손 — 13.2절 independence.md 와 13.3절 effect_size.md 가
# 함께 쓰는 표다. 행은 오른손·왼손·양손, 열은 남성·여성.
HAND = np.array([[934, 1070], [113, 92], [20, 8]], float)
N_HAND = HAND.sum()
CHI_HAND = chi2_from_table(HAND)
V_HAND = np.sqrt(CHI_HAND / (N_HAND * 1))
P_HAND = stats.chi2.sf(CHI_HAND, 2)
print(f"[표 확인] n = {N_HAND:.0f} · chi2 = {CHI_HAND:.3f} · "
      f"p = {P_HAND:.5f} · V = {V_HAND:.4f}")
print(f"          쪽은 11.806 · 0.00273 · 0.0726 이라 적는다")


# ════════════════════════════════════════════════════════════════════
# 1. effect_size — 「확실하다」와 「크다」는 서로 독립이다
#
# 쪽의 chisq_vs_cramersv 는 「표본을 키우면 카이제곱은 직선으로 커지고 V 는
# 수평선」을 이미 그린다. 그래서 들머리의 **머리글**만 그린다 — 두 문장이
# 서로 독립이라는 것. 가로축을 크기(V), 세로축을 확실함(p)으로 놓으면
# 네 칸이 생기고, 네 칸이 다 채워진다는 것이 「한 수에 뭉개진다」가 왜
# 잘못인가에 대한 답이다.
#
# 같은 칸 비율을 n 만 바꾸어 놓으면 **가로로만** 움직이지 않고 세로로만
# 움직인다 — 그 선이 이 그림의 뼈대다.
# ════════════════════════════════════════════════════════════════════
def table_at_n(base, n):
    """칸 비율을 그대로 두고 표본크기만 n 으로 바꾼다."""
    return base / base.sum() * n


NS = np.array([200, 500, 2237, 10_000], float)
pts = []
for n in NS:
    tb = table_at_n(HAND, n)
    ch = chi2_from_table(tb)
    pts.append((np.sqrt(ch / n), stats.chi2.sf(ch, 2), n))

# 크기가 큰데 확실하지 않은 보기 — 비율을 크게 벌리고 n 을 작게
BIG = np.array([[9.0, 1.0], [1.0, 9.0]])
big_pts = []
for n in (5, 12, 40):
    tb = table_at_n(BIG, n)
    ch = chi2_from_table(tb)
    big_pts.append((np.sqrt(ch / n), stats.chi2.sf(ch, 1), n))

print("\n[effect_size] 같은 칸 비율, n 만 바꾼다")
for v, pv, n in pts:
    print(f"  n = {n:7.0f}:  V = {v:.4f}  p = {pv:.3e}")
print("  크게 벌린 표")
for v, pv, n in big_pts:
    print(f"  n = {n:7.0f}:  V = {v:.4f}  p = {pv:.3e}")

fig, ax = plt.subplots(figsize=(11.2, 6.4))
ax.axhline(0.05, color="0.35", lw=2.0, ls="--")
ax.axvline(0.20, color="0.35", lw=2.0, ls="--")
ax.fill_between([0.0, 0.20], 1e-20, 0.05, color="tab:red", alpha=0.07)
ax.fill_between([0.20, 1.0], 1e-20, 0.05, color="tab:green", alpha=0.07)

vs = [q[0] for q in pts]
ps = [q[1] for q in pts]
ax.plot(vs, ps, "o-", color="tab:red", lw=2.4, ms=11, zorder=5,
        label="성별 × 주로 쓰는 손 — 칸 비율은 그대로, n 만 바꿨다")
for (v, pv, n) in pts:
    # 2,237 자리에는 동그라미와 설명 상자의 화살표가 겹치므로 왼쪽에 적는다
    dx, ha = ((-17, "right") if n == 2237 else (14, "left"))
    ax.annotate(f"n = {n:,.0f}", xy=(v, pv), xytext=(dx, 0),
                textcoords="offset points", fontsize=10.5,
                color="tab:red", fontweight="bold", va="center", ha=ha)
i2237 = int(np.argmin(abs(NS - 2237)))
ax.plot([vs[i2237]], [ps[i2237]], "o", ms=21, mfc="none", mec="tab:red",
        mew=2.6, zorder=6)

bvs = [q[0] for q in big_pts]
bps = [q[1] for q in big_pts]
ax.plot(bvs, bps, "s--", color="tab:green", lw=2.2, ms=10, zorder=5,
        label="비율을 크게 벌린 표 — 같은 식으로 n 만 바꿨다")
for (v, pv, n) in big_pts:
    ax.annotate(f"n = {n:,.0f}", xy=(v, pv), xytext=(-14, 0),
                textcoords="offset points", fontsize=10.5,
                color="tab:green", fontweight="bold", ha="right",
                va="center")

ax.set_yscale("log")
ax.set_ylim(1e-14, 6.0)
ax.set_xlim(0, 1.0)
# 로그 눈금의 기본 이름표는 mathtext 라 U+2212 를 끌고 와 한글 글꼴에서
# 두부가 된다. 맨 문자열로 직접 적는다 (ASCII 붙임표는 글꼴에 있다)
ax.set_yticks([1e-12, 1e-8, 1e-4, 0.05, 1.0],
              ["1e-12", "1e-8", "1e-4", "0.05", "1"])
ax.invert_yaxis()
# 네 칸의 이름은 축 비율로 적는다 — 뒤집은 축에서 「확실」이 위쪽이다
for fx, fy, ha, txt, col in [
        (0.025, 0.965, "left", "확실한데 작다", "tab:red"),
        (0.975, 0.965, "right", "확실하고 크다", "tab:green"),
        (0.025, 0.012, "left", "불확실하고 작다", "0.45"),
        (0.975, 0.012, "right", "불확실한데 크다", "0.45")]:
    ax.text(fx, fy, txt, transform=ax.transAxes, ha=ha, va="bottom"
            if fy < 0.5 else "top", fontsize=13.5, color=col,
            fontweight="bold")
ax.set_xlabel("연관의 크기 — 크래머 V", fontsize=12)
ax.set_ylabel("확실함 — p 값 (위로 갈수록 확실하다)", fontsize=12)
ax.set_title("세로로만 움직이는 선이 있고 가로로만 움직이는 선이 있다",
             fontsize=13)
ax.legend(fontsize=10.8, loc="center", bbox_to_anchor=(0.52, 0.80))
ax.grid(alpha=0.2)
ax.annotate(f"쪽의 표 — V = {V_HAND:.4f} 인데 p = {P_HAND:.5f}\n"
            f"「연관이 있다」는 확실하고 「연관이 크다」는 아니다",
            xy=(vs[i2237] + 0.012, ps[i2237]), xytext=(0.30, 2.0e-3),
            fontsize=11.5, color="tab:red", fontweight="bold", ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="tab:red",
                      alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))

fig.suptitle("「확실하다」와 「크다」가 한 수에 뭉개진다", fontsize=15,
             y=0.975)
plt.tight_layout()
save(fig, PRAC + "two_independent_sentences.png")


# ════════════════════════════════════════════════════════════════════
# 2. validity_conditions — 어겼을 때의 대처마다 치르는 값이 다르다
#
# 쪽의 validity_type1 이 「기대도수가 작을 때 세 검정의 실제 1 종 오류율」을
# 이미 그린다. 곧 들머리의 첫 beat(「문턱이 아니라 완만한 경사」)는 그 그림의
# 것이다. 여기서는 들머리의 **둘째 beat** 만 그린다 — 「어겼을 때 할 일도
# 하나가 아니고 각자 치르는 값이 달라서.」
#
# 처음에는 「조이는 쪽은 모두 5% 아래로 내려간다」를 그리려 했는데 재어 보니
# 그렇지 않았다. **범주를 합치는 길은 늘 5% 를 넘는다** — 검정력을 벌기는
# 하지만 그것은 명목을 넘겨서 번 것이다. 피셔는 늘 아래에 있고 그만큼
# 덜 잡는다. 그래서 1 종 오류율과 검정력을 **한 평면에** 놓는다. 그 평면이
# validity_type1 에 없는 축(검정력)을 들고 있고, 「공짜가 아니다」도 거기서
# 한눈에 보인다.
#
# 드문 범주가 전체의 3% 인 3×2 표에서 n 을 바꿔 최소 기대도수를 1.5 부터
# 6.9 까지 올린다. 합치는 길은 드문 행을 둘째 행에 붙인다.
# ════════════════════════════════════════════════════════════════════
RARE, ROW1 = 0.03, 0.55
ROWP = np.array([ROW1, 1 - ROW1 - RARE, RARE])
NGRID = [100, 160, 240, 340, 460]
REPS2 = 9_000
ALT = np.array([[0.5, 0.5], [0.355, 0.645], [0.5, 0.5]])


def three_paths(colp, n, reps, seed):
    """① 그대로 ② 드문 범주를 합친다 ③ 합친 표에서 피셔 정확검정."""
    r = np.random.default_rng(seed)
    cell = (ROWP[:, None] * colp).ravel()
    D = r.multinomial(n, cell, size=reps).reshape(reps, 3, 2)
    a = b = c = used = 0
    for tab in D:
        if tab.sum(0).min() == 0 or tab.sum(1).min() == 0:
            continue
        used += 1
        a += stats.chi2_contingency(tab, correction=False)[1] < 0.05
        m = np.array([tab[0], tab[1] + tab[2]], float)
        if m.sum(0).min() > 0 and m.sum(1).min() > 0:
            b += stats.chi2_contingency(m, correction=False)[1] < 0.05
            c += stats.fisher_exact(m.astype(int))[1] < 0.05
    return a / used, b / used, c / used


emins, T1, PW = [], [], []
print("\n[validity] 드문 범주가 3% 인 3x2 표 — 세 길을 재어 본다")
print("  최소기대      1 종 오류율                검정력")
print("            그대로  합치기   피셔    그대로  합치기   피셔")
for n in NGRID:
    emins.append(n * RARE * 0.5)
    T1.append(three_paths(np.tile([0.5, 0.5], (3, 1)), n, REPS2, 31 + n))
    PW.append(three_paths(ALT, n, REPS2, 77 + n))
    print(f"  {emins[-1]:6.1f}   {T1[-1][0]:7.4f}{T1[-1][1]:8.4f}"
          f"{T1[-1][2]:8.4f}  {PW[-1][0]:7.3f}{PW[-1][1]:8.3f}"
          f"{PW[-1][2]:8.3f}")
T1, PW = np.array(T1), np.array(PW)

PATHS = [("보정 없이 그대로", "0.40", "o-"),
         ("드문 범주를 합친다", "tab:purple", "s-"),
         ("합친 표에 정확검정", "tab:blue", "^-")]
EM = np.array(emins)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))

ax1.axhline(0.05, color="tab:red", lw=2.2, ls="--")
ax1.axvspan(EM[0] - 0.3, 5.0, color="tab:red", alpha=0.06)
for j, (nm, col, mk) in enumerate(PATHS):
    ax1.plot(EM, T1[:, j], mk, color=col, lw=2.6, ms=11, label=nm)
ax1.text(EM[0] - 0.1, 0.0585, "흔히 쓰는 문턱 5 의 왼쪽", ha="left",
         va="bottom", fontsize=10.5, color="tab:red", fontweight="bold")
ax1.axvline(5.0, color="tab:red", lw=1.6, ls=":")
ax1.set_xlabel("가장 작은 기대도수", fontsize=11)
ax1.set_ylabel("실제 1 종 오류율", fontsize=11)
ax1.set_ylim(0.030, 0.062)
ax1.set_xlim(EM[0] - 0.3, EM[-1] + 0.3)
ax1.set_title("자리가 길마다 다르다 — 합치면 가장 높고 조이면 늘 아래다",
              fontsize=12.5)
ax1.legend(fontsize=10.2, loc="lower right",
           bbox_to_anchor=(1.0, 0.16))
ax1.grid(alpha=0.25)
ax1.annotate("합치는 길은 다섯 자리 모두에서\n셋 가운데 가장 높다",
             xy=(EM[-1], T1[-1, 1] + 0.0004), xytext=(EM[-1] + 0.2, 0.0615),
             fontsize=10.8, color="tab:purple", fontweight="bold",
             ha="right", va="top",
             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.8))
ax1.annotate("정확검정은 다섯 자리 모두\n명목 아래에 있다",
             xy=(EM[2], T1[2, 2] - 0.0004), xytext=(EM[2] + 0.15, 0.0335),
             fontsize=10.8, color="tab:blue", fontweight="bold",
             ha="left", va="bottom",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8))

gain_m = (PW[:, 1] - PW[:, 0]) * 100
gain_f = (PW[:, 2] - PW[:, 0]) * 100
w = 0.34
bx = np.arange(len(EM))
ax2.bar(bx - w / 2, gain_m, width=w, color="tab:purple",
        label="드문 범주를 합친다")
ax2.bar(bx + w / 2, gain_f, width=w, color="tab:blue",
        label="합친 표에 정확검정")
for i in range(len(EM)):
    ax2.text(i - w / 2, gain_m[i] + 0.12, f"{gain_m[i]:+.1f}", ha="center",
             va="bottom", fontsize=10.5, fontweight="bold",
             color="tab:purple")
    ax2.text(i + w / 2, gain_f[i] + 0.12, f"{gain_f[i]:+.1f}", ha="center",
             va="bottom", fontsize=10.5, fontweight="bold", color="tab:blue")
ax2.axhline(0, color="0.3", lw=1.6)
ax2.set_xticks(bx, [f"{e:.1f}" for e in EM], fontsize=11)
ax2.set_xlabel("가장 작은 기대도수", fontsize=11)
ax2.set_ylabel("그대로 둔 길보다 더 잡는 양 (백분율 점)", fontsize=11)
ax2.set_ylim(0, 10.2)
ax2.set_title("더 잡는 만큼이 그 길의 벌이다", fontsize=12.5)
ax2.legend(fontsize=10.5, loc="upper right")
ax2.grid(axis="y", alpha=0.25)
ax2.text(0.03, 0.975, "합치는 길의 벌이는 왼쪽에서 더 쓴 5% 와\n"
         "함께 읽어야 한다 — 공짜가 아니다.\n"
         "자유도도 2 에서 1 로 줄어 묻는 물음이 바뀐다",
         transform=ax2.transAxes, ha="left", va="top", fontsize=10.8,
         color="0.25", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="#f6f6f6", ec="0.72"))

fig.suptitle("5 는 문턱이 아니라 완만한 경사다 — 어겼을 때 할 일도 "
             "하나가 아니다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, PRAC + "each_remedy_has_a_price.png")


# ════════════════════════════════════════════════════════════════════
# 3. cochran_q — 적힌 n 과 쓰이는 n
#
# 쪽의 cochran_informative 는 「모두 성공·모두 실패인 사람이 기여하지 않고,
# Q 의 순열분포가 11 개 값만 갖는다」를 그린다. cochran_q_size 는 기각률
# 곡선이다. 들머리가 그 둘과 다르게 말하는 것은 **머리글** 하나다 —
# 「적힌 n 과 쓰이는 n 이 다르다. … 사람을 더 모아도 그 사람들이 그런
# 사람이면 검정은 달라지지 않는다.」
#
# 그래서 쪽의 12 행에 모두 성공·모두 실패인 사람만 계속 더해 본다.
# 적힌 n 은 올라가고 쓰이는 n 과 Q 와 p 는 꿈쩍도 하지 않는다.
# ════════════════════════════════════════════════════════════════════
TASKS = np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0],
                  [1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 0],
                  [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]], float)


def cochran_q(d):
    n, k = d.shape
    G = d.sum(0)
    L = d.sum(1)
    T = L.sum()
    num = (k - 1) * (k * (G ** 2).sum() - T ** 2)
    den = k * T - (L ** 2).sum()
    return num / den


def exact_p(d, kk=3):
    """사람마다 이름표를 섞는 3^m 가지를 다 세어 정확한 p 를 낸다."""
    info = d[(d.sum(1) > 0) & (d.sum(1) < kk)]
    m = len(info)
    obs = cochran_q(d)
    tot = hit = 0
    for combo in product(range(kk), repeat=m):
        cand = np.array([np.roll(info[i], combo[i]) for i in range(m)], float)
        tot += 1
        if cochran_q(cand) >= obs - 1e-9:
            hit += 1
    return hit / tot, tot, obs


ADD = [0, 3, 10, 30, 100]
written, used, qs, ps = [], [], [], []
for a in ADD:
    extra = np.vstack([np.ones((a, 3)), np.zeros((a, 3))]) if a else \
        np.zeros((0, 3))
    d = np.vstack([TASKS, extra])
    L = d.sum(1)
    written.append(len(d))
    used.append(int(((L > 0) & (L < 3)).sum()))
    q = cochran_q(d)
    qs.append(q)
    ps.append(stats.chi2.sf(q, 2))

pe, ncomb, qobs = exact_p(TASKS)
print("\n[cochran] 모두 성공·모두 실패인 사람만 계속 더해 본다")
print(f"  배치 {ncomb} 가지를 다 세어 정확한 p = {pe:.4f} "
      f"(쪽은 0.0159) · 근사 p = {stats.chi2.sf(qobs, 2):.4f} (쪽은 0.0131)")
for a, w_, u_, q, p_ in zip(ADD, written, used, qs, ps):
    print(f"  더한 사람 {a:3d}:  적힌 n = {w_:3d} · 쓰이는 n = {u_} · "
          f"Q = {q:.4f} · p = {p_:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.8),
                               gridspec_kw={"width_ratios": [1.15, 1]})
bx = np.arange(len(ADD))
w = 0.38
ax1.bar(bx - w / 2, written, width=w, color="0.62", label="표에 적힌 n")
ax1.bar(bx + w / 2, used, width=w, color="tab:red",
        label="검정이 실제로 쓰는 n")
for i, (w_, u_) in enumerate(zip(written, used)):
    ax1.text(i - w / 2, w_ + 4, f"{w_}", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color="0.35")
    ax1.text(i + w / 2, u_ + 4, f"{u_}", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color="tab:red")
ax1.set_xticks(bx, [f"+{2 * a}" if a else "쪽의 자료\n(12 명)"
                    for a in ADD], fontsize=11)
ax1.set_xlabel("모두 성공·모두 실패인 사람을 더 모았다", fontsize=11)
ax1.set_ylabel("사람 수", fontsize=11)
ax1.set_ylim(0, max(written) * 1.22)
ax1.set_title("적힌 n 은 오르는데 쓰이는 n 은 아홉에 그대로다",
              fontsize=12.5)
ax1.legend(fontsize=10.5, loc="upper left")
ax1.grid(axis="y", alpha=0.25)

ax2.plot(bx, qs, "o-", color="tab:red", lw=2.6, ms=12,
         label="검정통계량 Q")
ax2.set_xticks(bx, [f"+{2 * a}" if a else "쪽의 자료" for a in ADD],
               fontsize=11)
ax2.set_xlabel("모두 성공·모두 실패인 사람을 더 모았다", fontsize=11)
ax2.set_ylabel("Q", fontsize=11)
ax2.set_ylim(0, 14)
ax2.axhline(stats.chi2.ppf(0.95, 2), color="0.35", lw=1.8, ls=":")
ax2.text(len(ADD) - 1.25, stats.chi2.ppf(0.95, 2) + 0.35,
         "자유도 2 의 5% 문턱 5.99", ha="right", va="bottom", fontsize=10.5,
         color="0.3", fontweight="bold")
ax2.text(len(ADD) / 2 - 0.5, qs[0] + 1.1,
         f"Q = {qs[0]:.4f} · p = {ps[0]:.4f} — 한 자리도 달라지지 않는다",
         ha="center", va="bottom", fontsize=12, color="tab:red",
         fontweight="bold")
ax2.set_title("사람을 더 모아도 그 사람이 그런 사람이면", fontsize=12.5)
ax2.grid(alpha=0.25)

fig.suptitle("적힌 n 과 쓰이는 n 이 다르다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, TEST + "written_n_versus_used_n.png")


# ════════════════════════════════════════════════════════════════════
# 4. homogeneity — 「셋이 같지 않다」에서 한 나라를 빼면
#
# 쪽의 homogeneity_profiles 는 「세 나라의 비율 프로파일과 12 칸의 기여」를
# 그린다. 들머리가 던지고 끝나는 물음은 그 다음이다 — 「어느 나라가 얼마나
# 다른지는 어떻게 읽어 내는가.」 그래서 **쌍마다 따로 보는 길**을 그리고,
# 거기서 바로 드러나는 함정(세 쌍의 합이 전체가 아니다)까지 적는다.
#
# 자료는 쪽의 병원 만족도 표 그대로다.
# ════════════════════════════════════════════════════════════════════
# 칸 자료가 쪽에 전부 적혀 있지 않으므로 쪽이 **출력으로 적어 둔 수**만
# 쓴다. 보기 1 (3) 의 열별 기여, 연습문제의 쌍별 통계량, 전체 212.94 다.
print("\n[homogeneity] 쪽이 출력으로 적어 둔 수만 쓴다")
PAIR = {"US · Canada": 11.73, "Canada · Mexico": 27.27,
        "US · Mexico": 210.57}
WHOLE = 212.94
CONTRIB = {"US": 74.45, "Canada": 1.87, "Mexico": 136.62}
print(f"  쌍별 통계량 {PAIR} · 전체 {WHOLE} · 합 {sum(PAIR.values()):.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                               gridspec_kw={"width_ratios": [1.1, 1]})

names = list(PAIR)
vals = [PAIR[k] for k in names]
bx = np.arange(3)
bc = ["tab:green", "tab:orange", "tab:red"]
ax1.bar(bx, vals, color=bc, width=0.58)
for i, v in enumerate(vals):
    # 가장 큰 막대는 꼭대기가 점선과 붙으므로 값을 막대 안에 흰 글자로
    if v > 150:
        ax1.text(i, v - 12, f"{v:.2f}", ha="center", va="top",
                 fontsize=13, fontweight="bold", color="white")
    else:
        ax1.text(i, v + 6, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=12.5, fontweight="bold")
ax1.axhline(WHOLE, color="0.25", lw=2.2, ls="--")
ax1.text(-0.42, WHOLE + 7, f"세 나라를 한꺼번에 본 {WHOLE}", ha="left",
         va="bottom", fontsize=11, color="0.25", fontweight="bold")
ax1.axhline(stats.chi2.ppf(0.95, 4), color="tab:blue", lw=1.8, ls=":")
ax1.set_xlim(-0.55, 3.25)
ax1.text(2.55, stats.chi2.ppf(0.95, 4), "가로 점선은\n자유도 4 의 5% 문턱 9.49",
         ha="left", va="center", fontsize=10, color="tab:blue",
         fontweight="bold")
ax1.set_xticks(bx, names, fontsize=11.5)
ax1.set_ylabel("카이제곱 통계량", fontsize=11)
ax1.set_ylim(0, 312)
ax1.set_title("쌍마다 다시 물으면 한 쌍이 거의 전부를 들고 있다",
              fontsize=12.5)
ax1.grid(axis="y", alpha=0.25)
ax1.annotate(f"세 쌍의 합 {sum(PAIR.values()):.2f} 는\n전체 {WHOLE} 가 "
             f"아니다 —\n쌍마다 기준선이 다르다", xy=(0.5, 250),
             xytext=(0.18, 272), fontsize=10.8, color="0.25",
             fontweight="bold", ha="left",
             bbox=dict(boxstyle="round,pad=0.42", fc="#f6f6f6", ec="0.7"))

cn = list(CONTRIB)
cv = [CONTRIB[k] for k in cn]
ax2.barh(np.arange(3), cv, color=["tab:orange", "tab:green", "tab:red"],
         height=0.55)
for i, v in enumerate(cv):
    ax2.text(v + 3, i, f"{v:.2f}  ({100 * v / WHOLE:.0f}%)", va="center",
             ha="left", fontsize=12, fontweight="bold")
ax2.set_yticks(np.arange(3), cn, fontsize=12.5)
ax2.set_xlim(0, 196)
ax2.set_xlabel("그 나라의 열이 전체 통계량에 낸 몫", fontsize=11)
ax2.set_title("「셋이 같지 않다」는 한 나라의 말이었다", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)
ax2.text(0.975, 0.50, "그 하나를 빼고\n남은 둘만 보면 11.73 이다",
         transform=ax2.transAxes, ha="right", va="center", fontsize=11.5,
         color="tab:green", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="tab:green"))

fig.suptitle("「셋이 같지 않다」에서 두 가지를 더", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, TEST + "which_country_carries_it.png")


# ════════════════════════════════════════════════════════════════════
# 5. independence — 행과 열의 합만 있으면 표 하나가 정해진다
#
# 쪽의 그림들은 예이츠 보정과 재표본 귀무분포를 그린다. 들머리의 머리글은
# 그것이 아니라 「행과 열의 합만 있으면 그것이 끝까지 계산된다」이고,
# 쪽에 그 표 자체를 그린 그림이 없다. 그래서 **합에서 표 하나가 솟아나는
# 것**을 그리고, 실제 표가 그 표와 어디서 갈리는지 바로 옆에 둔다.
# ════════════════════════════════════════════════════════════════════
RNAME = ["오른손", "왼손", "양손"]
CNAME = ["남성", "여성"]
R = HAND.sum(1)
C = HAND.sum(0)
EXP_HAND = np.outer(R, C) / N_HAND
CONTRIB_HAND = (HAND - EXP_HAND) ** 2 / EXP_HAND
print("\n[independence] 합만으로 표가 정해진다")
print(f"  행 합 {R} · 열 합 {C}")
for i, nm in enumerate(RNAME):
    print(f"  {nm}: 기대 {EXP_HAND[i, 0]:7.1f} {EXP_HAND[i, 1]:7.1f}  ·  "
          f"실제 {HAND[i, 0]:6.0f} {HAND[i, 1]:6.0f}  ·  "
          f"기여 {CONTRIB_HAND[i].sum():6.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.25, 1]})

ax1.axis("off")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.text(5.0, 9.5, "열 합이 정한 비율로 어느 행이든 쪼개진다", ha="center",
         va="center", fontsize=12.5, fontweight="bold", color="0.2")
frac = C / C.sum()
ax1.text(5.0, 8.80, f"파선이 그 비율 — 남성 {frac[0]:.3f} 대 "
         f"여성 {frac[1]:.3f}", ha="center", va="center", fontsize=12,
         color="tab:blue", fontweight="bold")
ax1.text(5.0, 8.15, "실선은 실제로 나뉜 자리다", ha="center", va="center",
         fontsize=11.5, color="0.3", fontweight="bold")
ys = [6.9, 4.6, 2.3]
for (y, nm, rtot) in zip(ys, RNAME, R):
    ax1.barh(y, 7.2, left=1.4, height=1.15, color="0.90", edgecolor="0.6")
    cut = 1.4 + 7.2 * frac[0]
    ax1.plot([cut, cut], [y - 0.575, y + 0.575], color="tab:blue", lw=2.6,
             ls="--")
    # 실제로 나뉜 자리를 함께 그어야 어긋남이 글자가 아니라 그림이 된다
    i0 = RNAME.index(nm)
    acut = 1.4 + 7.2 * HAND[i0, 0] / rtot
    off = abs(HAND[i0, 0] / rtot - frac[0])
    ax1.plot([acut, acut], [y - 0.575, y + 0.575],
             color="tab:red" if off > 0.05 else "0.35", lw=3.0)
    ax1.text(1.25, y, f"{nm}\n{rtot:.0f} 명", ha="right", va="center",
             fontsize=11.5, fontweight="bold", color="0.25")
    i = RNAME.index(nm)
    xl, xr = 1.4 + 7.2 * 0.17, 1.4 + 7.2 * 0.84
    ax1.text(xl, y + 0.24, f"기대 {EXP_HAND[i, 0]:.1f}", ha="center",
             va="center", fontsize=10.5, color="0.35")
    ax1.text(xr, y + 0.24, f"기대 {EXP_HAND[i, 1]:.1f}", ha="center",
             va="center", fontsize=10.5, color="0.35")
    gcol = "tab:red" if i == 2 else "0.2"
    ax1.text(xl, y - 0.26, f"실제 {HAND[i, 0]:.0f}", ha="center",
             va="center", fontsize=12, fontweight="bold", color=gcol)
    ax1.text(xr, y - 0.26, f"실제 {HAND[i, 1]:.0f}", ha="center",
             va="center", fontsize=12, fontweight="bold", color=gcol)
ax1.annotate("가장 멀리 벌어진 행", xy=(8.75, 2.3), xytext=(9.5, 1.15),
             fontsize=12, color="tab:red", fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
ax1.text(5.0, 0.35, f"행 합과 열 합만 적어 두면 여섯 칸의 기대도수가 "
         f"모두 나온다 (n = {N_HAND:.0f})", ha="center", va="center",
         fontsize=11, color="0.35")

by = np.arange(3)
cc = ["0.6", "0.6", "tab:red"]
ax2.barh(by, [CONTRIB_HAND[i].sum() for i in range(3)], color=cc,
         height=0.55)
for i in range(3):
    v = CONTRIB_HAND[i].sum()
    ax2.text(v + 0.14, i, f"{v:.2f}   ({100 * v / CHI_HAND:.0f}%)",
             va="center", ha="left", fontsize=12, fontweight="bold")
ax2.set_yticks(by, [f"{nm}\n({r:.0f} 명)" for nm, r in zip(RNAME, R)],
               fontsize=11.5)
ax2.set_xlim(0, 9.4)
ax2.set_xlabel(f"그 행이 통계량에 낸 몫 (표 전체가 {CHI_HAND:.2f})",
               fontsize=11)
ax2.set_title("가장 작은 행이 절반 넘게 낸다", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("행과 열의 합만 있으면 끝까지 계산된다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, TEST + "margins_decide_the_table.png")


# ════════════════════════════════════════════════════════════════════
# 6. independence_titanic — 기각했다는 말 뒤에 적을 것 셋
#
# 쪽의 titanic_permutation 은 「귀무가설 아래 여성 생존자 수의 분포와
# 233 의 자리」를 그린다 — 들머리의 둘째 문단(16.2 표준편차) 그대로다.
# 그래서 여기서는 들머리의 **마지막** 문단만 그린다 — 「적을 것이 셋이고
# 그 셋이 서로 다른 물음에 답한다.」 셋이 서로 다른 눈금 위에 있으므로
# 저마다의 축에 신뢰구간과 함께 놓고, 층을 나누면 또 달라진다는 것까지
# 함께 둔다. 수는 모두 쪽의 보기 6·7 과 연습문제 8 에 적힌 것이다.
# ════════════════════════════════════════════════════════════════════
A, B2, C2, D2 = 233, 81, 109, 468
EFF = [("비율의 차이", 0.5531, 0.4951, 0.6111, 0.0, "차이가 없으면 0"),
       ("위험비", 3.9280, 3.2770, 4.7084, 1.0, "차이가 없으면 1"),
       ("오즈비", 12.3507, 8.9000, 17.1393, 1.0, "차이가 없으면 1")]
STRATA = [("1 등실", 51.9037, 216), ("2 등실", 62.4510, 184),
          ("3 등실", 6.3830, 491), ("전체", 12.3507, 891)]
print("\n[titanic] 쪽이 적어 둔 세 효과크기와 층별 오즈비를 그대로 쓴다")
for nm, est, lo, hi, _, _ in EFF:
    print(f"  {nm:10s} {est:8.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
for nm, orr, n in STRATA:
    print(f"  {nm:6s} 오즈비 {orr:8.4f}  (n = {n})")

fig = plt.figure(figsize=(13.6, 5.1))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.45], wspace=0.42)
for j, (nm, est, lo, hi, nullv, nulltxt) in enumerate(EFF):
    ax = fig.add_subplot(gs[0, j])
    ax.errorbar([0], [est], yerr=[[est - lo], [hi - est]], fmt="o",
                ms=14, color="tab:red", ecolor="tab:red", elinewidth=3.4,
                capsize=12, capthick=3.0)
    ax.axhline(nullv, color="0.35", lw=2.0, ls="--")
    ax.text(-0.42, nullv, nulltxt, ha="left", va="bottom", fontsize=10.5,
            color="0.35", fontweight="bold")
    ax.text(0.08, est, f"{est:.3f}", ha="left", va="center", fontsize=13,
            fontweight="bold", color="tab:red")
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([])
    ax.set_title(nm, fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    lo_, hi_ = min(nullv, lo), max(hi, nullv)
    pad = (hi_ - lo_) * 0.30
    ax.set_ylim(lo_ - pad, hi_ + pad)

ax4 = fig.add_subplot(gs[0, 3])
sy = np.arange(4)
scol = ["tab:blue", "tab:blue", "tab:blue", "0.35"]
ax4.barh(sy, [s[1] for s in STRATA], color=scol, height=0.55)
for i, (nm, orr, n) in enumerate(STRATA):
    ax4.text(orr + 1.4, i, f"{orr:.2f}", va="center", ha="left",
             fontsize=12, fontweight="bold")
ax4.set_yticks(sy, [f"{s[0]}\n(n = {s[2]})" for s in STRATA], fontsize=11)
ax4.invert_yaxis()
ax4.set_xlim(0, 104)
ax4.set_xlabel("오즈비", fontsize=11)
ax4.set_title("층을 나누면 또 달라진다", fontsize=13)
ax4.grid(axis="x", alpha=0.25)
ax4.annotate("1·2 등실이 전체보다 훨씬 크다 —\n하나로 요약하면 이것을 잃는다",
             xy=(63.6, 1), xytext=(102, 2.05), fontsize=10.8,
             color="tab:blue", fontweight="bold", ha="right", va="center",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8))

fig.suptitle("기각했다는 말로 끝나지 않는다 — 셋이 서로 다른 물음에 답한다",
             fontsize=15, y=1.0)
save(fig, TEST + "three_things_left_to_write.png")
