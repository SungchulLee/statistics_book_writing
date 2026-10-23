r"""왜 지수를 옮기지 않고 벌점을 더했는가.

이 쪽 절 1 은 엘라스틱넷의 벌점을 alpha*L1 + (1-alpha)*L2/2 라는 **덧셈**으로
적고, 절 2 의 표는 alpha = 0 이 능형, alpha = 1 이 라쏘라고 한다. 그러면 당연히
떠오르는 물음이 있다 — 능형이 제곱이고 라쏘가 절댓값이니, **지수를 1 과 2 사이로
옮기면** 되지 않는가.

그 길은 이미 있었다. 프랭크와 프리드먼(1993)이 일반화 능형 벌점 sum|b_j|^q 를
내놓았고(그들의 그림 5), 푸(1998)가 그것을 「브리지 회귀」라 불렀다. 라쏘보다
먼저 있던 길이다.

그런데 그 길로는 갈 수 없다. q > 1 이면 벌점의 **0 에서의 기울기가 0 이 되어**
정확한 0 이 나오지 않는다. 판과 리(2001)가 L_q (q >= 1) 집안에서 희소해를 내는
것은 q = 1 뿐임을 증명했다.

① 두 집안의 제약영역 — 둘 다 마름모에서 원으로 가는데 모서리의 운명이 다르다.
② 0 에서의 기울기 — 기제는 이 한 줄이다.
③ 참값이 0 인 좌표의 |b_j| — q > 1 에서는 0 에 닿지 않고 내려가기만 한다.

만드는 파일: docs/ch18/elastic_net/img/why_additive.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from scipy.optimize import minimize                        # noqa: E402
from sklearn.linear_model import ElasticNet                # noqa: E402

TH = np.linspace(0, 2 * np.pi, 2000)
U, V = np.cos(TH), np.sin(TH)


def ball_lq(q):
    """{b : sum|b_j|^q = 1} 의 경계. 네 꼭짓점 (+-1,0),(0,+-1) 을 지난다."""
    r = (np.abs(U) ** q + np.abs(V) ** q) ** (-1.0 / q)
    return r * U, r * V


def ball_en(a):
    """{b : a|b|_1 + (1-a)|b|_2^2/2 = c} 를 같은 네 꼭짓점에 맞춘다."""
    c = a + (1 - a) / 2.0                      # (1,0) 에서의 값
    s = a * (np.abs(U) + np.abs(V))
    t = (1 - a) / 2.0 * (U ** 2 + V ** 2)
    # r 에 대해 s*r + t*r^2 = c 를 푼다
    r = np.where(t > 1e-12,
                 (-s + np.sqrt(s ** 2 + 4 * t * c)) / (2 * np.maximum(t, 1e-12)),
                 c / np.maximum(s, 1e-12))
    return r * U, r * V


# === ③ 실제로 세어 본다 =============================================
RNG = np.random.default_rng(0)
N, P = 80, 30
Xd = RNG.normal(size=(N, P))
Xd -= Xd.mean(0)
Xd /= Xd.std(0)
B = np.zeros(P)
B[:5] = [2.0, -1.5, 1.0, 0.8, -0.6]
yd = Xd @ B + RNG.normal(size=N)
yd -= yd.mean()
LAM = 0.12


def fit_lq(q, lam=LAM):
    """일반 L_q 벌점을 수치로 푼다 (q = 1 은 좌표하강이 정확한 0 을 준다)."""
    if abs(q - 1.0) < 1e-12:
        return ElasticNet(alpha=lam, l1_ratio=1.0, fit_intercept=False,
                          max_iter=200_000, tol=1e-12).fit(Xd, yd).coef_

    def f(b):
        r = yd - Xd @ b
        return r @ r / (2 * N) + lam * np.sum(np.abs(b) ** q)

    def g(b):
        r = yd - Xd @ b
        pen = lam * q * np.sign(b) * np.abs(b) ** (q - 1)
        return -Xd.T @ r / N + pen

    b0 = ElasticNet(alpha=lam, l1_ratio=1.0, fit_intercept=False,
                    max_iter=50_000).fit(Xd, yd).coef_
    res = minimize(f, b0, jac=g, method="L-BFGS-B",
                   options=dict(maxiter=50_000, ftol=1e-16, gtol=1e-12))
    return res.x


def fit_en(a, lam=LAM):
    return ElasticNet(alpha=lam, l1_ratio=max(a, 1e-8), fit_intercept=False,
                      max_iter=200_000, tol=1e-12).fit(Xd, yd).coef_


QS = [1.0, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0]
AS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]
FLOOR = 1e-17            # 로그 축에서 「정확히 0」을 앉힐 자리

LASSO = fit_en(1.0)
NULLIDX = np.flatnonzero((B == 0) & (LASSO == 0))     # 라쏘가 0 으로 보낸 잡음

MED_Q, MED_A = [], []
for q in QS:
    b = fit_lq(q)
    MED_Q.append(float(np.median(np.abs(b[NULLIDX]))))
for a in AS:
    b = fit_en(a)
    MED_A.append(float(np.median(np.abs(b[NULLIDX]))))

# lambda 를 통째로 훑어 「정확히 0 이 한 번이라도 나오는가」를 본다.
# 고정된 lambda 에서 비교하면 문턱 크기(alpha*lambda)와 모서리의 존재가 섞인다.
LGRID = np.logspace(np.log10(0.6), np.log10(0.003), 16)
MAXZ_Q = [max(int(np.sum(fit_lq(q, lam) == 0.0)) for lam in LGRID) for q in QS]
MAXZ_A = [max(int(np.sum(fit_en(a, lam) == 0.0)) for lam in LGRID) for a in AS]

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("① 두 집안은 모두 라쏘와 능형을 양 끝으로 갖는다")
print("   L_q 집안:  q = 1 이 라쏘, q = 2 가 능형")
print("   덧셈 집안: alpha = 1 이 라쏘, alpha = 0 이 능형")

print("\n② 벌점의 0 에서의 기울기  (b_j -> 0+)")
print("      q   d|b|^q/db 의 극한   |   alpha   덧셈 벌점의 기울기")
for q, a in zip(QS, AS):
    sq = "1" if abs(q - 1) < 1e-12 else "0"
    print(f"   {q:>4.2f}   {sq:>16}   |   {a:>5.2f}   {a:>18.2f}")
print("   **q 를 1 에서 조금만 올려도 기울기가 곧장 0 이다. "
      "alpha 는 줄어들 뿐 0 이 되지 않는다.**")

print(f"\n③ 참값이 0 인 좌표 {len(NULLIDX)} 개의 |b_j| 중앙값")
print(f"   (n = {N}, p = {P}, 참 변수 5 개, lambda = {LAM})")
print("      q    L_q 의 |b_j|      |   alpha   덧셈의 |b_j|")
for (q, mq), (a, ma) in zip(zip(QS, MED_Q), zip(AS, MED_A)):
    sq = "정확히 0" if mq == 0.0 else f"{mq:.2e}"
    sa = "정확히 0" if ma == 0.0 else f"{ma:.2e}"
    print(f"   {q:>4.2f}   {sq:>14}   |   {a:>5.2f}   {sa:>14}")
print("   **q > 1 은 어디서도 0 이 아니다. q 를 1 에 가까이 할수록 값이 "
      "작아질 뿐이다.**")
print("   그래서 「문턱을 두고 0 을 세는」 방식은 q 가 1 에 가까우면 "
      "없는 희소성을 만들어 낸다 —")
print(f"   q = 1.05 의 {MED_Q[2]:.1e} 는 1e-6 아래지만 0 이 아니다.")
print("   (출발점을 라쏘해·영벡터·난수로 바꿔도 같은 값에 닿는 것을 "
      "따로 확인했다. 풀개 탓이 아니다.)")

print(f"\n④ lambda 를 {len(LGRID)} 개 격자로 통째로 훑어 "
      "**정확히 0** 이 나오는 최대 개수")
print("      q   L_q      |   alpha   덧셈")
for (q, zq), (a, za) in zip(zip(QS, MAXZ_Q), zip(AS, MAXZ_A)):
    print(f"   {q:>4.2f}   {zq:>3d}      |   {a:>5.2f}   {za:>3d}")
print("   **q >= 1.05 는 어떤 lambda 에서도 0 이 하나도 없고, "
      "alpha > 0 은 모두 0 을 낸다.**")
print("   고정된 lambda 에서 견주면 문턱 크기(alpha*lambda)와 모서리의 "
      "존재가 섞이므로 이렇게 보아야 한다.")
print(f"   q = 1.02 가 {MAXZ_Q[1]} 개로 나오는 것은 반례가 아니라 "
      "**산술의 한계**다 —")
print(f"   ③ 에서 그 자리의 참값이 {MED_Q[1]:.1e} 이고, 배정밀도 최적화가 "
      "그것을 0 과 구별하지 못한다.")
print("   q 를 1 에 붙일수록 「0 처럼 보이는 값」이 나오지만 0 은 아니다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
CQ = plt.cm.autumn(np.linspace(0.0, 0.7, len(QS)))
CA = plt.cm.winter(np.linspace(0.0, 0.8, len(AS)))

for q, c in zip(QS, CQ):
    x, y = ball_lq(q)
    ax1.plot(x, y, color=c, lw=2.4, label=f"$q = {q:g}$")
for a, c in zip(AS, CA):
    x, y = ball_en(a)
    ax1.plot(x, y, color=c, lw=2.4, ls="--")
ax1.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], "k.", ms=10, zorder=5)
ax1.set_aspect("equal")
ax1.set_xlim(-1.35, 1.35)
ax1.set_ylim(-1.35, 1.35)
ax1.set_title("① 실선 $L_q$, 파선 덧셈 — 둘 다 마름모에서 원으로",
              fontsize=11.5)
ax1.legend(loc="upper right", fontsize=8.5, ncol=2)
ax1.grid(alpha=0.25)

bb = np.linspace(0, 0.30, 600)
for q, c in zip(QS, CQ):
    ax2.plot(bb, q * np.power(np.maximum(bb, 1e-300), q - 1), color=c,
             lw=2.8, label=f"$q = {q:g}$")
for a, c in zip(AS, CA):
    ax2.plot(bb, a + (1 - a) * bb, color=c, lw=2.8, ls="--")
ax2.plot([0] * len(QS), [1.0 if abs(q - 1) < 1e-12 else 0.0 for q in QS],
         "o", color="#d62728", ms=11, zorder=5)
ax2.plot([0] * len(AS), AS, "s", color="#1f77b4", ms=9, zorder=5)
ax2.set_xlim(-0.012, 0.30)
ax2.set_ylim(-0.08, 2.1)
ax2.set_xlabel("$\\beta_j$")
ax2.set_ylabel("벌점의 기울기 $\\partial P / \\partial \\beta_j$")
ax2.set_title("② $\\beta_j = 0$ 에서의 기울기가 전부다", fontsize=12)
ax2.legend(loc="upper left", fontsize=9, ncol=2, title="실선 $L_q$")
ax2.grid(alpha=0.25)
ax2.annotate("$q > 1$ 은 모두 $0$ 으로 내려앉는다", (0, 0),
             xytext=(1.2, 1.1), textcoords="offset fontsize", fontsize=10.5,
             color="#d62728",
             arrowprops=dict(arrowstyle="->", lw=1.6, color="#d62728"))
ax2.annotate("덧셈은 $\\alpha$ 에서 멈춘다", (0, 0.5),
             xytext=(2.0, 1.0), textcoords="offset fontsize", fontsize=10.5,
             color="#1f77b4",
             arrowprops=dict(arrowstyle="->", lw=1.6, color="#1f77b4"))

w = 0.38
xs = np.arange(len(QS))
ax3.bar(xs - w / 2, MAXZ_Q, width=w, color="#d62728", label="$L_q$ 집안")
ax3.bar(xs + w / 2, MAXZ_A, width=w, color="#1f77b4", label="덧셈 집안")
ax3.set_xticks(xs)
ax3.set_xticklabels([f"$q={q:g}$\n$\\alpha={a:g}$" for q, a in zip(QS, AS)],
                    fontsize=8.5)
ax3.set_ylabel(f"정확히 $0$ 인 계수의 최대 개수 (전체 ${P}$)")
ax3.set_title(f"③ $\\lambda$ 를 {len(LGRID)} 곳에서 훑어도", fontsize=12)
ax3.legend(loc="upper right", fontsize=10.5)
ax3.grid(alpha=0.25, axis="y")
ax3.text(3.35, P * 0.72,
         "$q \\geq 1.05$ 는 어느\n$\\lambda$ 에서도 $0$ 이 없다",
         fontsize=10.5, color="#d62728")

fig.tight_layout()
out = "docs/ch18/elastic_net/img/why_additive.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
