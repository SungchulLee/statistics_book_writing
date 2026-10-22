r"""1.5절 「그림으로 보는 통계학」 — 천식이 있으면 덜 죽는다는 규칙.

$1990$ 년대 폐렴 위험 예측 연구에서 학습기가 「천식 병력이 있으면 위험이 낮다」는
규칙을 배웠다. 자료에는 참으로 그렇게 적혀 있었다 — 천식이 있는 폐렴 환자는 곧바로
중환자실로 보내졌고, 그 공격적인 치료가 사망률을 낮추었기 때문이다. 모형이 배운 것은
천식의 효과가 아니라 **치료의 효과**였다.

여기서는 같은 구조를 직접 만들어 세 가지를 잰다.

① 관측된 연관과 참 인과가 **부호부터 다르다**.
② 그런데도 예측 성적표(AUC)는 그 변수를 쓰라고 말한다.
③ 그 예측으로 「저위험은 집으로」 정책을 세우면 사람이 더 죽는다.

만드는 파일: docs/ch01/modern/img/asthma_rule.png
실행:  python3 scripts/make_ch01_asthma_rule.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch01/modern/img/asthma_rule.png"
RNG = np.random.default_rng(20261022)

N = 120_000
P_ASTHMA = 0.12
B_ICU = -2.2          # 중환자실 치료가 사망 로그오즈를 낮추는 폭
B_ASTHMA = 0.8        # 천식 자체는 위험을 **높인다**
B_SEV = 1.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


sev = RNG.standard_normal(N)                       # 중증도
ast = (RNG.random(N) < P_ASTHMA).astype(float)     # 천식 병력
p_icu = sigmoid(-1.7 + 4.2 * ast + 1.1 * sev)      # 천식이면 거의 중환자실로
icu = (RNG.random(N) < p_icu).astype(float)
p_death = sigmoid(-2.2 + B_SEV * sev + B_ASTHMA * ast + B_ICU * icu)
death = (RNG.random(N) < p_death).astype(float)

obs1, obs0 = death[ast == 1].mean(), death[ast == 0].mean()
# 같은 치료를 받았다면 — 구조식에서 중환자실을 끄고 계산한 참 위험
p_noicu = sigmoid(-2.2 + B_SEV * sev + B_ASTHMA * ast)
cf1, cf0 = p_noicu[ast == 1].mean(), p_noicu[ast == 0].mean()
print(f"중환자실로 간 비율: 천식 {icu[ast==1].mean():.3f}, 아니면 {icu[ast==0].mean():.3f}")
print(f"① 자료에 적힌 사망률: 천식 {obs1:.4f}, 아니면 {obs0:.4f}  "
      f"→ 차이 {100*(obs1-obs0):+.2f} 백분율점")
print(f"   아무도 입원하지 않았다면: 천식 {cf1:.4f}, 아니면 {cf0:.4f}  "
      f"→ 차이 {100*(cf1-cf0):+.2f} 백분율점")

# --- ② 예측 성적표 ------------------------------------------------------
half = N // 2
tr, te = slice(0, half), slice(half, N)
X_with = np.column_stack([np.ones(N), sev, ast])
X_without = np.column_stack([np.ones(N), sev])


def fit_logit(X, y, steps=60):
    """뉴턴-랩슨으로 로지스틱 회귀를 적합한다."""
    b = np.zeros(X.shape[1])
    for _ in range(steps):
        p = sigmoid(X @ b)
        W = p * (1 - p)
        g = X.T @ (y - p)
        H = (X * W[:, None]).T @ X
        b = b + np.linalg.solve(H + 1e-9 * np.eye(len(b)), g)
    return b


def auc(score, y):
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


b_with = fit_logit(X_with[tr], death[tr])
b_without = fit_logit(X_without[tr], death[tr])
s_with = X_with[te] @ b_with
s_without = X_without[te] @ b_without
a_with, a_without = auc(s_with, death[te]), auc(s_without, death[te])
print(f"\n② 적합된 천식 계수 {b_with[2]:+.3f}  (참 인과계수는 {B_ASTHMA:+.1f})")
print(f"   시험자료 AUC: 천식을 쓴 모형 {a_with:.4f}, 안 쓴 모형 {a_without:.4f} "
      f"→ {a_with - a_without:+.4f}")

# --- ③ 그 모형으로 정책을 세우면 ---------------------------------------
QS = np.arange(0.05, 0.51, 0.05)
exc_with, exc_without, sent_ast = [], [], []
sev_te, ast_te = sev[te], ast[te]
p0 = sigmoid(-2.2 + B_SEV * sev_te + B_ASTHMA * ast_te)        # 집으로 보냈을 때
p_as_is = sigmoid(-2.2 + B_SEV * sev_te + B_ASTHMA * ast_te
                  + B_ICU * icu[te])                            # 지금 받는 치료대로
for q in QS:
    out = []
    for s in (s_with, s_without):
        home = s <= np.quantile(s, q)                            # 저위험 q 만큼 귀가
        deaths = np.where(home, p0, p_as_is).sum()
        out.append(10_000 * (deaths - p_as_is.sum()) / len(sev_te))
    exc_with.append(out[0])
    exc_without.append(out[1])
    home_w = s_with <= np.quantile(s_with, q)
    sent_ast.append(100 * ast_te[home_w].mean())
    print(f"③ 하위 {100*q:4.0f}% 귀가:  초과 사망 {out[0]:6.1f} 명 / 1 만 명 "
          f"(천식을 뺀 모형이면 {out[1]:6.1f}),  귀가자 중 천식 비율 "
          f"{sent_ast[-1]:5.2f}%  (전체 {100*P_ASTHMA:.0f}%)")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
C1, C2 = "#1f6f8b", "#c0532b"

ax = axes[0]
xx = np.arange(2)
ax.bar(xx - 0.2, [100 * obs0, 100 * obs1], 0.4, color=C1, label="자료에 적힌 사망률")
ax.bar(xx + 0.2, [100 * cf0, 100 * cf1], 0.4, color=C2,
       label="아무도 입원하지 않았다면")
for i, (a, b) in enumerate(zip([100 * obs0, 100 * obs1], [100 * cf0, 100 * cf1])):
    ax.text(i - 0.2, a + 0.4, f"{a:.1f}%", ha="center", fontsize=9)
    ax.text(i + 0.2, b + 0.4, f"{b:.1f}%", ha="center", fontsize=9)
ax.set_xticks(xx)
ax.set_xticklabels(["천식 없음", "천식 있음"], fontsize=10)
ax.set_ylabel("사망률 (%)")
ax.set_title("① 부호부터 다르다", fontsize=11)
ax.legend(fontsize=9, loc="upper left")

ax = axes[1]
ax.bar([0, 1], [a_without, a_with], 0.5, color=[C1, C2])
for i, v in enumerate([a_without, a_with]):
    ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=10)
ax.set_xticks([0, 1])
ax.set_xticklabels(["천식을 안 쓴 모형", "천식을 쓴 모형"], fontsize=10)
ax.set_ylim(min(a_without, a_with) - 0.03, max(a_without, a_with) + 0.03)
ax.set_ylabel("시험자료 AUC")
ax.set_title("② 예측 성적표는 쓰라고 말한다", fontsize=11)
ax.text(0.03, 0.93, f"적합된 천식 계수 {b_with[2]:+.2f}\n참 인과계수 {B_ASTHMA:+.1f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

ax = axes[2]
ax.plot(100 * QS, exc_with, "o-", color=C2, lw=2, label="천식을 쓴 모형")
ax.plot(100 * QS, exc_without, "s-", color=C1, lw=2, label="천식을 뺀 모형")
ax.axhline(0, color="0.4", lw=1)
ax.set_xlabel("저위험으로 분류해 집으로 보낸 비율 (%)")
ax.set_ylabel("초과 사망 (1 만 명당)")
ax.set_title("③ 그 예측으로 정책을 세우면", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.text(0.97, 0.06,
        f"하위 {100*QS[3]:.0f}% 를 보낼 때\n귀가자의 {sent_ast[3]:.1f}% 가 천식 "
        f"(전체 {100*P_ASTHMA:.0f}%)",
        transform=ax.transAxes, fontsize=9, ha="right",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
