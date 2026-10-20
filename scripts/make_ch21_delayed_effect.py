r"""늦게 갈라지는 곡선 — 로그순위 검정이 놓치는 효과.

처치가 처음 $t_0$ 개월 동안은 아무 일도 하지 않다가 그 뒤부터 위험을 절반으로 줄인다면,
비례위험이 깨지고 **로그순위 검정의 검정력이 주저앉는다.** 가중 로그순위와 RMST 는 버틴다.

① 한 번의 시험 — 늦게 갈라지는 두 곡선.
② 로그–로그 그림이 평행하지 않다.
③ 지연이 길어질수록 세 검정의 검정력이 갈린다.

만드는 파일: docs/ch21/cox/img/delayed_effect.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261020)

N_ARM = 150
TAU = 36.0  # 추적 종료 (개월)
LAM0 = np.log(2) / 12.0  # 대조군 중앙생존 12 개월
HR = 0.62  # 지연이 끝난 뒤의 위험비
REP = 1200
ALPHA = 0.05


def draw(n, t0, hr, rng):
    """위험이 t0 전에는 LAM0, 그 뒤에는 LAM0*hr 인 조각별 지수분포."""
    u = rng.random(n)
    s_t0 = np.exp(-LAM0 * t0)  # t0 까지 살아남을 확률
    early = u > s_t0
    t = np.empty(n)
    t[early] = -np.log(u[early]) / LAM0
    t[~early] = t0 - np.log(u[~early] / s_t0) / (LAM0 * hr)
    return t


def trial(t0, hr, rng, n=N_ARM):
    t = np.concatenate([draw(n, 0.0, 1.0, rng), draw(n, t0, hr, rng)])
    arm = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    ev = t <= TAU
    t = np.minimum(t, TAU)
    return t, ev, arm


def km(t, ev):
    """카플란-마이어. (시간, 생존확률) 을 계단으로 돌려준다."""
    o = np.argsort(t)
    t, ev = t[o], ev[o]
    n_risk = len(t) - np.arange(len(t))
    s = np.cumprod(np.where(ev, 1 - 1 / n_risk, 1.0))
    return t, s


def weighted_logrank(t, ev, arm, kind="logrank"):
    """동점이 없다고 보고(연속 시간) 계산한다. kind: logrank | fh01"""
    o = np.argsort(t)
    t, ev, arm = t[o], ev[o], arm[o]
    n = len(t)
    n_risk = n - np.arange(n)
    n1_risk = np.cumsum(arm[::-1])[::-1]
    if kind == "fh01":
        # 통합 KM 을 한 칸 앞의 값으로 — 늦은 차이에 무게를 준다
        s = np.cumprod(np.where(ev, 1 - 1 / n_risk, 1.0))
        s_prev = np.concatenate([[1.0], s[:-1]])
        w = 1.0 - s_prev
    else:
        w = np.ones(n)
    e = n1_risk / n_risk
    v = n1_risk * (n_risk - n1_risk) / n_risk**2
    U = np.sum(w[ev] * (arm[ev] - e[ev]))
    V = np.sum(w[ev] ** 2 * v[ev])
    return U / np.sqrt(V) if V > 0 else 0.0


def rmst(t, ev, tau=TAU):
    """tau 까지의 제한평균생존시간과 그 분산."""
    tt, s = km(t, ev)
    grid = np.concatenate([[0.0], tt[tt < tau], [tau]])
    sv = np.concatenate([[1.0], s[tt < tau]])
    area = np.sum(sv * np.diff(grid))
    # 분산 (표준적인 적분형 그린우드)
    o = np.argsort(t)
    ts, evs = t[o], ev[o]
    n_risk = len(ts) - np.arange(len(ts))
    var = 0.0
    tail = area
    for i in range(len(ts)):
        if ts[i] >= tau:
            break
        if evs[i]:
            # t_i 부터 tau 까지의 면적
            m = (grid[:-1] >= ts[i])
            a = np.sum(sv[m] * np.diff(grid)[m])
            var += a**2 / (n_risk[i] * (n_risk[i] - 1)) if n_risk[i] > 1 else 0.0
    del tail
    return area, var


def rmst_z(t, ev, arm):
    a0, v0 = rmst(t[arm == 0], ev[arm == 0])
    a1, v1 = rmst(t[arm == 1], ev[arm == 1])
    return (a1 - a0) / np.sqrt(v0 + v1)


from scipy import stats  # noqa: E402

CRIT = stats.norm.isf(ALPHA / 2)

# === ③ 지연 길이에 따른 검정력 =====================================
DELAYS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0])
power = {"로그순위": [], "가중 로그순위": [], "RMST 차이": []}
for t0 in DELAYS:
    z_lr, z_fh, z_rm = [], [], []
    for _ in range(REP):
        t, ev, arm = trial(t0, HR, RNG)
        z_lr.append(weighted_logrank(t, ev, arm, "logrank"))
        z_fh.append(weighted_logrank(t, ev, arm, "fh01"))
        z_rm.append(rmst_z(t, ev, arm))
    power["로그순위"].append(np.mean(np.abs(z_lr) > CRIT))
    power["가중 로그순위"].append(np.mean(np.abs(z_fh) > CRIT))
    power["RMST 차이"].append(np.mean(np.abs(z_rm) > CRIT))

# === ①② 보여 줄 한 번의 시험 =======================================
SHOW_T0 = 6.0
big = np.random.default_rng(7)
t, ev, arm = trial(SHOW_T0, HR, big, n=2000)

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))

for a, c, lab in [(0, "#1f77b4", "대조군"), (1, "#d62728", "처치군")]:
    tt, s = km(t[arm == a], ev[arm == a])
    ax1.step(np.concatenate([[0], tt]), np.concatenate([[1], s]), where="post",
             color=c, lw=2.4, label=lab)
ax1.axvline(SHOW_T0, color="#555555", lw=1.2, ls="--")
ax1.annotate(
    f"${SHOW_T0:.0f}$ 개월까지는\n두 곡선이 겹친다",
    xy=(SHOW_T0, 0.72),
    xytext=(16, 0.86),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)
ax1.set_xlabel("개월")
ax1.set_ylabel("생존확률")
ax1.set_title("① 늦게 갈라지는 곡선", fontsize=12)
ax1.set_xlim(0, TAU)
ax1.set_ylim(0, 1.02)
ax1.legend(loc="lower left", fontsize=10)
ax1.grid(alpha=0.25)

for a, c, lab in [(0, "#1f77b4", "대조군"), (1, "#d62728", "처치군")]:
    tt, s = km(t[arm == a], ev[arm == a])
    m = (s > 0.01) & (s < 0.995) & (tt > 0.5)
    ax2.plot(np.log(tt[m]), np.log(-np.log(s[m])), color=c, lw=2.4, label=lab)
ax2.set_xlabel("$\\log t$")
ax2.set_ylabel("$\\log(-\\log \\hat S(t))$")
ax2.set_title("② 평행하지 않다 — 비례위험이 깨졌다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25)
ax2.annotate(
    "앞쪽에서 붙어 있다가\n뒤로 갈수록 벌어진다",
    xy=(1.0, -1.25),
    xytext=(2.2, -2.6),
    ha="center",
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)

for (lab, ys), c, mk in zip(power.items(), ["#d62728", "#2ca02c", "#1f77b4"], "osD"):
    ax3.plot(DELAYS, np.array(ys) * 100, mk + "-", color=c, lw=2.4, ms=6, label=lab)
ax3.axhline(80, color="#555555", lw=1.0, ls="--")
ax3.text(0.2, 81.5, "검정력 $80\\%$", fontsize=9.5, color="#555555")
ax3.set_xlabel("효과가 나타나기까지의 지연 (개월)")
ax3.set_ylabel("검정력 ($\\%$)")
ax3.set_title("③ 지연이 길어지면 갈린다", fontsize=12)
ax3.set_ylim(0, 105)
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch21/cox/img/delayed_effect.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n설정: 팔당 {N_ARM} 명, 추적 {TAU:.0f} 개월, 대조군 중앙생존 12 개월,")
print(f"      지연이 끝난 뒤의 위험비 {HR}, 모의실험 {REP} 회, 양측 {ALPHA}")
print(f"\n{'지연(개월)':>10}{'로그순위':>12}{'가중 로그순위':>14}{'RMST 차이':>12}")
for i, t0 in enumerate(DELAYS):
    print(
        f"{t0:>10.0f}{power['로그순위'][i] * 100:>11.1f}%"
        f"{power['가중 로그순위'][i] * 100:>13.1f}%{power['RMST 차이'][i] * 100:>11.1f}%"
    )
i6 = int(np.where(DELAYS == 6.0)[0][0])
print(
    f"\n지연 6 개월에서: 로그순위 {power['로그순위'][i6] * 100:.1f}% 대 "
    f"가중 {power['가중 로그순위'][i6] * 100:.1f}%"
)
