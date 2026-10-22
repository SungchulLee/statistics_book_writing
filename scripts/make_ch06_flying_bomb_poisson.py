r"""나눗셈 한 번이 전시의 물음에 답했다 — 클라크 $1946$ 의 $\hat\lambda = \bar x$.

$1944$ 년 런던에 떨어진 V-$1$ 이 뭉쳐 떨어진다는 말이 돌았다. 클라크는 남런던
$144\ \mathrm{km^2}$ 를 $0.25\ \mathrm{km^2}$ 짜리 $576$ 칸으로 나누고 그 안의 폭탄
$537$ 발을 세어 $\hat\lambda = 537/576$ 을 얻었다. 포아송의 최대가능도추정이
표본평균이라는 정리 1 이 그 나눗셈 한 번이다.

① 클라크의 표를 다시 계산한다 — 인쇄된 기대도수 여섯이 모두 그대로 나온다.
② 뭉쳤다면 표가 어떻게 달라졌을까 — 양 끝이 부풀고 가운데가 꺼진다.
③ 그런데 맞은 것은 분포가 아니라 창이었다 — 창을 넓히면 적합이 무너진다.

만드는 파일: docs/ch06/mle/img/flying_bomb_poisson.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(7)
N_SQ, N_BOMB = 576, 537        # 클라크의 칸 수와 폭탄 수
SIDE_KM = 0.5                  # 0.25 km^2 칸의 한 변
GRID = 24                      # 24 x 24 = 576
KMAX = 5                       # 「5 이상」으로 묶는다
OBS = np.array([229, 211, 93, 35, 7, 1])   # 클라크가 인쇄한 실제 칸 수


def expected(n, lam):
    k = np.arange(KMAX)
    return n * np.r_[stats.poisson.pmf(k, lam), stats.poisson.sf(KMAX - 1, lam)]


def gof(counts):
    """칸별 폭탄 수 -> (lam_hat, 도수, 기대도수, chi2, df, p)."""
    n = len(counts)
    obs = np.array([(counts == k).sum() for k in range(KMAX)]
                   + [(counts >= KMAX).sum()], float)
    lam = counts.mean()
    exp = expected(n, lam)
    keep = exp >= 1.0                      # 기대도수가 너무 작은 칸은 묶는다
    o = np.r_[obs[keep], obs[~keep].sum()]
    e = np.r_[exp[keep], exp[~keep].sum()]
    if e[-1] < 1.0:                        # 꼬리를 앞 칸에 합친다
        o = np.r_[o[:-2], o[-2:].sum()]
        e = np.r_[e[:-2], e[-2:].sum()]
    chi2 = float(((o - e) ** 2 / e).sum())
    df = len(o) - 2                        # 모수 하나를 추정했다
    return lam, obs, exp, chi2, df, float(stats.chi2.sf(chi2, df))


# === ① 클라크의 표를 다시 계산 =======================================
LAM_C = N_BOMB / N_SQ
EXP_C = expected(N_SQ, LAM_C)
CHI_C = float(((OBS - EXP_C) ** 2 / EXP_C).sum())
P_C = float(stats.chi2.sf(CHI_C, 4))
PRINTED = np.array([226.74, 211.39, 98.54, 30.62, 7.14, 1.57])
print("① 클라크 1946 의 표를 다시 계산한다")
print(f"   lambda_hat = {N_BOMB}/{N_SQ} = {LAM_C:.6f}")
print("   칸당 폭탄 수   실제   기대(인쇄)   기대(다시 계산)")
for k in range(6):
    nm = "5 이상" if k == KMAX else str(k)
    print(f"   {nm:>10}  {OBS[k]:>5}  {PRINTED[k]:>10.2f}  {EXP_C[k]:>14.2f}")
print(f"   인쇄값과의 최대 차 {np.abs(EXP_C - PRINTED).max():.3f}")
print(f"   chi2 = {CHI_C:.4f} (인쇄 1.17), df = 4, p = {P_C:.4f} (인쇄 .88)")

# === ② 뭉쳤다면 =======================================================
L_KM = GRID * SIDE_KM


def counts_from_points(xy, x0, y0, side_km, grid):
    ix = np.clip(((xy[:, 0] - x0) / side_km).astype(int), 0, grid - 1)
    iy = np.clip(((xy[:, 1] - y0) / side_km).astype(int), 0, grid - 1)
    c = np.zeros((grid, grid), int)
    np.add.at(c, (iy, ix), 1)
    return c.ravel()


def uniform_pattern(n=N_BOMB):
    return RNG.uniform(0, L_KM, (n, 2))


def clustered_pattern(n=N_BOMB, n_cl=55, sd=0.45):
    ctr = RNG.uniform(0, L_KM, (n_cl, 2))
    who = RNG.integers(0, n_cl, n)
    return np.clip(ctr[who] + RNG.normal(0, sd, (n, 2)), 0, L_KM - 1e-9)


print("\n② 같은 576 칸에 같은 537 발을 다르게 떨어뜨리면")
PAT = {}
for nm, f in (("고르게 무작위", uniform_pattern), ("뭉쳐서", clustered_pattern)):
    c = counts_from_points(f(), 0, 0, SIDE_KM, GRID)
    lam, obs, exp, chi2, df, p = gof(c)
    PAT[nm] = obs
    print(f"   {nm:>8} : 0 칸 {obs[0]:>3.0f}  1 칸 {obs[1]:>3.0f}  "
          f"2 칸 {obs[2]:>3.0f}  3 칸 {obs[3]:>3.0f}  4 칸 {obs[4]:>3.0f}  "
          f"5+ {obs[5]:>3.0f}   chi2 = {chi2:>7.1f} (df {df})  p = {p:.4f}")
print(f"   {'클라크 실제':>8} : 0 칸 {OBS[0]:>3}  1 칸 {OBS[1]:>3}  "
      f"2 칸 {OBS[2]:>3}  3 칸 {OBS[3]:>3}  4 칸 {OBS[4]:>3}  5+ {OBS[5]:>3}"
      f"   chi2 = {CHI_C:>7.2f} (df 4)  p = {P_C:.4f}")
print("   뭉치면 양 끝(0 과 5+)이 부풀고 가운데(1~3)가 꺼진다 — "
      "클라크가 적은 바로 그 모양이다.")

# === ③ 창을 넓히면 ====================================================
# 밀도가 한쪽으로 치우친 판을 모의한다. 세 중심은 쇼·쇼(2019)가 지도 전체에
# 혼합정규로 찾은 「세 밀도 중심」을 흉내 낸 것이고 실제 좌표가 아니다.
# 중심이 넓게 퍼져 있어 밀도는 천천히 바뀐다 — 좁은 창에서는 거의 일정하다.
RNG_F = np.random.default_rng(1)
FIELD = 34.0
CENTERS = np.array([[15.0, 15.0], [20.0, 12.0], [16.0, 21.0]])
SD_FIELD = np.array([9.0, 8.0, 9.5])
W_FIELD = np.array([0.45, 0.30, 0.25])
N_FIELD = 2300          # 런던 전체에 떨어진 V-1 의 수에 맞춘 규모


def field_points(n=N_FIELD):
    who = RNG_F.choice(3, n, p=W_FIELD)
    xy = CENTERS[who] + RNG_F.normal(0, 1, (n, 2)) * SD_FIELD[who, None]
    ok = np.all((xy > 0) & (xy < FIELD), axis=1)
    return xy[ok]


XY = field_points()
SIDES = np.array([8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0])
print(f"\n③ 밀도가 치우친 판(모의)에서 창만 넓히며 같은 검정을 되풀이한다")
print(f"{'창 한 변':>8}{'창 넓이':>10}{'칸 수':>7}{'폭탄':>6}"
      f"{'lam_hat':>9}{'분산/평균':>10}{'chi2':>9}{'df':>4}{'p':>9}")
PS, VM, AREAS, LAMS = [], [], [], []
for L in SIDES:
    g = int(round(L / SIDE_KM))
    x0 = y0 = 0.5 * (FIELD - g * SIDE_KM)
    sel = np.all((XY > [x0, y0]) & (XY < [x0 + g * SIDE_KM, y0 + g * SIDE_KM]),
                 axis=1)
    c = counts_from_points(XY[sel], x0, y0, SIDE_KM, g)
    lam, obs, exp, chi2, df, p = gof(c)
    vm = c.var(ddof=1) / c.mean()
    PS.append(p)
    VM.append(vm)
    LAMS.append(lam)
    AREAS.append(g * g * SIDE_KM ** 2)
    print(f"{L:>7.0f}km{AREAS[-1]:>9.0f}㎢{g * g:>7}{sel.sum():>6}"
          f"{lam:>9.3f}{vm:>10.3f}{chi2:>9.1f}{df:>4}{p:>9.4f}")
print("   창이 좁으면 밀도가 거의 일정해 포아송이 맞고, 넓히면 "
      "분산/평균이 1 보다 커지며 적합이 무너진다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_OBS, C_EXP, C_CL = "#1f77b4", "#2ca02c", "#d62728"
LBL = ["$0$", "$1$", "$2$", "$3$", "$4$", "$5$ 이상"]
xs = np.arange(6)

ax1.bar(xs, OBS, 0.62, color=C_OBS, label="실제 칸 수 (클라크의 셈)")
ax1.plot(xs, EXP_C, "o--", color=C_EXP, lw=2.2, ms=9, zorder=4,
         label="포아송 기대 칸 수")
for k in range(6):
    ax1.text(k, max(OBS[k], EXP_C[k]) + 13, f"${OBS[k]}$", ha="center",
             fontsize=10, color=C_OBS)
ax1.set_xticks(xs)
ax1.set_xticklabels(LBL, fontsize=10)
ax1.set_xlabel("칸 하나에 떨어진 폭탄 수")
ax1.set_ylabel("그런 칸의 수 (전체 $576$)")
ax1.set_ylim(0, 268)
ax1.annotate(f"$\\hat\\lambda = 537/576 = {LAM_C:.4f}$\n"
             f"$\\chi^2 = {CHI_C:.2f}$ (자유도 $4$), $p = {P_C:.2f}$",
             xy=(2.6, 215), fontsize=11.5)
ax1.set_title("① 클라크 $1946$ 의 표", fontsize=12)
ax1.legend(loc="upper right", fontsize=9.5, bbox_to_anchor=(1.0, 0.72))
ax1.grid(alpha=0.25, axis="y")

w = 0.27
ax2.bar(xs - w, PAT["고르게 무작위"], w, color=C_EXP, label="고르게 무작위")
ax2.bar(xs, OBS, w, color=C_OBS, label="클라크의 실제 자료")
ax2.bar(xs + w, PAT["뭉쳐서"], w, color=C_CL, label="뭉쳐서 (모의)")
ax2.annotate("뭉치면 양 끝이 부풀고", xy=(0 + w, PAT["뭉쳐서"][0]),
             xytext=(0.55, 300), fontsize=10.5, color=C_CL,
             arrowprops=dict(arrowstyle="->", color=C_CL, lw=1.1))
ax2.annotate("가운데가 꺼진다", xy=(1 + w, PAT["뭉쳐서"][1]),
             xytext=(2.3, 258), fontsize=10.5, color=C_CL,
             arrowprops=dict(arrowstyle="->", color=C_CL, lw=1.1))
ax2.set_xticks(xs)
ax2.set_xticklabels(LBL, fontsize=10)
ax2.set_xlabel("칸 하나에 떨어진 폭탄 수")
ax2.set_ylabel("그런 칸의 수 (전체 $576$)")
ax2.set_ylim(0, 345)
ax2.set_title("② 뭉쳤다면 표가 이렇게 생긴다", fontsize=12)
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.25, axis="y")

FLOOR3 = 1e-6
pp = np.maximum(np.array(PS), FLOOR3)
ax3.semilogy(AREAS, pp, "o-", color=C_OBS, lw=2.6, ms=9,
             label="포아송 적합도검정의 $p$ 값")
for i, (a, pv, lm) in enumerate(zip(AREAS, pp, LAMS)):
    ax3.annotate(f"$\\hat\\lambda = {lm:.2f}$", (a, pv),
                 xytext=(0, -1.45 if i % 2 == 0 else 0.9),
                 textcoords="offset fontsize", fontsize=9.5, ha="center",
                 color="#555555")
ax3.axhline(0.05, color=C_CL, lw=1.6, ls="--")
ax3.text(730, 0.075, "$p = 0.05$", color=C_CL, fontsize=10.5)
ax3.axvline(144, color="#888888", lw=1.4, ls=":")
ax3.text(165, 9.0, "클라크가 고른 $144\\ \\mathrm{km^2}$", fontsize=10.5,
         color="#555555")
ax3.set_xlabel("창의 넓이 ($\\mathrm{km^2}$)")
ax3.set_ylabel("포아송 적합도검정의 $p$ 값")
ax3.set_yticks([1e-6, 1e-4, 1e-2, 1])
ax3.set_yticklabels(["$10^{-6}$ 이하", "$10^{-4}$", "$10^{-2}$", "$1$"])
ax3.minorticks_off()
ax3.set_ylim(2.5e-7, 50)
ax3.set_xlim(0, 1120)
ax3.set_title("③ 창을 넓히면 무너진다 (모의한 판)", fontsize=12)
ax3.legend(loc="lower left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch06/mle/img/flying_bomb_poisson.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
