r"""17.1절 「그림으로 보는 통계학」 — 컴퓨터 없이 재표집한 사람.

마할라노비스는 $1937$ 년부터 벵골의 황마 작황을 표본조사하면서 한 구역에 뿌릴
조사단위를 **두 묶음으로 갈라 서로 엇갈리게**(interpenetrating sub-samples) 뿌렸다.
두 묶음이 각각 독립인 추정값을 주므로 **그 차이 하나가 오차의 폭**이 된다. 공식이
아니라 자료를 쪼개서 오차를 재는 것 — 붓스트랩이 뒤에 하게 되는 일이다.

① 한 구역 — 면적 $50$ 제곱마일에 $4$ 에이커 단위를 제곱마일당 둘, 두 묶음으로.
② 두 묶음의 차이가 공식이 못 보는 것을 본다 — 한 조사단이 치우쳤을 때.
③ 논에서 던진 동전 — 기록이 진짜 동전인지 런 검정으로 묻는다.

만드는 파일: docs/ch17/bootstrap/img/interpenetrating_samples.png
실행:  python3 scripts/make_ch17_interpenetrating.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy.ndimage import gaussian_filter

OUT = "docs/ch17/bootstrap/img/interpenetrating_samples.png"
RNG = np.random.default_rng(20261022)

# --- 구역 하나를 만든다 -------------------------------------------------
# 비하르에서 쓴 설계: 구역은 약 50 제곱마일, 조사단위는 4 에이커, 밀도는
# 제곱마일당 둘. 곧 구역마다 단위 100 개, 한 묶음에 50 개다.
GRID = 256                      # 구역을 256 x 256 격자로 본다
ZONE_MI = np.sqrt(50.0)         # 한 변 약 7.07 마일
N_UNIT, N_HALF = 100, 50

field = gaussian_filter(RNG.standard_normal((GRID, GRID)), sigma=18, mode="wrap")
field = 0.35 + 0.12 * field / field.std()      # 황마가 심긴 땅의 비율로 읽는다
field = np.clip(field, 0.02, 0.95)
THETA = field.mean()                            # 우리가 맞히려는 구역 평균


def place(rng, n):
    """구역 안에 조사단위 n 개를 무작위로 뿌리고 격자 좌표를 돌려준다."""
    return rng.integers(0, GRID, (n, 2))


def read(idx, bias=0.0, rng=None, sd_obs=0.02):
    """조사단이 적어 온 값 — 기록 잡음과 (있다면) 조사단의 치우침이 섞인다."""
    v = field[idx[:, 0], idx[:, 1]] + bias
    if rng is not None:
        v = v + sd_obs * rng.standard_normal(len(v))
    return v


# --- 참 표준오차를 모의로 잰다 (공식을 쓰지 않는다) ---------------------
REPS = 20_000
th = np.empty(REPS)
for r in range(REPS):
    th[r] = read(place(RNG, N_UNIT), rng=RNG).mean()
SE_TRUE = th.std(ddof=1)
print(f"구역 평균 θ = {THETA:.4f},  n = {N_UNIT} 일 때 참 SE = {SE_TRUE:.5f}")

# --- ② 치우침을 키우며 두 자를 견준다 ----------------------------------
BIAS = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
REPS2 = 4_000
half_se, iid_se, alarm = [], [], []

# 치우침이 없을 때 |d| 가 얼마나 벌어지는지를 먼저 재서 경보선을 잡는다
d0 = np.empty(REPS2)
for r in range(REPS2):
    a, b = place(RNG, N_HALF), place(RNG, N_HALF)
    d0[r] = read(a, rng=RNG).mean() - read(b, rng=RNG).mean()
GATE = np.quantile(np.abs(d0), 0.95)
print(f"치우침이 없을 때 |d| 의 95% 분위수 = {GATE:.5f}")

for b_ in BIAS:
    hs, ii, hit = np.empty(REPS2), np.empty(REPS2), np.empty(REPS2, bool)
    for r in range(REPS2):
        v1 = read(place(RNG, N_HALF), rng=RNG)
        v2 = read(place(RNG, N_HALF), bias=b_, rng=RNG)
        d = v1.mean() - v2.mean()
        hs[r] = abs(d) / 2                       # 마할라노비스의 오차 폭
        ii[r] = np.concatenate([v1, v2]).std(ddof=1) / np.sqrt(N_UNIT)
        hit[r] = abs(d) > GATE
    half_se.append(hs.mean())
    iid_se.append(ii.mean())
    alarm.append(hit.mean())
    print(f"  치우침 {b_:.2f}:  |d|/2 = {hs.mean():.5f}  "
          f"s/√n = {ii.mean():.5f}  경보율 = {hit.mean():.3f}")

half_se = np.array(half_se) / SE_TRUE
iid_se = np.array(iid_se) / SE_TRUE
alarm = np.array(alarm)

# --- ③ 논에서 던진 동전 -------------------------------------------------
# 조사원은 던진 결과를 차례로 적어야 했다. 적은 것이 정말 동전이면 런의 수는
# 평균 (n+1)/2 를 따른다. 긴 연속을 피해 "그럴듯하게" 지어내면 런이 많아진다.
N_TOSS, REPS3 = 100, 200_000


def runs(seq):
    return 1 + int(np.sum(seq[1:] != seq[:-1]))


real = np.array([runs(RNG.integers(0, 2, N_TOSS)) for _ in range(REPS3)])


def fabricate(rng, n, cap=2):
    """같은 글자가 cap 개 넘게 이어지지 않게 지어낸 기록."""
    s = np.empty(n, int)
    s[0] = rng.integers(0, 2)
    run = 1
    for i in range(1, n):
        s[i] = s[i - 1] if (run < cap and rng.random() < 0.5) else 1 - s[i - 1]
        run = run + 1 if s[i] == s[i - 1] else 1
    return s


fake = np.array([runs(fabricate(RNG, N_TOSS)) for _ in range(20_000)])
lo, hi = np.quantile(real, [0.025, 0.975])
power = float(np.mean((fake < lo) | (fake > hi)))
print(f"런의 수 — 진짜 동전: 평균 {real.mean():.2f}, 표준편차 {real.std(ddof=1):.2f}, "
      f"가운데 95% = [{lo:.0f}, {hi:.0f}]")
print(f"지어낸 기록: 평균 {fake.mean():.2f},  적발률 = {power:.3f}")

# --- 그림 ---------------------------------------------------------------
fig = plt.figure(figsize=(15.0, 4.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.30)
C1, C2 = "#1f6f8b", "#c0532b"

# ① 구역과 두 묶음
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(field, cmap="YlGn", origin="lower",
               extent=[0, ZONE_MI, 0, ZONE_MI], vmin=0.05, vmax=0.7)
A, B = place(RNG, N_HALF), place(RNG, N_HALF)
sc = ZONE_MI / GRID
ax.scatter(A[:, 1] * sc, A[:, 0] * sc, s=42, marker="o",
           facecolor="none", edgecolor=C1, linewidth=1.6, label="첫째 묶음 50")
ax.scatter(B[:, 1] * sc, B[:, 0] * sc, s=46, marker="^",
           facecolor="none", edgecolor=C2, linewidth=1.6, label="둘째 묶음 50")
ax.set_title("① 한 구역 50 제곱마일 — 엇갈리게 뿌린 두 묶음", fontsize=11)
ax.set_xlabel("마일")
ax.set_ylabel("마일")
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("황마가 심긴 땅의 비율", fontsize=9)
cb.ax.tick_params(labelsize=8)

e1 = read(A, rng=RNG).mean()
e2 = read(B, rng=RNG).mean()
ax.text(0.03, 0.03,
        f"두 묶음의 추정값  {e1:.3f} / {e2:.3f}\n차이 {abs(e1-e2):.3f}"
        f"  →  오차 폭 {abs(e1-e2)/2:.3f}",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

# ② 두 자
ax = fig.add_subplot(gs[0, 1])
bx = BIAS * 100
ax.plot(bx, half_se, "o-", color=C1, lw=2, ms=6,
        label="두 묶음의 차이  |d| / 2")
ax.plot(bx, iid_se, "s-", color=C2, lw=2, ms=6,
        label="독립을 가정한 공식  s / √n")
ax.axhline(1.0, color="0.45", ls=":", lw=1.2)
ax.text(bx[-1], 1.04, "참 표준오차", color="0.35", fontsize=9,
        ha="right", va="bottom")
ax2 = ax.twinx()
ax2.plot(bx, alarm * 100, "--", color="0.45", lw=1.5,
         label="|d| 가 경보선을 넘는 비율 (오른쪽 축)")
ax2.set_ylabel("경보율 (%)", color="0.35", fontsize=10)
ax2.tick_params(axis="y", labelcolor="0.35", labelsize=9)
ax2.set_ylim(0, 105)
ax.set_xlabel("둘째 조사단의 치우침 (백분율점)")
ax.set_ylabel("추정한 오차 ÷ 참 표준오차")
ax.set_title("② 한 조사단이 치우치면 — 공식은 거의 못 본다", fontsize=11)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)
ax.set_ylim(0, max(half_se) * 1.18)

# ③ 런 검정
ax = fig.add_subplot(gs[0, 2])
bins = np.arange(real.min() - 0.5, max(real.max(), fake.max()) + 1.5)
ax.hist(real, bins=bins, color=C1, alpha=0.75, density=True,
        label="진짜 동전 100 번")
ax.hist(fake, bins=bins, color=C2, alpha=0.6, density=True,
        label="긴 연속을 피해 지어낸 기록")
ax.axvline(lo, color="0.3", ls="--", lw=1.2)
ax.axvline(hi, color="0.3", ls="--", lw=1.2)
ax.text(hi + 0.7, ax.get_ylim()[1] * 0.92,
        f"가운데 95%\n[{lo:.0f}, {hi:.0f}]", fontsize=9, color="0.3")
ax.set_xlabel("기록에 나타난 런의 수")
ax.set_ylabel("밀도")
ax.set_title(f"③ 기록이 동전인지 묻는다 — 적발률 {power*100:.0f}%", fontsize=11)
ax.legend(loc="upper left", fontsize=9)

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"저장: {OUT}")
