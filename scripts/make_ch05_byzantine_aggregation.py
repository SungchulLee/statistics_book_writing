r"""공격자는 이상점처럼 굴지 않는다 — 연합학습의 좌표별 집계.

이 쪽 $\S 3$ 의 민감도 곡선은 오염을 **점**으로 본다. 점은 멀리 갈수록 해롭고,
절사는 그 영향에 천장을 씌운다. 그런데 분산 학습에서 오염은 점이 아니라
**규칙을 아는 상대**다. 그러면 두 가지가 달라진다.

① 천장을 치는 데 멀리 갈 필요가 없다 — 정직한 값들 한복판에서 이미 다 친다.
② 좌표마다 조금씩이 $d$ 개 쌓이면 벡터의 오차는 $\sqrt d$ 로 큰다.
③ 그 대가는 공격이 없는 날에도 매번 치른다.

만드는 파일: docs/ch05/mean/img/byzantine_aggregation.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)
N, F, K = 20, 2, 2          # 일꾼 20, 공격자 2, 양 끝에서 2 씩 절사 (10%)
SIG = 1.0


def aggregate(v):
    """v: (..., n). 평균 · 절사평균 · 좌표별 중앙값."""
    s = np.sort(v, axis=-1)
    return {"평균": v.mean(-1),
            f"{100 * K // N}\\% 절사평균": s[..., K:-K].mean(-1),
            "중앙값": np.median(v, -1)}


KEYS = list(aggregate(np.zeros((1, N))))

# === ① 천장은 가까이 있다 ============================================
R1 = 20000
TS = np.r_[0.0, np.logspace(-1.2, 3, 30)]
print(f"① 일꾼 {N}, 공격자 {F}, 정직한 값은 $N(0, 1)$. 공격자는 모두 t 를 보낸다")
print(f"{'t':>9}" + "".join(f"{k.replace(chr(92)+'%','%'):>14}" for k in KEYS))
CURVE = {k: [] for k in KEYS}
for t in TS:
    h = RNG.standard_normal((R1, N - F)) * SIG
    v = np.concatenate([h, np.full((R1, F), t)], axis=1)
    a = aggregate(v)
    for k in KEYS:
        CURVE[k].append(float(np.abs(a[k]).mean()))
for i in (0, 8, 16, 24, len(TS) - 1):
    print(f"{TS[i]:>9.2f}" + "".join(f"{CURVE[k][i]:>14.4f}" for k in KEYS))
CEIL, TSTAR = {}, {}
for k in KEYS[1:]:
    c = max(CURVE[k])
    CEIL[k] = c
    TSTAR[k] = float(TS[int(np.argmax(np.array(CURVE[k]) >= 0.9 * c))])
    print(f"   {k.replace(chr(92) + '%', '%'):>12} : 천장 {c:.4f}, "
          f"그 90% 에 닿는 가장 작은 t = {TSTAR[k]:.2f}")
print(f"   정직한 값의 표준편차가 {SIG} 다 — 공격자는 **정상 범위 안에서** "
      f"이미 할 수 있는 피해를 거의 다 입힌다.")
T_ATT = max(TSTAR.values())

# === ② 좌표마다 조금씩, d 개 ========================================
DS = [1, 3, 10, 30, 100, 300, 1000, 3000]
R2 = 400
print(f"\n② 같은 공격(t = {T_ATT:.2f})을 d 개 좌표에 똑같이 건다")
print(f"{'d':>7}" + "".join(f"{k.replace(chr(92)+'%','%'):>14}" for k in KEYS)
      + f"{'0.26 sqrt(d)':>14}" + f"{'평균/큰거짓말':>14}")
L2 = {k: [] for k in KEYS}
BIGMEAN = []
for d in DS:
    err = {k: 0.0 for k in KEYS}
    for _ in range(R2):
        h = RNG.standard_normal((d, N - F)) * SIG
        v = np.concatenate([h, np.full((d, F), T_ATT)], axis=1)
        a = aggregate(v)
        for k in KEYS:
            err[k] += float(np.linalg.norm(a[k]))
    big = 0.0
    for _ in range(80):                      # 견줌: 같은 자리에 큰 거짓말을 걸면
        h = RNG.standard_normal((d, N - F)) * SIG
        v = np.concatenate([h, np.full((d, F), 1000.0)], axis=1)
        big += float(np.linalg.norm(aggregate(v)[KEYS[0]]))
    BIGMEAN.append(big / 80)
    for k in KEYS:
        L2[k].append(err[k] / R2)
    print(f"{d:>7}" + "".join(f"{L2[k][-1]:>14.3f}" for k in KEYS)
          + f"{0.26 * np.sqrt(d):>14.3f}" + f"{BIGMEAN[-1]:>14.1f}")
print("   세 줄이 모두 sqrt(d) 로 큰다. 절사가 막는 것은 **좌표 하나의** 크기이지 "
      "벡터의 길이가 아니다.")
print(f"   같은 d 에서 큰 거짓말(t = 1000)을 평균에 걸면 d = {DS[-1]} 에서 "
      f"{BIGMEAN[-1]:,.0f} 이다 — 로버스트 규칙이 사 준 것은 **그쪽**이다.")

# === ③ 공격이 없는 날의 값 ===========================================
print(f"\n③ 공격자가 없을 때의 l2 오차 (일꾼 {N} 명 모두 정직)")
print(f"{'d':>7}" + "".join(f"{k.replace(chr(92)+'%','%'):>14}" for k in KEYS)
      + f"{'중앙값/평균':>13}")
CLEAN = {k: [] for k in KEYS}
for d in DS:
    err = {k: 0.0 for k in KEYS}
    for _ in range(R2):
        v = RNG.standard_normal((d, N)) * SIG
        a = aggregate(v)
        for k in KEYS:
            err[k] += float(np.linalg.norm(a[k]))
    for k in KEYS:
        CLEAN[k].append(err[k] / R2)
    print(f"{d:>7}" + "".join(f"{CLEAN[k][-1]:>14.3f}" for k in KEYS)
          + f"{CLEAN[KEYS[2]][-1] / CLEAN[KEYS[0]][-1]:>13.3f}")
rat = np.mean([CLEAN[KEYS[2]][i] / CLEAN[KEYS[0]][i] for i in range(len(DS))])
print(f"   중앙값은 공격이 없어도 평균보다 오차가 {rat:.3f} 배다 "
      f"(정규에서 sqrt(pi/2) = {np.sqrt(np.pi / 2):.3f}).")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
COL = {KEYS[0]: "#d62728", KEYS[1]: "#ff7f0e", KEYS[2]: "#1f77b4"}

for k in KEYS:
    ax1.loglog(np.maximum(TS, 0.05), CURVE[k], "o-", color=COL[k], lw=2.4,
               ms=6, label=k.replace("\\%", "%"))
ax1.axvspan(0.05, 3 * SIG, color="#2ca02c", alpha=0.10)
ax1.text(0.33, 23, "정직한 값이\n놓이는 범위", fontsize=10.5, color="#2ca02c",
         ha="center")
for k in KEYS[1:]:
    ax1.plot([TSTAR[k]], [0.9 * CEIL[k]], "*", color=COL[k], ms=18, zorder=5)
ax1.set_xlabel("공격자가 보내는 값 $t$")
ax1.set_ylabel("집계값이 참값에서 밀린 거리")
ax1.set_xticks([0.1, 1, 10, 100, 1000])
ax1.set_xticklabels(["$0.1$", "$1$", "$10$", "$10^{2}$", "$10^{3}$"])
ax1.set_yticks([0.1, 1, 10, 100])
ax1.set_yticklabels(["$0.1$", "$1$", "$10$", "$10^{2}$"])
ax1.minorticks_off()
ax1.set_ylim(0.07, 300)
ax1.set_title("① 천장은 가까이 있다 ($\\star$ 가 $90\\%$ 지점)", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25)

for k in KEYS:
    ax2.loglog(DS, L2[k], "o-", color=COL[k], lw=2.4, ms=7,
               label=k.replace("\\%", "%"))
ax2.loglog(DS, 0.26 * np.sqrt(DS), "--", color="#666666", lw=1.6,
           label="$0.26\\sqrt{d}$")
ax2.loglog(DS, BIGMEAN, ":", color=COL[KEYS[0]], lw=2.2,
           label="평균 — 큰 거짓말 $t = 1000$")
ax2.set_xlabel("좌표의 수 $d$")
ax2.set_ylabel("집계 벡터의 $\\ell_2$ 오차")
ax2.set_xticks([1, 10, 100, 1000])
ax2.set_xticklabels(["$1$", "$10$", "$10^{2}$", "$10^{3}$"])
ax2.set_yticks([0.1, 1, 10, 100, 1e4])
ax2.set_yticklabels(["$0.1$", "$1$", "$10$", "$10^{2}$", "$10^{4}$"])
ax2.minorticks_off()
ax2.set_title(f"② 좌표마다 조금씩, $d$ 개 ($t = {T_ATT:.1f}$)", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25)

for k in KEYS:
    ax3.loglog(DS, CLEAN[k], "o-", color=COL[k], lw=2.4, ms=7,
               label=k.replace("\\%", "%"))
ax3.set_xlabel("좌표의 수 $d$")
ax3.set_ylabel("집계 벡터의 $\\ell_2$ 오차")
ax3.set_xticks([1, 10, 100, 1000])
ax3.set_xticklabels(["$1$", "$10$", "$10^{2}$", "$10^{3}$"])
ax3.set_yticks([0.1, 1, 10])
ax3.set_yticklabels(["$0.1$", "$1$", "$10$"])
ax3.minorticks_off()
ax3.annotate(f"중앙값이 평균보다\n꾸준히 ${rat:.2f}$ 배",
             xy=(0.04, 0.93), xycoords="axes fraction", fontsize=11,
             color=COL[KEYS[2]], va="top")
ax3.set_title("③ 공격이 없는 날에도 치르는 값", fontsize=12)
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch05/mean/img/byzantine_aggregation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
