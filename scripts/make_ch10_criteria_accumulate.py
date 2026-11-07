"""11.5절 인과 판정 기준 — 목록이 어떻게 증거가 되는가.

들머리의 물음을 그린다. 「어느 하나도 필요조건도 충분조건도 아닌 목록이
어떻게 인과의 증거가 될 수 있는가.」 답은 **증거가 곱해지기 때문**이다.

이 그림은 수치를 **가정하고** 그 가정에서 따라 나오는 것만 보인다. 기준
하나가 충족될 확률을 인과일 때 0.7, 교란일 때 0.3 으로 두면 우도비가
7/3 = 2.333 이다. 하나로는 사후확률이 거의 움직이지 않고(충분조건이
아니다), 몇 개를 놓쳐도 결론이 죽지 않는다(필요조건이 아니다). 그런데
여덟 가운데 여섯이 맞으면 같은 작은 수들이 쌓여 사후확률이 크게 올라간다.

오른쪽은 그 셈이 기대는 가정을 드러낸다. 기준들이 서로 **독립일 때만**
우도비가 그대로 곱해진다. 현실에서는 겹치므로 「사실상 몇 개가 독립인가」를
줄이면 같은 여섯이 훨씬 적게 움직인다. 쪽이 「충족되는 기준이 많을수록
논거가 강해진다」고만 적어 둔 문장에 크기를 붙이는 그림이다.

시간성은 따로 둔다 --- 쪽의 주석대로 아홉 가운데 그것 하나만 논리적
필요조건이라, 저울이 아니라 문이다. 깨지면 사후확률이 0 이 된다.

쪽의 cornfield 는 교란변수에게 요구되는 세기를, potential_outcomes_gap 은
평균 뒤에 숨은 개인차를 그린다. 여기서는 그 둘이 건드리지 않는 것 ---
목록을 읽는 산술 --- 만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch11/causation/img/"

P_CAUSE = 0.7        # 인과일 때 기준 하나가 충족될 확률 (가정)
P_CONF = 0.3         # 교란일 때 (가정)
K = 8                # 시간성을 뺀 저울 여덟
LR = P_CAUSE / P_CONF
LR_MISS = (1 - P_CAUSE) / (1 - P_CONF)


def posterior_checked(prior, met, checked, eff=None):
    """checked 개를 확인해 그 가운데 met 개가 충족되었을 때의 사후확률.

    확인하지 않은 기준은 아무 증거도 주지 않는다.
    """
    w = 1.0 if eff is None else eff / K
    odds = prior / (1 - prior)
    odds *= LR ** (met * w) * LR_MISS ** ((checked - met) * w)
    return odds / (1 + odds)


def posterior(prior, met, total=K, eff=None):
    """met 개가 충족되고 나머지가 어긋났을 때의 사후확률.

    eff 를 주면 「사실상 독립인 기준의 수」가 그만큼이라고 보고 지수를
    eff/total 로 줄인다. 기준들이 겹치면 증거가 그만큼 덜 쌓인다.
    """
    w = 1.0 if eff is None else eff / total
    odds = prior / (1 - prior)
    odds *= LR ** (met * w) * LR_MISS ** ((total - met) * w)
    return odds / (1 + odds)


ks = np.arange(0, K + 1)
PRIORS = [(0.10, "열에 하나쯤이라 보았다면", "tab:blue"),
          (0.30, "셋에 하나쯤", "tab:green"),
          (0.50, "반반", "tab:red")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.9))

# ── (좌) 작은 수가 여덟 번 곱해지면 ────────────────────────────────
for pr, name, col in PRIORS:
    ax1.plot(ks, [100 * posterior(pr, k) for k in ks], "o-", ms=7, lw=2.4,
             color=col, label=f"{name} (사전 {100*pr:.0f}%)")
p1 = posterior_checked(0.30, 1, 1)
ax1.plot([1], [100 * p1], "*", ms=18, color="0.35", zorder=5)
ax1.text(1.0, 36, f"기준 하나만 확인했다면 {100*p1:.0f}%\n"
         "— 충분조건이 아니다", ha="center", va="top", fontsize=10.5,
         color="0.3", fontweight="bold")
p36 = posterior(0.30, 6)
ax1.annotate(f"여덟 중 둘이 어긋나도 {100*p36:.1f}%\n— 필요조건이 아니다",
             xy=(6, 100 * p36), xytext=(1.9, 68), fontsize=10.5,
             color="tab:green", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.6))
ax1.set_xticks(ks)
ax1.set_xlim(-0.3, K + 0.3)
ax1.set_ylim(0, 102)
ax1.set_xlabel(f"저울에 올린 기준 {K} 개 가운데 충족된 수", fontsize=11)
ax1.set_ylabel("인과라고 볼 확률 (%)", fontsize=11)
ax1.set_title(f"기준 하나의 우도비는 {LR:.3f} 뿐이다", fontsize=12.5)
ax1.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
ax1.grid(alpha=0.25)

# ── (우) 그 셈이 기대는 가정 ───────────────────────────────────────
effs = np.linspace(1, K, 120)
for pr, name, col in PRIORS:
    ax2.plot(effs, [100 * posterior(pr, 6, eff=e) for e in effs], "-",
             lw=2.4, color=col, label=f"사전 {100*pr:.0f}%")
for e, lab in ((K, "여덟이 모두 따로 논다"), (3.0, "사실상 셋뿐이라면")):
    ax2.axvline(e, color="0.4", ls="--", lw=1.5)
    v = 100 * posterior(0.30, 6, eff=e)
    ax2.plot([e], [v], "o", ms=11, mfc="none", mec="tab:green", mew=2.4)
    top = e > K - 1
    ax2.text(e - 0.1 if top else e + 0.1, 30 if top else 6, lab,
             ha="right" if top else "left", va="bottom",
             fontsize=10.5, color="0.3", fontweight="bold")
ax2.text(0.5, 0.97, f"여덟 가운데 여섯이 충족된 같은 자료", ha="center",
         va="top", transform=ax2.transAxes, fontsize=11.5, fontweight="bold",
         color="0.2",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))
ax2.set_xlim(0.8, K + 0.2)
ax2.set_ylim(0, 102)
ax2.set_xlabel("사실상 독립인 기준의 수", fontsize=11)
ax2.set_ylabel("인과라고 볼 확률 (%)", fontsize=11)
ax2.set_title("기준들이 겹치면 증거는 그만큼 덜 쌓인다", fontsize=12.5)
ax2.legend(fontsize=10, loc="lower right", framealpha=0.95)
ax2.grid(alpha=0.25)

fig.suptitle("하나도 필요·충분조건이 아닌데 목록이 증거가 되는 까닭 — "
             "증거는 더해지지 않고 곱해진다", fontsize=13.5, y=1.0)
plt.tight_layout()
path = OUT + "criteria_accumulate.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n가정: 기준 하나가 충족될 확률 — 인과일 때 {P_CAUSE}, 교란일 때 "
      f"{P_CONF}")
print(f"  충족의 우도비 {LR:.4f} · 불일치의 우도비 {LR_MISS:.4f}")
print(f"\n사전 30% 에서, 저울 {K} 개 가운데 충족된 수에 따른 사후확률")
for k in ks:
    print(f"  {k} 개  →  {100 * posterior(0.30, k):6.2f}%")
print(f"\n확인한 기준이 모두 충족된 경우 (사전 30%)")
for k in ks:
    print(f"  {k} 개 확인  →  {100 * posterior_checked(0.30, k, k):6.2f}%")
print("\n여섯이 충족된 같은 자료에서, 사실상 독립인 기준의 수를 줄이면")
for e in (8, 6, 5, 4, 3, 2, 1):
    print(f"  {e} 개가 독립  →  사전 30% 에서 "
          f"{100 * posterior(0.30, 6, eff=e):6.2f}%")
