r"""교호작용이 있느냐 없느냐가 **척도에 달려 있고**, 그 선택이 정책을 가른다.

Hammond · Selikoff · Seidman($1979$)이 보온공들에게서 얻은 폐암 사망비 네 개를
그대로 쓴다.

|            | 비흡연 | 흡연  |
|------------|------|------|
| 석면 없음   | $1.00$ | $10.85$ |
| 석면 있음   | $5.17$ | $53.24$ |

① 원 척도에서는 선이 크게 벌어진다 — 가법 기대값 $15.02$ 에 견주어 $3.5$ 배다.
② 로그 척도에서는 거의 평행하다 — 곱셈 기대값 $56.09$ 에 견주어 $0.95$ 배다.
③ 그런데 막을 수 있는 **사망 수**는 원 척도가 말한다.

만드는 파일: docs/ch11/anova_two_way/img/asbestos_scale.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

# 논문이 보고한 사망비와, 「둘 다」 집단의 십만 인년당 폐암 사망률
RR = np.array([[1.00, 10.85],      # 석면 없음 : 비흡연, 흡연
               [5.17, 53.24]])     # 석면 있음 : 비흡연, 흡연
RATE_BOTH = 601.6                  # 십만 인년당
BASE = RATE_BOTH / RR[1, 1]        # 「둘 다 아님」의 절대 사망률

add_exp = 1 + (RR[1, 0] - 1) + (RR[0, 1] - 1)
mul_exp = RR[1, 0] * RR[0, 1]
print(f"관측된 「둘 다」 사망비 {RR[1, 1]:.2f}")
print(f"  가법 기대값 {add_exp:.2f}  ->  관측이 {RR[1, 1] / add_exp:.2f} 배")
print(f"  곱셈 기대값 {mul_exp:.2f}  ->  관측이 {RR[1, 1] / mul_exp:.3f} 배")
print(f"\n기준 사망률 {BASE:.2f} /십만 인년 (「둘 다」 {RATE_BOTH} 에서 되짚음)")

# 십만 인년당 막을 수 있는 사망 수
asb_nonsmk = (RR[1, 0] - RR[0, 0]) * BASE
asb_smk = (RR[1, 1] - RR[0, 1]) * BASE
smk_noasb = (RR[0, 1] - RR[0, 0]) * BASE
smk_asb = (RR[1, 1] - RR[1, 0]) * BASE
print(f"\n십만 인년당 막을 수 있는 폐암 사망")
print(f"  석면을 없애면 : 비흡연 {asb_nonsmk:.0f}, 흡연 {asb_smk:.0f} "
      f"({asb_smk / asb_nonsmk:.1f} 배)")
print(f"  담배를 끊으면 : 비석면 {smk_noasb:.0f}, 석면 {smk_asb:.0f} "
      f"({smk_asb / smk_noasb:.1f} 배)")

# 로그 척도의 교호작용 대비
psi_raw = (RR[1, 1] - RR[1, 0]) - (RR[0, 1] - RR[0, 0])
psi_log = (np.log(RR[1, 1]) - np.log(RR[1, 0])) \
    - (np.log(RR[0, 1]) - np.log(RR[0, 0]))
print(f"\n교호작용 대비  원 척도 {psi_raw:+.2f},  로그 척도 {psi_log:+.4f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
xs = np.array([0, 1])
LBL = ["비흡연", "흡연"]
C0, C1 = "#1f77b4", "#d62728"

for ax, scale in ((ax1, "raw"), (ax2, "log")):
    ax.plot(xs, RR[0], "o-", color=C0, lw=2.8, ms=10, label="석면 없음")
    ax.plot(xs, RR[1], "s-", color=C1, lw=2.8, ms=10, label="석면 있음")
    ax.set_xticks(xs)
    ax.set_xticklabels(LBL, fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.25, 1.35)

for i in (0, 1):
    for j in (0, 1):
        ax1.annotate(f"${RR[i, j]:.2f}$", (j, RR[i, j]),
                     textcoords="offset points", xytext=(8, -4),
                     fontsize=11, color=C0 if i == 0 else C1)
ax1.plot([0, 1], [RR[1, 0], RR[1, 0] + (RR[0, 1] - RR[0, 0])], ":",
         color="#555555", lw=2.0)
ax1.annotate(f"가법이라면 ${add_exp:.2f}$", xy=(1, add_exp),
             xytext=(1.03, 24), fontsize=10.5, color="#555555",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax1.set_ylabel("폐암 사망비")
ax1.set_title("① 원 척도 — 선이 벌어진다", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)

ax2.set_yscale("log")
ax2.set_yticks([1, 2, 5, 10, 20, 50])
ax2.set_yticklabels(["$1$", "$2$", "$5$", "$10$", "$20$", "$50$"])
ax2.minorticks_off()
ax2.plot([0, 1], [RR[1, 0], RR[1, 0] * (RR[0, 1] / RR[0, 0])], ":",
         color="#555555", lw=2.0)
ax2.annotate(f"곱셈이라면 ${mul_exp:.2f}$\n(관측 ${RR[1, 1]:.2f}$)",
             xy=(1, mul_exp), xytext=(0.12, 22), fontsize=10.5,
             color="#555555",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax2.set_ylabel("폐암 사망비 (로그 눈금)")
ax2.set_title("② 로그 척도 — 거의 평행하다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)

w = 0.36
pos = np.arange(2)
ax3.bar(pos - w / 2, [asb_nonsmk, smk_noasb], w, color="#9ecae1",
        label="다른 노출이 없을 때")
ax3.bar(pos + w / 2, [asb_smk, smk_asb], w, color="#d62728",
        label="다른 노출이 있을 때")
for p, lo, hi in zip(pos, [asb_nonsmk, smk_noasb], [asb_smk, smk_asb]):
    ax3.text(p - w / 2, lo + 12, f"${lo:.0f}$", ha="center", fontsize=11)
    ax3.text(p + w / 2, hi + 12, f"${hi:.0f}$", ha="center", fontsize=11)
    ax3.text(p, hi + 58, f"${hi / lo:.1f}$ 배", ha="center", fontsize=11.5,
             color="#d62728")
ax3.set_xticks(pos)
ax3.set_xticklabels(["석면을 없애면", "담배를 끊으면"], fontsize=11)
ax3.set_ylim(0, max(asb_smk, smk_asb) * 1.28)
ax3.set_ylabel("십만 인년당 막을 수 있는 폐암 사망")
ax3.set_title("③ 사람 수는 원 척도가 말한다", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch11/anova_two_way/img/asbestos_scale.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
