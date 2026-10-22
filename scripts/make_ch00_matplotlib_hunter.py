r"""뇌파를 보려고 만든 것이 이 책의 그림을 다 그렸다 — 존 헌터와 matplotlib.

헌터는 뇌파 분석 프로그램을 MATLAB 으로 짜다 한계에 부딪혀 Python 으로 옮겼고,
마음에 드는 2D 작도 꾸러미가 없어 직접 만들었다. 그래픽스 경험이 없어서 MATLAB 을
흉내 냈는데, **그 흉내가 설계였다** — 사용자 인터페이스를 고정해 두면 속을 다시
설계해도 남의 코드가 깨지지 않는다.

① 그가 보던 자료의 모양 (모의한 뇌파다 — 실제 기록이 아니다).
② 이 책의 그림을 장별로 센다.
③ 이 책이 두 인터페이스를 어떻게 나누어 쓰는지 센다.

②③ 은 이 저장소를 **돌릴 때마다 다시 세므로** 그림의 수는 책이 자라면 바뀐다.

만드는 파일: docs/ch00/tools/img/matplotlib_hunter.png
"""

import collections
import pathlib
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(3)

# === ① 모의한 뇌파 ====================================================
FS, SEC, NCH = 256, 8.0, 8
T = np.arange(0, SEC, 1 / FS)


def eeg_channel(seed, burst):
    """1/f 바탕 + 알파리듬. burst 구간에는 3 Hz 극서파 같은 것을 얹는다."""
    r = np.random.default_rng(seed)
    n = len(T)
    f = np.fft.rfftfreq(n, 1 / FS)
    amp = 1.0 / np.maximum(f, 0.5)                 # 1/f 바탕
    ph = r.uniform(0, 2 * np.pi, len(f))
    x = np.fft.irfft(amp * np.exp(1j * ph), n)
    x = x / x.std()
    x += 0.35 * np.sin(2 * np.pi * 10 * T + r.uniform(0, 6))   # 알파 10 Hz
    if burst is not None:
        a, b = burst
        m = (T >= a) & (T <= b)
        env = np.sin(np.pi * (T[m] - a) / (b - a)) ** 2
        sw = (np.sin(2 * np.pi * 3 * T[m]) ** 7
              + 0.6 * np.sin(2 * np.pi * 6 * T[m] + 1.2))
        x[m] += 3.2 * env * sw
    return x


BURST = (3.1, 5.6)
CH = [eeg_channel(10 + i, BURST if i in (1, 2, 3, 5) else None)
      for i in range(NCH)]
print(f"① 모의한 뇌파 {NCH} 채널 · {SEC:.0f} 초 · {FS} Hz "
      f"(발작 비슷한 구간 {BURST[0]}–{BURST[1]} 초, 채널 넷에만)")

# === ② 이 책의 그림을 센다 ============================================
by_ch = collections.Counter()
for p in pathlib.Path("docs").rglob("img/*.png"):
    m = re.match(r"ch(\d+)", p.relative_to("docs").parts[0])
    if m:
        by_ch[int(m.group(1))] += 1
TOTAL_PNG = sum(by_ch.values())
N_SCRIPT = len(list(pathlib.Path("scripts").glob("make_*.py")))
print(f"\n② 그림 {TOTAL_PNG:,} 장 · 그리는 스크립트 {N_SCRIPT} 개")
print("   " + "  ".join(f"ch{c:02d}:{by_ch[c]}" for c in sorted(by_ch)))

# === ③ 두 인터페이스 ==================================================
src = []
for p in pathlib.Path("scripts").glob("*.py"):
    src.append(p.read_text())
for p in pathlib.Path("docs").rglob("*.md"):
    src.append(p.read_text())
SRC = "\n".join(src)
PLT = collections.Counter(re.findall(r"\bplt\.([a-z_]+)\(", SRC))
OO = collections.Counter(
    re.findall(r"\b(?:ax|axes|axs|fig)[\w\[\]0-9, ]*\.([a-z_]+)\(", SRC))
N_PLT, N_OO = sum(PLT.values()), sum(OO.values())
print(f"\n③ plt.* 를 부른 횟수 {N_PLT:,} · ax·fig.* 를 부른 횟수 {N_OO:,} "
      f"(객체 쪽이 {N_OO / N_PLT:.1f} 배)")
print("   plt 쪽 상위 다섯 :", ", ".join(f"{k} {v:,}"
                                     for k, v in PLT.most_common(5)))
print("   ax 쪽 상위 다섯  :", ", ".join(f"{k} {v:,}"
                                     for k, v in OO.most_common(5)))
FRAME = ["subplots", "show", "close", "tight_layout", "figure"]
share = sum(PLT[k] for k in FRAME) / N_PLT
print(f"   plt 호출 가운데 '틀 만들고 닫기'({', '.join(FRAME)})가 "
      f"{100 * share:.1f}%")
PAIRS = [("plot", "plot"), ("title", "set_title"), ("xlabel", "set_xlabel"),
         ("ylabel", "set_ylabel"), ("legend", "legend"), ("grid", "grid")]
print(f"\n   {'하는 일':>8}{'plt 쪽':>10}{'ax 쪽':>10}")
for a, b in PAIRS:
    print(f"   {a:>8}{PLT[a]:>10,}{OO[b]:>10,}")

# === 그림 ============================================================
fig = plt.figure(figsize=(15.6, 5.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.28)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
C_A, C_B = "#1f77b4", "#d62728"

STEP = 6.0
for i, x in enumerate(CH):
    col = C_B if i in (1, 2, 3, 5) else "#444444"
    ax1.plot(T, x - i * STEP, color=col, lw=0.7)
    ax1.text(-0.35, -i * STEP, f"$\\mathrm{{ch}}_{{{i + 1}}}$", fontsize=9,
             ha="right", va="center", color=col)
ax1.axvspan(*BURST, color=C_B, alpha=0.09)
ax1.annotate("발작 비슷한 구간", xy=(4.9, -STEP + 3.0), xytext=(5.9, 5.0),
             fontsize=10.5, color=C_B,
             arrowprops=dict(arrowstyle="->", color=C_B, lw=1.1))
ax1.set_xlim(-1.1, SEC)
ax1.set_ylim(-STEP * NCH, 7.5)
ax1.set_yticks([])
ax1.set_xlabel("시간 (초)")
ax1.set_title("① 그가 보던 자료 (모의한 뇌파)", fontsize=12)
for side in ("left", "right", "top"):
    ax1.spines[side].set_visible(False)

chs = sorted(by_ch)
ax2.bar([f"{c:02d}" for c in chs], [by_ch[c] for c in chs], 0.72, color=C_A)
ax2.set_xticks(range(len(chs)))
ax2.set_xticklabels([f"${c}$" for c in chs], fontsize=8.5)
ax2.set_xlabel("장")
ax2.set_ylabel("그림(PNG)의 수")
ax2.annotate(f"모두 ${TOTAL_PNG:,}$ 장".replace(",", "{,}")
             + f"\n스크립트 ${N_SCRIPT}$ 개",
             xy=(0.5, 0.95), xycoords="axes fraction", fontsize=11.5,
             ha="center", va="top", color="#333333")
ax2.set_ylim(0, max(by_ch.values()) * 1.3)
ax2.set_title("② 이 책의 그림", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

labs = [f"{a}\n/ {b}" if a != b else a for a, b in PAIRS]
xs = np.arange(len(PAIRS))
w = 0.38
def money(v):
    """mathtext 안에서 쉼표를 제대로 찍으려면 $...$ 안에 {,} 로 넣어야 한다."""
    return "$" + f"{v:,}".replace(",", "{,}") + "$"


ax3.bar(xs - w / 2, [PLT[a] for a, _ in PAIRS], w, color="#ff7f0e",
        label=f"$\\mathtt{{plt.}}$ — MATLAB 투 (모두 {money(N_PLT)})")
ax3.bar(xs + w / 2, [OO[b] for _, b in PAIRS], w, color=C_A,
        label=f"$\\mathtt{{ax.}}$ — 객체 쪽 (모두 {money(N_OO)})")
ax3.set_yscale("log")
ax3.set_xticks(xs)
ax3.set_xticklabels(labs, fontsize=8.5)
ax3.set_ylabel("이 책에서 부른 횟수")
ax3.set_yticks([1, 10, 100, 1000])
ax3.set_yticklabels(["$1$", "$10$", "$10^{2}$", "$10^{3}$"])
ax3.minorticks_off()
ax3.set_ylim(0.6, 2e4)
ax3.set_title("③ 열어 둔 문과 들어간 방", fontsize=12)
ax3.legend(loc="upper center", fontsize=9.5)
ax3.grid(alpha=0.25, axis="y")

fig.subplots_adjust(left=0.05, right=0.985, top=0.9, bottom=0.12)
out = "docs/ch00/tools/img/matplotlib_hunter.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
