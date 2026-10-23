r"""자료보다 몇 배를 쓰는가 — 만든 사람이 적어 둔 어림의 값.

이 쪽은 pandas 로 자료를 다루는 법을 익힌다. 그 라이브러리를 만든 사람이 「자료
크기의 5~10 배 램을 갖고 있어야 한다」는 어림을 적어 둔 적이 있다. 그 값을 잰다.

① 같은 자료를 CSV 로 적었을 때의 바이트와 DataFrame 이 실제로 차지하는 바이트.
② 연산 한 번이 지나갈 때 치솟는 봉우리 (프로세스 RSS 로 잰다).
③ dtype 을 손보면 얼마나 줄어드는가.

만드는 파일: docs/ch00/tools/img/pandas_memory.png
"""

import gc
import io
import threading
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

import pandas as pd   # noqa: E402
import psutil   # noqa: E402

PROC = psutil.Process()

RNG = np.random.default_rng(0)
N = 400_000
CITIES = ["서울", "부산", "대구", "인천", "광주", "대전", "울산"]
GRADES = ["A", "B", "C", "D"]


def make():
    return pd.DataFrame({
        "id": np.arange(N, dtype=np.int64),
        "city": RNG.choice(CITIES, N),
        "grade": RNG.choice(GRADES, N),
        "score": RNG.normal(70, 12, N),
        "count": RNG.integers(0, 100, N),
    })


df = make()

# === ① CSV 바이트 대 DataFrame 바이트 ================================
buf = io.StringIO()
df.to_csv(buf, index=False)
csv_bytes = len(buf.getvalue().encode("utf-8"))
mem_shallow = int(df.memory_usage(deep=False).sum())
mem_deep = int(df.memory_usage(deep=True).sum())

MB = 1024 ** 2
print(f"행 {N:,} × 열 {df.shape[1]}")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("   재는 방법                        크기(MB)   CSV 대비")
print(f"   CSV 로 적으면                   {csv_bytes/MB:>9.1f}   {1.0:>7.2f}x")
print(f"   memory_usage(deep=False)      {mem_shallow/MB:>9.1f}   "
      f"{mem_shallow/csv_bytes:>7.2f}x")
print(f"   memory_usage(deep=True)       {mem_deep/MB:>9.1f}   "
      f"{mem_deep/csv_bytes:>7.2f}x")
print("   deep=False 는 문자열이 가리키는 실제 바이트를 세지 않는다 — "
      "**열 하나를 포인터 배열로만 센다.**")

# === ② 연산이 지나갈 때의 봉우리 ====================================
# tracemalloc 은 파이썬이 한 할당만 센다. pandas·numpy 의 버퍼는 C 쪽 malloc 이라
# 보이지 않으므로, 프로세스의 실제 상주 메모리(RSS)를 딴 실로 훑어 봉우리를 잡는다.
def peak_rss(fn, settle=0.05):
    gc.collect()
    time.sleep(settle)
    base = PROC.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], PROC.memory_info().rss)
            time.sleep(0.001)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    out = fn()
    stop.set()
    th.join()
    peak[0] = max(peak[0], PROC.memory_info().rss)
    del out
    gc.collect()
    return peak[0] - base


OPS = [
    ("groupby + mean", lambda: df.groupby("city")["score"].mean()),
    ("sort_values", lambda: df.sort_values("score")),
    ("merge (자기 자신)", lambda: df.merge(
        df[["id", "score"]].rename(columns={"score": "score2"}), on="id")),
    ("to_csv (문자열로)", lambda: df.to_csv(index=False)),
]
print("\n② 연산이 지나갈 때 늘어난 상주 메모리 (RSS 봉우리 - 바탕)")
print("   RSS 는 할당기 사정을 타므로 돌릴 때마다 조금씩 흔들린다.")
print("   흔들리지 않는 것은 **모두 1 배 아래**라는 사실이다.")
print("   연산                     늘어난 양(MB)   자료 대비")
PEAKS = []
for name, fn in OPS:
    d = peak_rss(fn)
    PEAKS.append((name, d / MB, d / mem_deep))
    print(f"   {name:<22} {d/MB:>13.1f}   {d/mem_deep:>7.2f}x")
print(f"   이번 실행의 가장 큰 값이 {max(r for _, _, r in PEAKS):.2f} 배다. "
      f"**연산 하나가 자료의 몇 배를")
print("   더 쓰지는 않는다** — 적어도 이 크기, 이 연산들에서는 그렇다.")

# === ③ dtype 을 손보면 ==============================================
df2 = df.copy()
df2["city"] = df2["city"].astype("category")
df2["grade"] = df2["grade"].astype("category")
df2["score"] = df2["score"].astype(np.float32)
df2["count"] = df2["count"].astype(np.int8)
df2["id"] = df2["id"].astype(np.int32)
mem2 = int(df2.memory_usage(deep=True).sum())
print("\n③ dtype 을 손보면")
print(f"   그대로          {mem_deep/MB:>8.1f} MB")
print(f"   category + 축소  {mem2/MB:>8.1f} MB   ({mem_deep/mem2:.1f} 배 줄었다)")
print(f"   CSV 대비 {mem2/csv_bytes:.2f}x 로 내려온다.")
print("   문자열 열 둘을 category 로 바꾼 것이 거의 전부다 — "
      "같은 낱말이 되풀이되기 때문이다.")

RATIO_MAX = max(r for _, _, r in PEAKS)
print(f"\n   정리하면 배수는 **연산이 아니라 저장에서** 나온다.")
print(f"   연산 봉우리는 많아야 {RATIO_MAX:.2f} 배인데 ①의 저장이 이미 "
      f"{mem_deep/csv_bytes:.2f} 배이고,")
print(f"   그 가운데 {mem_deep/mem2:.1f} 배가 dtype 을 고르는 것만으로 사라진다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_A, C_B, C_C = "#d62728", "#1f77b4", "#2ca02c"

names1 = ["CSV 로 적으면", "deep=False", "deep=True"]
vals1 = [csv_bytes / MB, mem_shallow / MB, mem_deep / MB]
bars = ax1.bar(names1, vals1, color=["#999999", "#9ecae1", C_A], alpha=0.9)
for b, v in zip(bars, vals1):
    ax1.text(b.get_x() + b.get_width() / 2, v + max(vals1) * 0.02,
             f"${v:.1f}$ MB", ha="center", fontsize=10.5)
ax1.set_ylabel("메모리 (MB)")
ax1.set_ylim(0, max(vals1) * 1.18)
ax1.set_title(f"① 배수는 저장에서 나온다 (행 ${N:,}$)".replace(",", "{,}"),
              fontsize=12)
ax1.grid(alpha=0.25, axis="y")

names2 = [n for n, _, _ in PEAKS]
vals2 = [r for _, _, r in PEAKS]
ax2.barh(range(len(names2)), vals2, color=C_B, alpha=0.9)
ax2.axvline(1.0, color="#333333", lw=2.0, ls="--")
ax2.text(1.05, -0.45, "자료 자체", fontsize=10.5, color="#333333")
for i, v in enumerate(vals2):
    ax2.text(v + 0.05, i, f"${v:.2f}$x", va="center", fontsize=10.5)
ax2.set_yticks(range(len(names2)))
ax2.set_yticklabels(names2, fontsize=10.5)
ax2.set_xlabel("자료 크기의 몇 배까지 치솟는가")
ax2.set_xlim(0, max(vals2) * 1.25)
ax2.invert_yaxis()
ax2.set_title("② 연산 봉우리는 자료보다 작다", fontsize=12)
ax2.grid(alpha=0.25, axis="x")

names3 = ["그대로", "category\n+ 작은 dtype"]
vals3 = [mem_deep / MB, mem2 / MB]
bars = ax3.bar(names3, vals3, color=[C_A, C_C], alpha=0.9)
for b, v in zip(bars, vals3):
    ax3.text(b.get_x() + b.get_width() / 2, v + max(vals3) * 0.02,
             f"${v:.1f}$ MB", ha="center", fontsize=11)
ax3.annotate("", xy=(1, vals3[1]), xytext=(1, vals3[0]),
             arrowprops=dict(arrowstyle="->", color="#333333", lw=2.2))
ax3.text(1.08, (vals3[0] + vals3[1]) / 2, f"${mem_deep/mem2:.1f}$ 배",
         fontsize=11.5, color="#333333")
ax3.set_ylabel("메모리 (MB)")
ax3.set_ylim(0, max(vals3) * 1.18)
ax3.set_xlim(-0.6, 1.9)
ax3.set_title("③ dtype 을 손보면", fontsize=12)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch00/tools/img/pandas_memory.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
