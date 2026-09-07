"""로봇 청소기의 관측과 누적 지도 그림을 생성한다.

docs/ch01/paradigms/reinforcement.md 에 실리는
img/robot_cleaner_map.png 를 만든다. 시뮬레이션 본체는 그 페이지의 파이썬
예제와 동일하며(같은 시드, 같은 방), 여기에 그리기 코드만 덧붙였다.

실행:  python3 scripts/make_robot_cleaner_map.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

H, W = 11, 21          # room size
RADAR = 2              # the robot only sees cells this close
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
STEP_OF = {v: k for k, v in MOVES.items()}


# === Two rooms joined by a single doorway (same as the page) ===
def make_room():
    wall = np.zeros((H, W), dtype=bool)
    wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = True
    wall[:, 10] = True                       # partition
    wall[5, 10] = False                      # doorway
    return wall


def visible(pos):
    r, c = pos
    return [(i, j) for i in range(r - RADAR, r + RADAR + 1)
                   for j in range(c - RADAR, c + RADAR + 1)
                   if 0 <= i < H and 0 <= j < W]


def agent_map(pos, dirty, known, known_dirty, rng):
    """Walk to the nearest cell that is known-dirty or still unexplored."""
    parent, queue, goal = {pos: None}, deque([pos]), None
    while queue:
        cur = queue.popleft()
        if cur != pos and (known_dirty[cur] or known[cur] == 0):
            goal = cur
            break
        for dr, dc in MOVES.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if (0 <= nxt[0] < H and 0 <= nxt[1] < W
                    and nxt not in parent and known[nxt] != 2):
                parent[nxt] = cur
                queue.append(nxt)
    if goal is None:
        return list(MOVES)[rng.integers(4)]
    cur = goal
    while parent[cur] != pos:
        cur = parent[cur]
    return STEP_OF[(cur[0] - pos[0], cur[1] - pos[1])]


# === Rendering ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

UNKNOWN, WALL_C, CLEAN, DIRTY = "#E9ECEF", "#37474F", "#DCEBFB", "#FFCC80"
CMAP = [UNKNOWN, WALL_C, CLEAN, DIRTY]
SNAPS = [1, 40, 120]
OUT = "docs/ch01/paradigms/img/robot_cleaner_map.png"


def episode(seed=0):
    """Replay the agent, capturing o_t and x_t at the snapshot times."""
    rng = np.random.default_rng(seed)
    wall = make_room()
    dirty = ~wall.copy()
    total = dirty.sum()
    pos = (1, 1)
    known = np.zeros((H, W), np.int8)
    known_dirty = np.zeros((H, W), bool)
    path, frames = [pos], {}

    for step in range(1, max(SNAPS) + 1):
        dirty[pos] = False
        for cell in visible(pos):
            known[cell] = 2 if wall[cell] else 1
            known_dirty[cell] = dirty[cell]

        if step in SNAPS:
            grid = np.zeros((H, W), int)
            grid[known == 2] = 1
            grid[(known == 1) & ~known_dirty] = 2
            grid[(known == 1) & known_dirty] = 3
            obs = np.zeros((5, 5), int)
            for a in range(5):
                for b in range(5):
                    i, j = pos[0] - 2 + a, pos[1] - 2 + b
                    inside = 0 <= i < H and 0 <= j < W
                    obs[a, b] = (1 if not inside or wall[i, j]
                                 else 3 if dirty[i, j] else 2)
            frames[step] = (grid.copy(), obs, pos, list(path),
                            1 - dirty.sum() / total)

        dr, dc = MOVES[agent_map(pos, dirty, known, known_dirty, rng)]
        nxt = (pos[0] + dr, pos[1] + dc)
        if not wall[nxt]:
            pos = nxt
            path.append(pos)
    return frames


def paint(ax, grid):
    h, w = grid.shape
    for i in range(h):
        for j in range(w):
            ax.add_patch(Rectangle((j, h - 1 - i), 1, 1,
                                   facecolor=CMAP[grid[i, j]],
                                   edgecolor="white", linewidth=0.4))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    frames = episode()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.4),
                             gridspec_kw={"height_ratios": [1, 1.9]})

    for k, t in enumerate(SNAPS):
        grid, obs, pos, path, cov = frames[t]

        ax = axes[0][k]
        paint(ax, obs)
        ax.plot(2.5, 2.5, "o", color="#D32F2F", markersize=9, zorder=5)
        ax.set_title(f"$t={t}$", fontsize=11, pad=6)

        ax = axes[1][k]
        paint(ax, grid)
        ax.plot([p[1] + 0.5 for p in path],
                [H - 1 - p[0] + 0.5 for p in path],
                color="#1565C0", linewidth=1.0, alpha=0.75, zorder=4)
        ax.plot(pos[1] + 0.5, H - 1 - pos[0] + 0.5, "o",
                color="#D32F2F", markersize=8, zorder=5)
        ax.add_patch(Rectangle((pos[1] - 2, H - 1 - pos[0] - 2), 5, 5,
                               fill=False, edgecolor="#D32F2F", linewidth=1.4,
                               linestyle="--", zorder=6))
        ax.set_xlabel(f"청소 완료 {cov:.0%}", fontsize=10, labelpad=4)

    axes[0][0].set_ylabel("관측 $o_t$\n(레이더 5×5)", fontsize=11, labelpad=12)
    axes[1][0].set_ylabel("상태 $x_t$\n(누적 지도)", fontsize=11, labelpad=12)

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#BBB")
               for c in CMAP]
    handles += [Line2D([], [], color="#1565C0", lw=1.4),
                Line2D([], [], color="#D32F2F", marker="o", lw=0, markersize=7),
                Line2D([], [], color="#D32F2F", lw=1.4, ls="--")]
    labels = ["미지 (?)", "벽", "청소 완료", "더러움",
              "지나온 경로", "청소기", "현재 레이더 범위"]
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("로봇 청소기: 매 순간의 관측과 누적된 지도", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig.savefig(OUT, dpi=170, facecolor="white")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
