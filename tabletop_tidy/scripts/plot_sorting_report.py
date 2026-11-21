"""Render a tabletop sorting report as a standalone PNG."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from textwrap import shorten

from matplotlib import font_manager, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator


OBJECTS = {"cup": "杯子", "spoon": "勺子", "wrench": "扳手", "screwdriver": "螺丝刀",
           "stapler": "订书机", "tape_dispenser": "胶带座"}
BOXES = {"kitchen": "餐厨用品", "tools": "工具", "office": "办公用品"}
CHECKS = [("正确盒内", "inside_correct_tray"), ("低于盒沿", "fits_below_rim"),
          ("贴近盒底", "supported"), ("静稳", "stable")]
# 颜色用于图形标记，文字统一为黑色。
REASONS = [
    ("滑落", "#548B78", ("Object slipped",)),
    ("接触不足", "#D89C53", ("Both fingers must contact",)),
    ("碰撞", "#BF7064", ("Collision:",)),
    ("入盒不合格", "#A58BB5", ("Object did not settle",)),
    ("运动失败", "#7595B4", ("IK failed", "TCP tracking error")),
    ("目标被推移", "#C5BA67", ("Target moved",)),
    ("其他", "#A5AAA8", ()),
]
PAPER = "#FBFCFA"
RULE = "#D8DEDA"
PASSED = "#DCEDE4"
FAILED = "#F4DED6"
NO_CANDIDATES = "No GraspGenX candidates"


def rejection_counts(attempts):
    counts = Counter()
    for attempt in attempts:
        for reason, count in attempt.get("rejected_candidates", {}).items():
            # 缺少候选是观测结果，没有实际试跑，单独显示在图中。
            if reason.startswith(NO_CANDIDATES):
                continue
            label = next((label for label, _, phrases in REASONS if any(p in reason for p in phrases)), "其他")
            counts[label] += count
    return counts


def render(report, path, output):
    evaluation = report["final_evaluation"]
    lids = report["final_lids"]
    planned = {obj["name"]: obj["attempts"] for obj in report["objects"]}
    order = {name: i for i, name in enumerate(planned)}
    objects = sorted(evaluation["objects"], key=lambda obj: order.get(obj["name"], len(order)))
    closed = sum(lid["closed"] for lid in lids)
    success = report["success"]
    count = len(objects)
    fig = Figure(figsize=(14, max(7.2, 4.44 + .46 * count)), dpi=160, facecolor=PAPER)
    FigureCanvasAgg(fig)

    fig.text(.047, .929, "桌面整理运行报告", fontsize=35, weight="bold", va="center")
    fig.text(.048, .862, f"种子 {report['seed']}    ·    正确入盒 {evaluation['sorted_count']}/{count}"
             f"    ·    盒盖闭合 {closed}/{len(lids)}", fontsize=15, va="center")
    fig.text(.952, .929, "验收通过" if success else "验收未通过", ha="right", va="center",
             fontsize=15, weight="bold", bbox={"facecolor": PASSED if success else FAILED,
                                               "edgecolor": "none", "pad": 8})

    opened = {lid["category"]: lid["angle_degrees"] for lid in report["lids"]
              if lid["action"] == "open" and "angle_degrees" in lid}
    gap = .012
    width = (.906 + gap) / max(1, len(lids)) - gap
    for index, lid in enumerate(lids):
        x = .047 + index * (width + gap)
        fig.add_artist(Rectangle((x, .709), width, .104, transform=fig.transFigure,
                                 facecolor="#EEF2EE" if lid["closed"] else FAILED, edgecolor="none"))
        category = lid["category"]
        opening = f"{opened[category]:.1f}°" if category in opened else "—"
        state = "已闭合" if lid["closed"] else "未闭合"
        fig.text(x + .012, .782, BOXES.get(category, category), fontsize=13, weight="bold", va="center")
        fig.text(x + width - .012, .782, state, fontsize=11, ha="right", va="center")
        fig.text(x + .012, .742, f"开盖 {opening}    ·    最终 {lid['angle_degrees']:.2f}°",
                 fontsize=11, va="center")

    fig.text(.047, .650, "逐件验收", fontsize=18, weight="bold")
    fig.text(.047, .613, "按处理顺序排列，判定取自最终状态，分数来自GraspGenX", fontsize=11)
    fig.text(.623, .650, "试跑淘汰原因", fontsize=18, weight="bold")
    fig.text(.623, .613, "同一抓取候选的不同放置方案可能分别计次", fontsize=11)

    table = fig.add_axes((.047, .205, .545, .365))
    chart = fig.add_axes((.623, .205, .330, .365), sharey=table)
    table.set(xlim=(0, 1), ylim=(max(count, 1) - .5, -.5))
    table.set_axis_off()
    columns = [.012, .229, .381, .500, .627, .735, .830, .943]
    labels = ["物品", "目标盒", *(label for label, _ in CHECKS), "尝试轮次", "选中分数"]
    for index, (x, label) in enumerate(zip(columns, labels)):
        table.text(x, -.78, label, ha="left" if index == 0 else "center", va="center", fontsize=10.5)

    rejected = []
    for row, obj in enumerate(objects):
        name, category = obj["name"], obj["category"]
        attempts = planned.get(name, [])
        rejected.append(rejection_counts(attempts))
        score = next((attempt["score"] for attempt in reversed(attempts) if "score" in attempt), None)
        table.text(columns[0], row, OBJECTS.get(name, name), fontsize=13, va="center")
        table.text(columns[1], row, BOXES.get(category, category), fontsize=11, ha="center", va="center")
        for x, (_, key) in zip(columns[2:6], CHECKS):
            check = obj.get(key)
            if check is None:
                table.text(x, row, "—", fontsize=10, ha="center", va="center")
                continue
            table.scatter(x, row, s=184, color=PASSED if check else FAILED, edgecolors="none")
            table.text(x, row, "✓" if check else "×", fontsize=10, ha="center", va="center")
        table.text(columns[6], row, str(len(attempts)), fontsize=12, ha="center", va="center")
        table.text(columns[7], row, f"{score:.3f}" if score is not None else "—",
                   fontsize=12, ha="center", va="center")

    for boundary in range(count + 1):
        table.axhline(boundary - .5, color=RULE, linewidth=.65, clip_on=False)
        chart.axhline(boundary - .5, color=RULE, linewidth=.65, zorder=0, clip_on=False)
    if not objects:
        table.text(.012, 0, "报告未记录物品验收结果", fontsize=12, va="center")

    totals = [sum(counts.values()) for counts in rejected]
    start = [0] * count
    legend = []
    for label, color, _ in REASONS:
        values = [counts[label] for counts in rejected]
        if any(values):
            chart.barh(range(count), values, left=start, height=.5, color=color, zorder=3)
            legend.append(Patch(facecolor=color, label=label))
            start = [left + value for left, value in zip(start, values)]

    limit = max(totals, default=0) * 1.18
    for row, (obj, total) in enumerate(zip(objects, totals)):
        attempts = planned.get(obj["name"], [])
        missing = any(reason.startswith(NO_CANDIDATES) for attempt in attempts
                      for reason in attempt.get("rejected_candidates", {}))
        label = str(total) if attempts else "未执行"
        if missing:
            label = f"{total} · 曾缺候选" if total else "缺少候选"
            limit = max(limit, total * 1.42)
        chart.annotate(label, (total, row), xytext=(5, 0), textcoords="offset points", va="center", fontsize=11)

    chart.set(xlim=(0, max(1, limit)), yticks=[], xlabel="淘汰次数")
    chart.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    chart.grid(axis="x", color=RULE, linewidth=.55)
    chart.tick_params(axis="x", length=0, labelsize=10)
    for spine in chart.spines.values():
        spine.set_visible(False)

    if legend:
        fig.legend(handles=legend, loc="center left", bbox_to_anchor=(.616, .089), ncol=min(5, len(legend)),
                   frameon=False, fontsize=9, handlelength=1.0, columnspacing=.8)
    fig.text(.047, .103, "✓ 通过    × 未通过    — 无记录", fontsize=10)
    error = report.get("error") or next((lid["error"] for lid in report["lids"] if "error" in lid), "")
    footer = f"记录目录｜{path.parent.name}"
    if error:
        footer = f"运行错误｜{shorten(error, width=105, placeholder='…')}"
    fig.text(.047, .045, footer, fontsize=9.5)
    fig.savefig(output, format="png", facecolor=fig.get_facecolor())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to sorting_report.json")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path (default: next to the JSON report)")
    parser.add_argument("--font", type=Path, help="Use a specific Chinese TTF, OTF or TTC font")
    args = parser.parse_args()
    output = args.output or args.report.with_suffix(".png")
    if output.suffix.lower() != ".png":
        parser.error("The output path must end in .png")

    font = args.font
    if font is None:
        names = ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei",
                 "Noto Sans CJK JP", "Droid Sans Fallback", "SimHei")
        available = {entry.name for entry in font_manager.fontManager.ttflist}
        family = next((name for name in names if name in available), None)
        if family is None:
            parser.error("Install a Chinese font or provide its file with --font")
    else:
        try:
            font_manager.fontManager.addfont(str(font))
            family = font_manager.FontProperties(fname=str(font)).get_name()
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(f"Unable to load font: {error}")

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with rc_context({"font.family": family, "text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "axes.unicode_minus": False}):
            render(report, args.report, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Unable to plot {args.report}: {error}\n")
    print(output)


if __name__ == "__main__":
    main()
