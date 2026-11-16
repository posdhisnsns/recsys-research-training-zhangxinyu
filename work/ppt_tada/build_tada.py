# -*- coding: utf-8 -*-
"""构建 TADA 组会汇报 PPT（蓝白极简 · 43 页）。"""
import os
import sys

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ppt_adarec"))
from deck_lib import (  # noqa: E402
    Deck, block, card, rect, round_box, arrow, add_table, add_math,
    para, textbox, FONT_CN,
    INK, BODY, GRAY, MUTED, BLUE, BLUE_MID, BLUE_LT, TINT, TINT2, LINE, WHITE,
    W, H, MARGIN, CW, WARNINGS,
)

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOGO = os.path.join(ROOT, "work", "ppt_adarec", "cqu_logo_hd.png")
AST = os.path.join(ROOT, "knowledge_base", "paper_reading", "assets", "TADA")
CH = os.path.join(AST, "charts")
OUT = os.path.join(ROOT, "information", "PPT", "张心雨_TADA文献阅读及代码复现_组会汇报.pptx")

TOTAL = 44
PART1 = "PART 1 · 文献阅读"
PART2 = "PART 2 · 代码复现"
PART3 = "PART 3 · 学习总结"
TOP = 1.88


def caption(slide, y, text, num, x=MARGIN, w=None, size=15):
    block(slide, x, y, w or CW, 0.30, [dict(text=text, size=size, color=MUTED, ls=1.0)],
          where=f"图注{num}")


def notes(slide, y, text, num, size=17, h=0.36):
    block(slide, MARGIN, y, CW, h, [dict(text=text, size=size, color=GRAY, sub=True)],
          where=f"注释{num}")


# ============================ 1 封面 ============================
def s01(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 9.10, 0, 4.233, H, TINT2)
    rect(s, 9.10, 0, 0.05, H, BLUE_LT)
    s.shapes.add_picture(LOGO, Inches(0.95), Inches(0.60), width=Inches(2.15))
    rect(s, 0.95, 1.92, 2.0, 0.05, BLUE)
    block(s, 0.95, 2.16, 7.7, 0.40,
          [dict(text="研究生组会汇报 · 2026 年 11 月", size=20, color=GRAY, ls=1.0)],
          where="封面kicker")
    block(s, 0.95, 2.66, 8.0, 1.44,
          [dict(text="TADA：尾部感知数据增强", size=40, bold=True, color=INK, ls=1.05),
           dict(text="改善长尾序列推荐", size=40, bold=True, color=INK, ls=1.05)],
          where="封面标题")
    block(s, 0.95, 4.24, 7.6, 0.80,
          [dict(text="Tail-Aware Data Augmentation for Long-Tail Sequential Recommendation",
                size=16, color=GRAY, ls=1.25)], where="封面英文")
    rect(s, 0.95, 5.12, 5.4, 0.012, LINE)
    block(s, 0.95, 5.30, 6.2, 0.46,
          [dict(text="**汇报人：张心雨**", size=24, color=BLUE, ls=1.0)],
          where="封面汇报人")
    block(s, 0.95, 5.86, 6.6, 0.40,
          [dict(text="文献阅读 · 代码复现 · 结果对照（WWW 2026）", size=17,
                color=GRAY, ls=1.0)], where="封面范围")
    for (lab, val), y in zip([("论文", "WWW 2026"), ("代码", "KingGugu/TADA"),
                              ("汇报范围", "阅读 + 复现")], [2.20, 3.42, 4.64]):
        block(s, 9.62, y, 3.2, 0.30, [dict(text=lab, size=15, color=BLUE_MID, ls=1.0)],
              where="封面右侧标签")
        block(s, 9.62, y + 0.30, 3.3, 0.42,
              [dict(text=val, size=21, bold=True, color=INK, ls=1.0)], where="封面右侧值")


# ============================ 2 提纲 ============================
def s02(prs, dk):
    s = dk.page("汇报提纲",
                "三部分：论文方法与结果、代码复现与差异分析、本学期学习总结。",
                PART1, 2)
    items = [
        ("01", "文献阅读", "第 3–16 页", [
            "1.1 背景 · 问题 · 方法框架",
            "1.2 候选集、算子、混合与损失",
            "1.3 论文 RQ1–RQ5 结果",
        ]),
        ("02", "代码复现", "第 17–41 页", [
            "2.1 文件职责　2.2 模块实现",
            "2.3 参数 · 数据 · 模型 · 训练",
            "2.4 流程与增强示例",
            "2.5 复现结果与差异",
            "2.6 记录与结果图表",
        ]),
        ("03", "学习总结", "第 42–44 页", [
            "3.1 本学期工作与产出",
            "3.2 后续学习计划",
        ]),
    ]
    x = MARGIN
    for code, title, pages, lines in items:
        card(s, x, 1.88, 3.83, 4.60, fill=TINT, line=LINE)
        rect(s, x + 0.02, 1.88, 3.79, 0.06, BLUE)
        block(s, x + 0.30, 2.12, 3.3, 0.5,
              [dict(text=code, size=28, bold=True, color=BLUE_LT, ls=1.0)], where="目录号")
        block(s, x + 0.30, 2.64, 3.3, 0.45,
              [dict(text=title, size=26, bold=True, color=INK, ls=1.0)], where="目录标题")
        block(s, x + 0.30, 3.14, 3.3, 0.30,
              [dict(text=pages, size=15, color=MUTED, ls=1.0)], where="目录页码")
        paras = [dict(text=t, size=17, color=BODY, sb=10 if i else 4)
                 for i, t in enumerate(lines)]
        block(s, x + 0.30, 3.56, 3.30, 2.75, paras, where="目录条目")
        x += 4.13


# ============================ 分隔页 ============================
def section(prs, num, code, title, lines, note):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, 0.16, H, BLUE)
    block(s, 1.30, 1.98, 3.0, 0.80,
          [dict(text=code, size=44, bold=True, color=BLUE_LT, ls=1.0)], where="分隔号")
    block(s, 1.30, 2.75, 9.0, 0.85,
          [dict(text=title, size=44, bold=True, color=INK, ls=1.0)], where="分隔标题")
    rect(s, 1.30, 3.85, 3.2, 0.05, BLUE)
    paras = [dict(text=t, size=18, color=BODY, sb=8 if i else 0) for i, t in enumerate(lines)]
    block(s, 1.30, 4.15, 9.5, 2.40, paras, where="分隔条目")
    block(s, 1.30, 6.62, 9.5, 0.4,
          [dict(text=note, size=16, color=MUTED, ls=1.0)], where="分隔注")
    s.shapes.add_picture(LOGO, Inches(11.30), Inches(0.52), width=Inches(1.41))
    block(s, 11.0, 7.03, 1.8, 0.3,
          [dict(text=f"{num:02d} / {TOTAL:02d}", size=13, color=MUTED, ls=1.0)],
          align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, where="页码")
    rect(s, 0, 7.46, W, 0.04, BLUE)
    return s


def s03(prs):
    section(prs, 3, "01", "文献阅读",
            ["1.1　研究背景 / 研究问题 / 方法框架",
             "1.2　候选集、尾部算子、表示混合与损失",
             "1.3　论文 RQ1–RQ5 的实验结果"],
            "对应原文 Section 3–5 与附录 Table 2 / 3 / 4 / 6 / 7")


def s16(prs):
    section(prs, 17, "02", "代码复现",
            ["2.1　文件职责　　2.2　四个模块的具体实现",
             "2.3　参数设置 / 数据处理 / 模型设定 / 模型训练",
             "2.4　训练流程与增强示例",
             "2.5　复现结果、差异归因与参数实验",
             "2.6　实验记录与结果图表"],
            "代码：KingGugu/TADA（官方实现）+ 本地实验入口")


def s41(prs):
    section(prs, 42, "03", "学习总结",
            ["3.1　本学期的工作与产出",
             "3.2　后续学习计划"],
            "论文复现、实验设计与结果分析")


# ============================ 4 背景 ============================
def s04(prs, dk):
    s = dk.page("1.1 研究背景：长尾分布与头尾跷跷板",
                "序列推荐依赖历史交互；真实数据中长尾分布使尾部物品与短序列用户训练不足。",
                PART1, 4)
    paras = [
        dict(text="**三个事实**", size=20, color=INK),
        dict(text="用户与物品的交互次数均呈长尾：多数物品只有个位数交互。", size=18, bullet=True, sb=8),
        dict(text="尾部物品嵌入训练不足，排序时难以被召回。", size=18, bullet=True, sb=6),
        dict(text="短序列无法完整表达偏好，头部与尾部用户差距被放大。", size=18, bullet=True, sb=6),
        dict(text="**现有方法的两个矛盾**", size=20, color=INK, sb=12),
        dict(text="知识迁移只搬头部表示，不增加尾部交互频次。", size=18, bullet=True, sb=8),
        dict(text="改善尾部的同时容易拉低整体与头部，形成跷跷板。", size=18, bullet=True, sb=6),
    ]
    block(s, MARGIN, TOP, 6.55, 4.55, paras, where="p4正文")
    s.shapes.add_picture(os.path.join(AST, "fig1.png"), Inches(7.55), Inches(1.90),
                         width=Inches(4.60))
    caption(s, 6.72, "图 1：Beauty 的长尾分布与头尾跷跷板（论文 Fig.1）", 4, x=7.55, w=5.2)


# ============================ 5 问题与目的 ============================
def s05(prs, dk):
    s = dk.page("1.1 研究问题与研究目的",
                "目标：增加有意义的尾部交互，同时保持整体与头部表现。",
                PART1, 5)
    cards = [
        ("Q1 候选从哪来", "现有增强随机替换，语义容易漂移。需要兼顾全局相似与序列先后关系。", BLUE),
        ("Q2 怎么改序列", "尾部信号要增加，但原偏好不能丢；短序列与长序列应区别对待。", BLUE_MID),
        ("Q3 如何跨用户", "需要融合协同信号，同时区分头偏好与尾偏好两组用户。", BLUE_LT),
    ]
    y = 1.88
    for title, desc, color in cards:
        card(s, MARGIN, y, 7.05, 1.30, fill=TINT, line=LINE)
        rect(s, MARGIN + 0.02, y, 0.07, 1.30, color)
        block(s, MARGIN + 0.28, y + 0.10, 6.6, 0.34,
              [dict(text=title, size=19, bold=True, color=INK, ls=1.0)], where="p5卡题")
        block(s, MARGIN + 0.28, y + 0.50, 6.55, 0.72,
              [dict(text=desc, size=17, color=BODY, ls=1.15)], where="p5卡文")
        y += 1.42

    card(s, 7.90, 1.88, 4.82, 2.95, fill=TINT2, line=None)
    block(s, 8.16, 2.00, 4.32, 2.75, [
        dict(text="Q4 效果边界", size=19, bold=True, color=BLUE, ls=1.0),
        dict(text="能否在多骨干、多数据集上同时改善整体、头部与尾部？", size=16, sb=8),
        dict(text="研究目的", size=19, bold=True, color=BLUE, ls=1.0, sb=14),
        dict(text="用候选集与两个算子增加尾部交互，用表示混合保留偏好，用跨序列增强引入协同信号。",
             size=16, sb=8),
    ], where="p5右卡")
    card(s, 7.90, 4.98, 4.82, 1.24, fill=TINT, line=LINE, accent=BLUE)
    block(s, 8.16, 5.08, 4.32, 1.04, [
        dict(text="与既有方法的差别", size=17, bold=True, color=INK, ls=1.0),
        dict(text="不改骨干结构、不新增可学习模块，只在训练数据与表示层做增强。", size=15, sb=6),
    ], where="p5右下卡")


# ============================ 6 方法框架 ============================
def s06(prs, dk):
    s = dk.page("1.1 研究方法：候选集 → 算子 → 混合 → 训练",
                "TADA 由四个环节串成，全部作用在训练阶段，推理时不需要额外结构。",
                PART1, 6)
    steps = [("① 候选集", "线性相似 + 相邻共现"),
             ("② 尾部算子", "T-Substitute / T-Insert"),
             ("③ 表示混合", "Beta 权重逐位置混合"),
             ("④ 跨序列增强", "按头尾偏好分组混合"),
             ("⑤ 两阶段训练", "L_main + L_op + L_cross")]
    bw, gap = 2.20, 0.36
    x = MARGIN
    for i, (name, desc) in enumerate(steps):
        solid = i in (0, 4)
        round_box(s, x, 1.86, bw, 1.20, fill=BLUE if solid else WHITE, line=BLUE_LT)
        block(s, x + 0.10, 1.98, bw - 0.20, 0.32,
              [dict(text=name, size=17, bold=True, color=WHITE if solid else INK, ls=1.0)],
              align=PP_ALIGN.CENTER, where="p6流程名")
        block(s, x + 0.12, 2.42, bw - 0.24, 0.60,
              [dict(text=desc, size=14, color=WHITE if solid else GRAY, ls=1.1)],
              align=PP_ALIGN.CENTER, where="p6流程文")
        if i < len(steps) - 1:
            arrow(s, x + bw + 0.05, 2.32, 0.26, 0.26)
        x += bw + gap

    card(s, MARGIN, 3.26, CW, 2.66, fill=TINT, line=LINE, accent=BLUE)
    block(s, MARGIN + 0.32, 3.40, CW - 0.7, 2.38, [
        dict(text="**为什么这样拆**", size=19, color=INK, ls=1.0),
        dict(text="候选集解决“改什么”：只改语义相关或顺序相邻的物品，避免随机替换带来的漂移。", size=17, sb=8),
        dict(text="算子解决“改哪里”：替换增加尾部曝光，插入补充引出尾部的上下文。", size=17, sb=6),
        dict(text="混合与跨序列解决“怎么用”：在表示层混合，保留原偏好并引入同组用户的协同信号。", size=17, sb=6),
        dict(text="两阶段解决“什么时候加”：先训出可靠嵌入，再叠加增强损失，避免噪声样本污染早期训练。", size=17, sb=6),
    ], where="p6说明")
    card(s, MARGIN, 6.06, CW, 0.78, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.30, 6.14, CW - 0.6, 0.62, [
        dict(text="所有增强都发生在训练期：推理时只保留原序列与骨干前向，参数量与延迟与原模型一致。",
             size=15, color=INK),
    ], where="p6补")


def s06b(prs, dk):
    s = dk.page("1.1 方法总览：TADA 的三条支路",
                "a 支路构造候选，b 支路做序列级增强，c 支路做跨用户增强，最后汇入同一个编码器与损失。",
                PART1, 7)
    s.shapes.add_picture(os.path.join(AST, "fig2.png"), Inches(1.10), Inches(1.88),
                         width=Inches(11.15))
    caption(s, 6.58, "图 2：TADA 总体框架（论文 Fig.2）", 6, x=1.10, w=11.15)


# ============================ 公式页 ============================
def formula_page(prs, dk, num, title, kicker, formulas, syms, note, fs=21):
    s = dk.page(title, kicker, PART1, num)
    fh = 0.47 * len(formulas) + 0.38
    card(s, MARGIN, 1.86, CW, fh, fill=WHITE, line=LINE, accent=BLUE)
    paras = [dict(text=f, size=fs, color=INK, math=True, ls=1.0,
                  sb=0 if i == 0 else 10) for i, f in enumerate(formulas)]
    block(s, MARGIN + 0.34, 1.96, CW - 0.6, fh - 0.18, paras, where=f"公式p{num}")

    rows = (len(syms) + 1) // 2
    sy = 1.86 + fh + 0.18
    sh_card = 6.44 - sy
    pitch = min(0.50, (sh_card - 0.50) / rows)
    card(s, MARGIN, sy, CW, sh_card, fill=TINT, line=LINE)
    block(s, MARGIN + 0.26, sy + 0.09, 3.0, 0.30,
          [dict(text="符号说明", size=17, bold=True, color=BLUE, ls=1.0)], where=f"符号题p{num}")
    colw = (CW - 0.90) / 2
    pt_gap = max(3.0, pitch * 72 - 17 * 1.22)
    for ci, chunk in enumerate([syms[:rows], syms[rows:]]):
        sh = textbox(s, MARGIN + 0.26 + ci * (colw + 0.40), sy + 0.44, colw,
                     sh_card - 0.50)
        tf = sh.text_frame
        for i, (sym, mean) in enumerate(chunk):
            p = para(tf, first=(i == 0), space_before=0 if i == 0 else pt_gap,
                     line_spacing=1.0)
            add_math(p, sym, 17, BLUE, bold=True)
            add_math(p, "　" + mean, 17, BODY, font=FONT_CN)

    block(s, MARGIN, 6.52, CW, 0.40,
          [dict(text=note, size=16, color=GRAY, sub=True, ls=1.0)], where=f"注释p{num}")


def s07(prs, dk):
    formula_page(
        prs, dk, 8,
        "1.2 公式（1/4）｜候选集：线性相关 + 相邻共现",
        "候选集在训练前一次构建；训练时只做查表与采样，避免逐 batch 计算物品对相似度。",
        ["(2)　M̂_{S} = argmin_{M_S} ‖ M_{D} − M_{D} M_{S} ‖²_{F} + λ_{r} ‖ M_{S} ‖²_{F} ，　diag(M_{S}) ≤ ξ",
         "(3)　P = ( M_{D}^{T} M_{D} + λ_{r} I )^{−1} ，　M̂_{S} = I − P · diag(γ) ，　γ_{j} = λ_{r} 或 (1−ξ)/P_{jj}",
         "(4)　c_{v} = c_{v}^{r} ∪ c_{v}^{c}"],
        [("M_{D}", "用户–物品二值交互矩阵"),
         ("M_{S}", "物品–物品相似度矩阵"),
         ("λ_{r}", "岭回归系数，控制收缩强度"),
         ("ξ", "对角元素上界，ξ∈[0,1)"),
         ("P", "逆矩阵缓存，避免重复求逆"),
         ("γ_{j}", "对角修正项，按约束是否触发取值"),
         ("c_{v}^{r}", "相关性候选，取 top-K"),
         ("c_{v}^{c}", "共现候选，来自相邻位置"),
         ("c_{v}", "合并后的候选集合"),
         ("K", "每个物品保留的候选数量")],
        "共现项只统计训练前缀：头物品取前后相邻的尾物品，尾物品取所有前驱；既保留相似性，也保留先后关系。")


def s08(prs, dk):
    formula_page(
        prs, dk, 9,
        "1.2 公式（2/4）｜尾部感知算子：替换与插入",
        "两个算子分别增加尾物品的“出现次数”与“上下文”，并按序列长度自适应选择。",
        ["(5)　p ∼ Uniform(a, b) ，　0 < a < b < 1",
         "(6)　s'_{u} = T-Substitute(s_{u}) = ( v_{1}, …, v̄_{idx} , …, v_{|s|} )",
         "(9)　s'_{u} = T-Insert(s_{u}) = ( …, v̄_{c}, v_{idx} , … )",
         "(10)　s_{u}^{e} = Extend(s_{u}) ，　|s'_{u}| = |s_{u}^{e}|",
         "(11)　p_{insert} = 1 − |s_{u}|/N ，　p_{substitute} = |s_{u}|/N"],
        [("p", "本次操作的选中概率"),
         ("a, b", "采样区间端点，控制增强强度"),
         ("v̄_{idx}", "替换位置采到的候选物品"),
         ("idx", "被选中或插入的位置下标"),
         ("v̄_{c}", "插在该位置之前的候选物品"),
         ("s_{u}^{e}", "扩展视图，复制插入位置的尾物品"),
         ("|s_{u}|", "当前训练序列长度（去填充）"),
         ("N", "最大序列长度，取 50")],
        "短序列偏向插入、长序列偏向替换；插入后长度对齐由 s_{u}^{e} 保证，两视图才能逐位置混合。")


def s09(prs, dk):
    formula_page(
        prs, dk, 10,
        "1.2 公式（3/4）｜表示级混合与跨序列增强",
        "增强序列不直接当作新样本，而是在编码器输出层与原始表示混合，保留原有偏好信息。",
        ["(7)　E'_{u} = Look-up(M_{V}, s'_{u}) ，　H'_{u} = Encoder(E'_{u})",
         "(8)　H_{u}^{ao} = λ_{m} H_{u} + (1 − λ_{m}) H'_{u} ，　λ_{m} ∼ Beta(α, α)",
         "(12)　{ H'_{u} } = Shuffle( { H_{u} } )",
         "(13)　{ H_{u}^{ac} } = Λ { H_{u} } + ( I − Λ ) { H'_{u} }"],
        [("M_{V}", "共享物品嵌入矩阵"),
         ("H_{u}", "原始序列表示"),
         ("H'_{u}", "增强序列表示（同编码器）"),
         ("λ_{m}", "混合权重，逐位置采样"),
         ("α", "Beta 分布参数，控制混合偏向"),
         ("H_{u}^{ao}", "算子混合后的表示"),
         ("Λ", "逐样本权重构成的对角矩阵"),
         ("H_{u}^{ac}", "跨序列混合后的表示")],
        "跨序列混合按尾物品占比把 batch 分成头偏好组与尾偏好组，组内打乱后再混合；正负目标嵌入沿用同一配对与权重。")


def s10(prs, dk):
    s = dk.page("1.2 公式（4/4）｜训练目标与两阶段流程",
                "第一阶段只用主损失训练原骨干，第二阶段再加入两项增强损失。",
                PART1, 11)
    card(s, MARGIN, 1.86, CW, 1.55, fill=WHITE, line=LINE, accent=BLUE)
    block(s, MARGIN + 0.34, 1.94, CW - 0.6, 1.40, [
        dict(text="(14)　L_{main} = BCE(H_{u}, E_{u}^{+}, E_{u}^{−}) = −log σ(H_{u}·E_{u}^{+}) − log[1 − σ(H_{u}·E_{u}^{−})]",
             size=17, math=True, ls=1.0),
        dict(text="(15)　L_{operator} = BCE(H_{u}^{ao}, E_{u}^{+}, E_{u}^{−})　　(16)　L_{cross} = BCE(H_{u}^{ac}, E_{u}^{ac+}, E_{u}^{ac−})",
             size=17, math=True, ls=1.0, sb=10),
        dict(text="(17)　L = L_{main} + L_{operator} + L_{cross}", size=18, math=True, ls=1.0, sb=10),
    ], where="p10公式")

    add_table(s, MARGIN, 3.56, CW, [2.2, 6.4, 3.2], [
        ["损失项", "作用", "阶段"],
        ["L_{main}", "原始视图的下一样本预测，主监督信号", "一、二阶段"],
        ["L_{operator}", "算子增强视图与原始正负目标对齐", "第二阶段"],
        ["L_{cross}", "跨用户混合视图与其混合目标对齐", "第二阶段"],
    ], row_h=0.50, header_h=0.52, size=18, sub=True, where="p10表")
    notes(s, 5.78, "三项损失等权相加，论文未设额外系数；增广后的目标嵌入不参与梯度，只作为监督信号。", 10,
          size=17, h=0.36)
    card(s, MARGIN, 6.30, CW, 0.52, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.30, 6.36, CW - 0.6, 0.40, [
        dict(text="两阶段的意义：阶段一保证嵌入可用，阶段二才让增强视图参与优化；两阶段共用同一评测口径。",
             size=15, color=INK),
    ], where="p10补")


# ============================ 11 RQ1 ============================
def s11(prs, dk):
    s = dk.page("1.3 结果 RQ1：主实验（论文 Table 2）",
                "TADA 接入三种骨干、三个数据集，整体与尾物品指标均提升；下表节选总体与尾物品 H@10。",
                PART1, 12)
    rows = [["数据集", "骨干", "原骨干 H@10", "TADA H@10", "提升", "尾物品 H@10"],
            ["Toys", "GRU4Rec", "0.0364", "0.0512", "[[+40.7%]]", "0.0230 → 0.0368"],
            ["Toys", "SASRec", "0.0716", "**0.0846**", "[[+18.2%]]", "0.0607 → 0.0716"],
            ["Toys", "FMLP-Rec", "0.0555", "0.0664", "[[+19.6%]]", "0.0454 → 0.0528"],
            ["Beauty", "GRU4Rec", "0.0383", "0.0555", "[[+44.9%]]", "0.0119 → 0.0195"],
            ["Beauty", "SASRec", "0.0605", "**0.0732**", "[[+21.0%]]", "0.0304 → 0.0409"],
            ["Beauty", "FMLP-Rec", "0.0539", "0.0658", "[[+22.1%]]", "0.0270 → 0.0322"],
            ["Sports", "GRU4Rec", "0.0155", "0.0279", "[[+80.0%]]", "0.0052 → 0.0095"],
            ["Sports", "SASRec", "0.0317", "**0.0402**", "[[+26.8%]]", "0.0185 → 0.0208"],
            ["Sports", "FMLP-Rec", "0.0242", "0.0304", "[[+25.6%]]", "0.0118 → 0.0166"]]
    add_table(s, MARGIN, TOP, CW, [1.3, 2.0, 2.2, 2.0, 1.9, 3.0], rows,
              row_h=0.40, header_h=0.44, size=17, where="p11表")
    notes(s, 5.98, "提升幅度按同一骨干的原结果计算；SASRec 加粗表示该数据集上整体最优的骨干。", 11,
          size=16, h=0.32)
    caption(s, 6.36, "数据来源：论文 Table 2（WWW 2026）", 11)


# ============================ 12 RQ2 ============================
def s12(prs, dk):
    s = dk.page("1.3 结果 RQ2：消融实验（论文 Table 3）",
                "在 Sports/GRU4Rec 上逐项移除组件，观察整体与尾部指标的降幅。",
                PART1, 13)
    add_table(s, MARGIN, TOP, 7.6, [3.3, 1.7, 1.7, 1.7], [
        ["变体", "总体 H@10", "尾物品 H@10", "尾用户 H@10"],
        ["完整 TADA", "**0.0279**", "**0.0095**", "**0.0273**"],
        ["w/o 相关性候选", "0.0273", "0.0094", "0.0265"],
        ["w/o 共现候选", "0.0248", "0.0094", "0.0243"],
        ["w/o 尾部算子", "0.0231", "0.0089", "0.0224"],
        ["w/o 跨序列增强", "0.0213", "0.0067", "0.0215"],
        ["原骨干 GRU4Rec", "0.0155", "0.0052", "0.0150"],
    ], row_h=0.48, header_h=0.50, size=17, where="p12表")
    block(s, MARGIN, 5.48, 7.4, 0.32, [
        dict(text="尾用户按训练序列长度划分，与头尾物品分组是两套口径。", size=16, color=GRAY),
    ], where="注释12")
    caption(s, 5.88, "数据来源：论文 Table 3（WWW 2026）", 12, w=7.4)

    card(s, 8.32, TOP, 4.39, 4.10, fill=TINT, line=LINE, accent=BLUE)
    block(s, 8.58, 1.98, 3.90, 3.85, [
        dict(text="原文结论", size=19, bold=True, color=BLUE, ls=1.0),
        dict(text="跨序列增强影响最大：整体 H@10 由 0.0279 降至 0.0213。", size=16, sb=10),
        dict(text="尾部算子次之，尾部指标同时下降。", size=16, sb=8),
        dict(text="相关性候选影响最小，但仍有小幅贡献。", size=16, sb=8),
        dict(text="四类组件叠加后，整体仍显著高于原骨干。", size=16, sb=8),
    ], where="p12侧栏")


# ============================ 13 RQ3 ============================
def s13(prs, dk):
    s = dk.page("1.3 结果 RQ3：扩展骨干（论文 Table 4）",
                "在 NARM、LightSANs、gMLP 上重复实验，检验方法是否依赖特定架构。",
                PART1, 14)
    add_table(s, MARGIN, TOP, 8.3, [1.3, 2.2, 1.9, 1.9, 1.6], [
        ["数据集", "骨干", "原结果 H@10", "+TADA H@10", "提升"],
        ["Toys", "NARM", "0.0371", "0.0500", "[[+34.8%]]"],
        ["Toys", "LightSANs", "0.0501", "0.0649", "[[+29.5%]]"],
        ["Toys", "gMLP", "0.0482", "0.0660", "[[+36.9%]]"],
        ["Beauty", "NARM", "0.0410", "0.0494", "[[+20.5%]]"],
        ["Beauty", "LightSANs", "0.0566", "0.0630", "[[+11.3%]]"],
        ["Beauty", "gMLP", "0.0473", "0.0658", "[[+39.1%]]"],
        ["Sports", "NARM", "0.0213", "0.0297", "[[+39.4%]]"],
        ["Sports", "LightSANs", "0.0212", "0.0293", "[[+38.2%]]"],
        ["Sports", "gMLP", "0.0217", "0.0317", "[[+46.1%]]"],
    ], row_h=0.40, header_h=0.44, size=17, where="p13表")
    caption(s, 6.14, "数据来源：论文 Table 4（WWW 2026）", 13)

    card(s, 9.28, TOP, 3.43, 3.55, fill=TINT2, line=None)
    block(s, 9.52, 2.00, 2.98, 3.3, [
        dict(text="论文报告的平均提升", size=18, bold=True, color=BLUE, ls=1.0),
        dict(text="整体 34.0%", size=17, sb=10),
        dict(text="头部 23.7%", size=17, sb=8),
        dict(text="尾部 44.9%", size=17, sb=8),
        dict(text="结论：不依赖特定架构，接入即可用。", size=15, sb=12),
    ], where="p13侧栏")


# ============================ 14 RQ4 ============================
def s14(prs, dk):
    s = dk.page("1.3 结果 RQ4：训练开销与尾物品覆盖率",
                "左表为训练时间（分钟），右表为尾物品覆盖率 TCov@5，均取自论文。",
                PART1, 15)
    add_table(s, MARGIN, TOP, 6.05, [2.4, 1.8, 1.8], [
        ["设置", "训练时间", "其中候选构建"],
        ["Beauty / GRU4Rec", "12.3 → 18.0", "0.7"],
        ["Beauty / FMLP-Rec", "21.3 → 25.9", "0.7"],
        ["Sports / GRU4Rec", "19.9 → 29.1", "1.7"],
        ["Sports / FMLP-Rec", "37.6 → 35.4", "1.7"],
    ], row_h=0.56, header_h=0.52, size=17, where="p14左表")
    block(s, MARGIN, 5.24, 6.05, 1.2, [
        dict(text="增强阶段为线性开销；候选构建只在训练前做一次。", size=16, bullet=True),
        dict(text="稠密相似矩阵与求逆是主要预处理成本，论文未计入。", size=16, bullet=True, sb=6),
    ], where="p14左文")

    add_table(s, 7.05, TOP, 5.70, [2.6, 1.5, 1.5], [
        ["设置", "原骨干", "TADA"],
        ["Beauty / GRU4Rec", "0.5719", "**0.5871**"],
        ["Beauty / FMLP-Rec", "0.6432", "**0.6777**"],
        ["Sports / GRU4Rec", "0.5381", "**0.5772**"],
        ["Sports / FMLP-Rec", "0.6916", "**0.7189**"],
    ], row_h=0.56, header_h=0.52, size=17, where="p14右表")
    block(s, 7.05, 5.24, 5.70, 1.2, [
        dict(text="覆盖率上升说明尾部物品被真正排进推荐结果，而不只是损失下降。", size=16, bullet=True),
        dict(text="CMR、RepPad 等基线的覆盖率并不总是高于原骨干。", size=16, bullet=True, sb=6),
    ], where="p14右文")
    caption(s, 6.48, "数据来源：论文 Table 6 与 Table 7（WWW 2026）", 14)


# ============================ 15 RQ5 ============================
def s15(prs, dk):
    s = dk.page("1.3 结果 RQ5：超参敏感性（论文 Fig.3 / Fig.4）",
                "上图扫描增强概率区间 (a, b)，下图扫描 Beta 参数 α；纵轴为尾部指标。",
                PART1, 16)
    s.shapes.add_picture(os.path.join(AST, "fig3.png"), Inches(1.20), Inches(1.90),
                         width=Inches(10.90))
    s.shapes.add_picture(os.path.join(AST, "fig4.png"), Inches(0.60), Inches(4.30),
                         width=Inches(12.10))
    caption(s, 6.46, "图 3：(a, b) 扫描（上）与 α 扫描（下），论文 Fig.3 / Fig.4", 15)
    block(s, MARGIN, 6.70, CW, 0.28, [
        dict(text="论文认为较优区间：(a,b) 取 (0.2,0.8)、(0.3,0.5)、(0.3,0.7)；α 取 0.3–0.5。",
             size=15, color=GRAY),
    ], where="p15注")


# ============================ 17 文件职责 ============================
def s17(prs, dk):
    s = dk.page("2.1 文件职责：源码的七个模块",
                "官方仓库 KingGugu/TADA 的源码结构，按“数据 → 候选 → 模型 → 训练”顺序阅读。",
                PART2, 18)
    add_table(s, MARGIN, TOP, CW, [3.0, 5.2, 3.9], [
        ["文件", "职责", "关键函数 / 类"],
        ["utils.py", "读入数据、留一划分、指标计算", "get_user_seqs"],
        ["datasets.py", "序列样本、两种尾部算子", "augment / substitute / insert"],
        ["ht_process.py", "头尾划分、邻接共现、偏好分组", "classify_head_and_tail 等"],
        ["LIS.py", "线性物品相似度与候选缓存", "get_linear_item_similarity"],
        ["models.py / modules.py", "SASRec、GRU4Rec、FMLPRec 三种骨干", "SASRecModel / FMLPRecModel"],
        ["trainers.py", "三项损失、全量排序评测", "iteration / whole_mix_learning"],
        ["main.py", "参数入口、预训练加载、实验管理", "main / --aug / --experiment"],
    ], row_h=0.50, header_h=0.50, size=15, where="p17表")
    notes(s, 6.04, "TADA 本身不新增可学习参数：所有改动集中在数据处理、候选缓存与训练循环三处。", 17,
          size=15, h=0.32)


# ============================ 18 候选集实现 ============================
def s18(prs, dk):
    s = dk.page("2.2 模块实现（1/4）：候选集构建",
                "对应论文式(2)(3)(4)，实现在 LIS.py 与 ht_process.py。",
                PART2, 19)
    block(s, MARGIN, TOP, CW, 3.9, [
        dict(text="**相关性候选**（LIS.py）", size=20, color=INK),
        dict(text="构造训练前缀二值矩阵 X，计算 XᵀX + λI 并求逆，得到 P。", size=18, bullet=True, sb=8),
        dict(text="按对角约束修正 γ，得到相似矩阵；对每个物品取 Top-K。", size=18, bullet=True, sb=6),
        dict(text="阈值过滤：相似度低于 th 的候选丢弃；为空时回退为前 K 个。", size=18, bullet=True, sb=6),
        dict(text="**共现候选**（ht_process.py）", size=20, color=INK, sb=12),
        dict(text="头物品：收集序列中前后相邻的尾物品；尾物品：收集所有前驱物品。", size=18, bullet=True, sb=8),
    ], where="p18文")
    card(s, MARGIN, 5.90, CW, 0.86, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.30, 5.98, CW - 0.6, 0.70, [
        dict(text="合并后按物品缓存为候选字典，训练时只做随机采样；共现统计只覆盖训练前缀，验证与测试目标不会泄漏。",
             size=16, color=INK),
    ], where="p18注")


# ============================ 19 头尾与共现 ============================
def s19(prs, dk):
    s = dk.page("2.2 模块实现（2/4）：头尾划分与偏好分组",
                "划分只使用训练前缀，避免把答案统计进候选。",
                PART2, 20)
    block(s, MARGIN, TOP, 6.15, 4.4, [
        dict(text="**头尾划分**（classify_head_and_tail）", size=20, color=INK),
        dict(text="统计 seq[:-2] 中物品频次，前 head_ratio=0.2 记为头物品。", size=18, bullet=True, sb=8),
        dict(text="用户按训练序列长度同样取前 20% 为头用户。", size=18, bullet=True, sb=6),
        dict(text="头尾集合写入分类文件，供候选与评测共用。", size=18, bullet=True, sb=6),
    ], where="p19左")
    block(s, 6.85, TOP, 5.90, 4.4, [
        dict(text="**偏好分组**（classify_user_preference）", size=20, color=INK),
        dict(text="计算每条序列的尾物品占比 tail_ratio。", size=18, bullet=True, sb=8),
        dict(text="tail_ratio ≥ 0.5 记为尾偏好用户，否则为头偏好用户。", size=18, bullet=True, sb=6),
        dict(text="分组用于跨序列增强：只在同组内打乱与混合。", size=18, bullet=True, sb=6),
    ], where="p19右")
    notes(s, 6.10, "注意两处阈值含义不同：ht_process 中硬编码的 0.5 是偏好阈值 β，命令行参数 th 是候选相似度阈值。",
          19, size=16, h=0.34)


# ============================ 20 算子实现 ============================
def s20(prs, dk):
    s = dk.page("2.2 模块实现（3/4）：尾部算子的代码路径",
                "对应论文式(5)(6)(9)(10)(11)，集中在 datasets.py 的 augment。",
                PART2, 21)
    block(s, MARGIN, TOP, CW, 4.0, [
        dict(text="**算子选择**", size=20, color=INK),
        dict(text="p_insert = 1 − len(seq)/max_len；短序列偏向插入，长序列偏向替换。", size=18,
             bullet=True, sb=8),
        dict(text="**T-Substitute**", size=20, color=INK, sb=12),
        dict(text="采样 p ~ U(a,b)，遍历头物品位置，按概率 p 从候选集随机取替换项。", size=18,
             bullet=True, sb=8),
        dict(text="**T-Insert**", size=20, color=INK, sb=12),
        dict(text="采样同一个 p，遍历尾物品位置，在该位置前插入候选物品，并复制原尾物品得到扩展视图。",
             size=18, bullet=True, sb=8),
        dict(text="两种算子共用同一候选字典与位置 mask，保证编码时对齐。", size=18, bullet=True, sb=6),
    ], where="p20文")
    notes(s, 5.92, "候选集合以列表形式保存，采样时随机取；若需严格逐位复现，需要固定集合遍历顺序与数据加载的随机状态。",
          20, size=16, h=0.60)


# ============================ 21 训练实现 ============================
def s21(prs, dk):
    s = dk.page("2.2 模块实现（4/4）：训练循环与三项损失",
                "对应论文式(7)(8)(12)(13)(14)–(17)，实现在 trainers.py。",
                PART2, 22)
    add_table(s, MARGIN, TOP, CW, [2.4, 6.0, 3.4], [
        ["函数", "做什么", "对应公式"],
        ["iteration", "取 batch、前向、三项 BCE 累加、反向更新", "式(14)–(17)"],
        ["self_mix_learning", "两视图共享编码器，按 Beta 权重逐位置混合", "式(7)(8)(15)"],
        ["whole_mix_learning", "按偏好分组打乱并混合表示与目标嵌入", "式(12)(13)(16)"],
        ["get_full_sort_score", "屏蔽已见物品，全量排序并分组统计", "评测协议"],
    ], row_h=0.62, header_h=0.52, size=17, where="p21表")
    block(s, MARGIN, 4.86, CW, 1.6, [
        dict(text="阶段一：只用 L_main 训练，产出 *-original.pt 预训练权重。", size=18, bullet=True),
        dict(text="阶段二：aug=1 时加载同骨干同数据集的权重，再按式(17) 联合优化。", size=18, bullet=True, sb=6),
        dict(text="检查点按验证集总体 NDCG@20 选择，测试阶段使用原始序列做全量排序。", size=18, bullet=True, sb=6),
    ], where="p21文")


# ============================ 22 参数 1 ============================
def s22(prs, dk):
    s = dk.page("2.3 参数设置（1/2）：符号与默认值",
                "符号按论文记法，取值以代码默认值为准。",
                PART2, 23)
    add_table(s, MARGIN, TOP, CW, [1.9, 5.4, 4.5], [
        ["符号", "含义", "取值"],
        ["N", "最大序列长度", "50"],
        ["B", "batch size", "256"],
        ["d", "嵌入维度 hidden_size", "64"],
        ["lr", "Adam 学习率", "0.001"],
        ["α（代码 beta）", "Beta 分布参数", "0.3"],
        ["a, b", "增强概率采样区间", "0.31 / 0.51（部分骨干 0.61 / 0.81）"],
        ["β（代码 th_pref）", "头尾偏好阈值", "0.5（硬编码）"],
        ["K", "候选数量 top_k", "10"],
    ], row_h=0.44, header_h=0.46, size=17, where="p22表")
    notes(s, 5.94, "其他：epochs = 500、patience = 10、weight_decay = 0、full_sort 开启；SASRec/GRU4Rec 2 层，FMLPRec 4 层。",
          22, size=16, h=0.34)


# ============================ 23 参数 2 ============================
def s23(prs, dk):
    s = dk.page("2.3 参数设置（2/2）：论文设定与本地命令",
                "本地按论文记录逐项传参，未能开放的项在说明列注明。",
                PART2, 24)
    add_table(s, MARGIN, TOP, CW, [2.7, 3.0, 3.0, 3.4], [
        ["参数", "论文设定", "本地命令", "说明"],
        ["embedding / batch", "64 / 256", "64 / 256", "对齐"],
        ["lr / epochs", "0.001 / —", "0.001 / 500", "对齐"],
        ["α", "{0.1, …, 0.5}", "0.3", "默认值，未扫描"],
        ["(a, b)", "网格搜索", "0.31/0.51 或 0.61/0.81", "按骨干分别设置"],
        ["K", "{10, 20, 30, 40, 50}", "10", "默认值"],
        ["偏好阈值 β", "{0.4, 0.5, 0.6}", "0.5", "**代码内硬编码**"],
        ["运行次数", "5 次取均值", "单次（Beauty 补 5 个种子）", "**未做 5 次平均**"],
    ], row_h=0.50, header_h=0.50, size=16, where="p23表")
    notes(s, 5.98, "命名需要区分：命令行 --beta 是 Beta 参数 α；ht_process 里的 0.5 才是偏好阈值 β；--th 是候选相似度阈值。",
          23, size=16, h=0.34)


# ============================ 24 数据处理 ============================
def s24(prs, dk):
    s = dk.page("2.3 数据处理：留一划分与训练前缀",
                "所有统计只基于训练前缀，验证与测试目标不参与候选构建。",
                PART2, 25)
    block(s, MARGIN, TOP, CW, 4.2, [
        dict(text="**留一划分**", size=20, color=INK),
        dict(text="每条序列最后一项为测试目标，倒数第二项为验证目标。", size=18, bullet=True, sb=8),
        dict(text="训练输入与逐位置目标错开一位，序列左填充到 N 并记录 mask。", size=18, bullet=True, sb=6),
        dict(text="**训练前缀**", size=20, color=INK, sb=12),
        dict(text="头尾频次、共现邻接、线性相似都只在 seq[:-2] 上统计。", size=18, bullet=True, sb=8),
        dict(text="这样候选集不会包含答案信息，评估口径才成立。", size=18, bullet=True, sb=6),
        dict(text="**评测**", size=20, color=INK, sb=12),
        dict(text="全量排序（full_sort），屏蔽训练中已交互物品，报告 H@5/10/20 与 N@5/10/20。", size=18,
             bullet=True, sb=8),
    ], where="p24文")
    card(s, MARGIN, 5.90, CW, 0.88, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.30, 5.98, CW - 0.6, 0.72, [
        dict(text="易错点：候选统计必须只用训练前缀；一旦把验证或测试目标计入，头部物品会被高估，整体指标随之虚高。",
             size=15, color=INK),
    ], where="p24补")


# ============================ 25 模型设定 ============================
def s25(prs, dk):
    s = dk.page("2.3 模型设定：三种骨干与共享编码器",
                "TADA 只要求骨干能输出可混合的序列表示，并与 BCE 训练接口兼容。",
                PART2, 26)
    block(s, MARGIN, TOP, 6.15, 4.3, [
        dict(text="**SASRec**", size=20, color=INK),
        dict(text="自注意力 + 因果 mask，2 层，默认骨干。", size=18, bullet=True, sb=8),
        dict(text="**GRU4Rec**", size=20, color=INK, sb=12),
        dict(text="门控循环网络，2 层，训练波动相对更大。", size=18, bullet=True, sb=8),
        dict(text="**FMLPRec**", size=20, color=INK, sb=12),
        dict(text="频域滤波 + MLP，4 层，滤波层固定启用。", size=18, bullet=True, sb=8),
    ], where="p25左")
    block(s, 6.85, TOP, 5.90, 4.3, [
        dict(text="**两视图共享**", size=20, color=INK),
        dict(text="原始与增强序列走同一嵌入表与同一编码器，参数不复制。", size=18, bullet=True, sb=8),
        dict(text="**接入成本**", size=20, color=INK, sb=12),
        dict(text="不新增模块、不改损失接口；只需多一次前向即可得到增强表示。", size=18, bullet=True, sb=8),
        dict(text="这也是论文能横跨三种架构的原因。", size=18, bullet=True, sb=6),
    ], where="p25右")


# ============================ 26 模型训练 ============================
def s26(prs, dk):
    s = dk.page("2.3 模型训练：两阶段、早停与选点",
                "官方流程是先训原骨干，再加增强训练；两阶段共用同一套数据与评测。",
                PART2, 27)
    block(s, MARGIN, TOP, CW, 4.2, [
        dict(text="**阶段一：原骨干训练**", size=20, color=INK),
        dict(text="只优化 L_main，得到 *-original.pt；论文与仓库都提供该权重。", size=18, bullet=True, sb=8),
        dict(text="**阶段二：增强训练**", size=20, color=INK, sb=12),
        dict(text="加载对应权重后启用两个算子与跨序列增强，优化三项损失之和。", size=18, bullet=True, sb=8),
        dict(text="**早停与选点**", size=20, color=INK, sb=12),
        dict(text="按验证集总体 NDCG@20 选择最佳检查点，patience = 10，epoch 上限 500。", size=18,
             bullet=True, sb=8),
        dict(text="**两阶段衔接**：阶段二只加载同骨干、同数据集的权重，避免跨数据集混用。", size=18,
             bullet=True, sb=12),
        dict(text="**实验管理**：--experiment 归档配置、日志与权重，已完成且配置一致的实验自动跳过。", size=18,
             bullet=True, sb=10),
    ], where="p26文")
    card(s, MARGIN, 6.06, CW, 0.74, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.30, 6.14, CW - 0.6, 0.58, [
        dict(text="训练成本：阶段二每个 batch 多一次前向，实测总耗时约为原骨干的 1.0–1.5 倍（论文 Table 6）。",
             size=15, color=INK),
    ], where="p26补")


# ============================ 27 流程 ============================
def s27(prs, dk):
    s = dk.page("2.4 训练执行流程：从数据到指标",
                "入口为 main.py / run_experiments.py，流程可拆成七步。",
                PART2, 28)
    steps = [("读数据", "留一划分"), ("训练前缀", "seq[:-2]"), ("头尾统计", "频次 20%"),
             ("候选缓存", "相似 + 共现"), ("阶段一", "仅 L_main"), ("增强训练", "三项损失"),
             ("选点测试", "全量排序")]
    bw, gap = 1.56, 0.26
    x = MARGIN
    for i, (name, desc) in enumerate(steps):
        solid = i in (0, 6)
        round_box(s, x, 1.92, bw, 1.00, fill=BLUE if solid else WHITE, line=BLUE_LT)
        block(s, x + 0.06, 2.02, bw - 0.12, 0.32,
              [dict(text=name, size=16, bold=True, color=WHITE if solid else INK, ls=1.0)],
              align=PP_ALIGN.CENTER, where="p27流程名")
        block(s, x + 0.10, 2.38, bw - 0.20, 0.46,
              [dict(text=desc, size=14, color=WHITE if solid else GRAY, ls=1.05)],
              align=PP_ALIGN.CENTER, where="p27流程文")
        if i < len(steps) - 1:
            arrow(s, x + bw + 0.03, 2.30, 0.20, 0.24)
        x += bw + gap

    card(s, MARGIN, 3.24, CW, 1.45, fill=TINT, line=LINE, accent=BLUE)
    block(s, MARGIN + 0.30, 3.36, CW - 0.6, 1.2, [
        dict(text="**两条时间线**", size=19, color=INK, ls=1.0),
        dict(text="预处理：读数据、划头尾、建候选，一次性完成并缓存。", size=17, sb=6),
        dict(text="训练：阶段一训原骨干，阶段二在每个 batch 内生成增强视图并累加三项损失。", size=17, sb=4),
    ], where="p27说明")

    items = [("骨干默认", "SASRec"), ("embedding", "64"), ("batch_size", "256"),
             ("epochs", "500"), ("patience", "10"), ("显卡", "本地单卡 8 GB")]
    bw2, gap2 = 3.86, 0.30
    for i, (k, v) in enumerate(items):
        r_, c_ = divmod(i, 3)
        x = MARGIN + c_ * (bw2 + gap2)
        y = 4.90 + r_ * 0.86
        card(s, x, y, bw2, 0.72, fill=WHITE, line=LINE, radius=0.1)
        block(s, x + 0.22, y + 0.08, bw2 - 0.4, 0.26,
              [dict(text=k, size=15, color=GRAY, ls=1.0)], where="p27参数名")
        block(s, x + 0.22, y + 0.35, bw2 - 0.4, 0.30,
              [dict(text=v, size=19, bold=True, color=INK, ls=1.0)], where="p27参数值")


# ============================ 28 增强示例 ============================
def s28(prs, dk):
    s = dk.page("2.4 增强示例：一条序列的三种视图",
                "头物品替换后引入尾部信号；插入则在尾物品前补上上下文，并复制尾物品对齐。",
                PART2, 29)
    # 原始序列
    def seq_row(y, labels, colors, title, x0=1.30, bw=0.86, gap=0.10):
        block(s, 0.72, y + 0.10, 0.62, 0.30,
              [dict(text=title, size=15, color=GRAY, ls=1.0)], where="p28行标")
        x = x0
        for lab, col in zip(labels, colors):
            fill = TINT2 if col == "h" else RGB_TAIL
            round_box(s, x, y, bw, 0.50, fill=fill, line=BLUE_LT, radius=0.15)
            block(s, x, y + 0.10, bw, 0.30,
                  [dict(text=lab, size=16, bold=True, color=INK, ls=1.0)],
                  align=PP_ALIGN.CENTER, where="p28元素")
            x += bw + gap

    from pptx.dml.color import RGBColor
    RGB_TAIL = RGBColor(0xFB, 0xE3, 0xD0)

    seq_row(2.05, ["h1", "h2", "t1", "h3", "h4"], ["h", "h", "t", "h", "h"], "原始")
    seq_row(2.95, ["h1", "t2", "t1", "h3", "h4"], ["h", "t", "t", "h", "h"], "替换")
    seq_row(3.85, ["h1", "h2", "t2", "t1", "h3"], ["h", "h", "t", "t", "h"], "插入")
    seq_row(4.75, ["h1", "h2", "t1", "t1", "h3"], ["h", "h", "t", "t", "h"], "扩展")

    block(s, 0.72, 5.52, 8.3, 1.25, [
        dict(text="替换：头物品 h2 按概率 p 换成候选 t2，尾部物品曝光增加。", size=15, bullet=True),
        dict(text="插入：在尾物品 t1 前插入候选 t2，补上“引出尾部”的上下文。", size=15, bullet=True, sb=5),
        dict(text="扩展：复制插入位置的 t1，使两视图等长，便于逐位置混合。", size=15, bullet=True, sb=5),
    ], where="p28说明")

    card(s, 9.35, 1.95, 3.36, 3.40, fill=TINT, line=LINE, accent=BLUE)
    block(s, 9.58, 2.06, 2.95, 3.15, [
        dict(text="读图说明", size=18, bold=True, color=BLUE, ls=1.0),
        dict(text="蓝色：头物品　橙色：尾物品", size=15, sb=8),
        dict(text="替换只在头物品位置发生，候选来自该物品的集合。", size=15, sb=8),
        dict(text="插入只在尾物品位置发生，插入项来自候选集合。", size=15, sb=8),
        dict(text="两种算子共用同一个采样概率 p。", size=15, sb=8),
    ], where="p28侧栏")


# ============================ 29 复现主结果 ============================
def s29(prs, dk):
    s = dk.page("2.5 复现结果（1/5）：主结果对照",
                "以代码自带的论文数据为基准，按同一配置复刻；九组中八组总体偏差在 ±2.15%。",
                PART2, 30)
    add_table(s, MARGIN, TOP, 8.0, [1.4, 2.0, 2.2, 2.0, 1.7], [
        ["数据集", "骨干", "论文数据 H@10", "复刻 H@10", "相对变化"],
        ["Toys", "GRU4Rec", "0.0512", "0.0501", "[[−2.15%]]"],
        ["Toys", "SASRec", "0.0846", "0.0859", "[[+1.54%]]"],
        ["Toys", "FMLP-Rec", "0.0664", "0.0671", "[[+1.05%]]"],
        ["Beauty", "GRU4Rec", "0.0555", "0.0515", "[[−7.21%]]"],
        ["Beauty", "SASRec", "0.0732", "0.0745", "[[+1.78%]]"],
        ["Beauty", "FMLP-Rec", "0.0658", "0.0668", "[[+1.52%]]"],
        ["Sports", "GRU4Rec", "0.0279", "0.0275", "[[−1.43%]]"],
        ["Sports", "SASRec", "0.0402", "0.0402", "[[0.00%]]"],
        ["Sports", "FMLP-Rec", "0.0304", "0.0301", "[[−0.99%]]"],
    ], row_h=0.38, header_h=0.44, size=17, where="p29表")
    caption(s, 6.26, "数据来源：本地实验日志（2026 年 10 月，seed=2025/2024）；基准为论文数据", 29)

    card(s, 8.85, TOP, 3.86, 3.60, fill=TINT2, line=None)
    block(s, 9.10, 2.00, 3.36, 3.35, [
        dict(text="结论", size=18, bold=True, color=BLUE, ls=1.0),
        dict(text="SASRec 与 FMLP-Rec 更贴近论文，偏差多在 2% 以内。", size=15, sb=8),
        dict(text="GRU4Rec/Beauty 低 7.21%，是唯一超过 5% 的组合。", size=15, sb=8),
        dict(text="本地只复现了 TADA，没有 CMR、RepPad 等基线的本地对照。", size=15, sb=8),
    ], where="p29侧栏")


# ============================ 30 头尾分组 ============================
def s30(prs, dk):
    s = dk.page("2.5 复现结果（2/5）：头尾分组对照",
                "总体接近不代表分组一致，下表为相对论文数据的 Top-10 变化。",
                PART2, 31)
    add_table(s, MARGIN, TOP, CW, [1.3, 2.0, 1.9, 1.9, 1.9, 1.9], [
        ["数据集", "骨干", "头物品 H@10", "尾物品 H@10", "头用户 H@10", "尾用户 H@10"],
        ["Toys", "GRU4Rec", "[[+0.80%]]", "[[−5.71%]]", "[[−1.94%]]", "[[−2.15%]]"],
        ["Toys", "SASRec", "[[+1.42%]]", "[[+1.40%]]", "[[+4.37%]]", "[[+0.70%]]"],
        ["Toys", "FMLP-Rec", "[[−3.96%]]", "[[+6.63%]]", "[[+3.24%]]", "[[+0.75%]]"],
        ["Beauty", "GRU4Rec", "[[−9.54%]]", "[[+2.05%]]", "[[−6.80%]]", "[[−7.38%]]"],
        ["Beauty", "SASRec", "[[+0.69%]]", "[[+3.91%]]", "[[+4.76%]]", "[[+0.74%]]"],
        ["Beauty", "FMLP-Rec", "[[+1.91%]]", "[[+0.62%]]", "[[+1.13%]]", "[[+1.55%]]"],
        ["Sports", "GRU4Rec", "[[−1.35%]]", "[[−3.16%]]", "[[−4.90%]]", "[[−0.37%]]"],
        ["Sports", "SASRec", "[[−0.31%]]", "[[+0.96%]]", "[[−1.05%]]", "[[+0.25%]]"],
        ["Sports", "FMLP-Rec", "[[−1.45%]]", "[[+0.60%]]", "[[+1.68%]]", "[[−1.31%]]"],
    ], row_h=0.40, header_h=0.44, size=16, where="p30表")
    notes(s, 5.92, "GRU4Rec/Beauty 的下降集中在头物品与头尾用户，而尾物品反而高 2.05%：不能简单概括为“尾部没提升”。",
          30, size=16, h=0.34)
    caption(s, 6.34, "数据来源：本地实验日志；比较基准为论文数据", 30)


# ============================ 31 种子实验 ============================
def s31(prs, dk):
    s = dk.page("2.5 复现结果（3/5）：Beauty 随机种子对照",
                "针对 GRU4Rec/Beauty 的偏差，固定其余配置，只更换随机种子。",
                PART2, 32)
    add_table(s, MARGIN, TOP, 7.9, [2.4, 1.4, 2.0, 1.9, 2.0], [
        ["运行", "seed", "总体 H@10", "总体 N@10", "尾物品 H@10"],
        ["论文数据", "2025", "0.0555", "0.0279", "0.0195"],
        ["复刻 model-3", "2025", "0.0515", "0.0265", "0.0199"],
        ["model-4", "2026", "0.0519", "0.0270", "0.0203"],
        ["model-5", "42", "0.0530", "0.0275", "0.0209"],
        ["model-6", "520", "0.0542", "0.0283", "0.0189"],
        ["model-7", "1314", "0.0518", "0.0265", "0.0199"],
        ["五个 seed 均值", "—", "0.0525", "0.0272", "0.0200"],
    ], row_h=0.44, header_h=0.46, size=16, where="p31表")
    caption(s, 5.98, "数据来源：本地实验日志（2026 年 10 月），均值 ± 标准差：H@10 0.0525 ± 0.0011", 31)

    card(s, 8.75, TOP, 3.96, 3.55, fill=TINT, line=LINE, accent=BLUE)
    block(s, 9.00, 1.98, 3.46, 3.30, [
        dict(text="结果解读", size=18, bold=True, color=BLUE, ls=1.0),
        dict(text="五个种子的 H@10 落在 0.0515–0.0542。", size=15, sb=8),
        dict(text="均值 0.0525 仍低于论文数据的 0.0555。", size=15, sb=8),
        dict(text="随机性解释了一部分差距，但无法完全解释。", size=15, sb=8),
        dict(text="预训练权重、实现版本与环境差异尚未排除。", size=15, sb=8),
    ], where="p31侧栏")


# ============================ 32 消融 ============================
def s32(prs, dk):
    s = dk.page("2.5 复现结果（4/5）：消融实验",
                "在 Sports 上分别对 GRU4Rec 与 FMLPRec 做组件移除，seed 固定为 2025。",
                PART2, 33)
    block(s, MARGIN, 1.78, 6.0, 0.32,
          [dict(text="Sports / GRU4Rec（总体 H@10）", size=17, bold=True, color=BLUE, ls=1.0)],
          where="p32左题")
    add_table(s, MARGIN, 2.14, 5.90, [3.0, 1.6], [
        ["设置", "总体 H@10"],
        ["完整 TADA（复刻）", "**0.0275**"],
        ["w/o 相关性候选", "0.0247"],
        ["w/o 共现候选", "0.0246"],
        ["w/o 尾部算子", "0.0234"],
        ["w/o 跨序列增强", "0.0210"],
        ["仅主任务损失", "0.0174"],
    ], row_h=0.44, header_h=0.44, size=16, where="p32左表")

    block(s, 6.85, 1.78, 6.0, 0.32,
          [dict(text="Sports / FMLPRec（总体 H@10）", size=17, bold=True, color=BLUE, ls=1.0)],
          where="p32右题")
    add_table(s, 6.85, 2.14, 5.85, [3.0, 1.6], [
        ["设置", "总体 H@10"],
        ["完整 TADA（复刻）", "**0.0301**"],
        ["w/o 相关性候选", "0.0264"],
        ["w/o 共现候选", "0.0325"],
        ["w/o 尾部算子", "0.0217"],
        ["w/o 跨序列增强", "0.0279"],
        ["仅主任务损失", "0.0224"],
    ], row_h=0.44, header_h=0.44, size=16, where="p32右表")
    block(s, MARGIN, 5.28, CW, 1.3, [
        dict(text="GRU4Rec：跨序列增强与仅主任务损失降幅最大，排序与论文一致。", size=17, bullet=True),
        dict(text="FMLPRec：去掉共现候选反而升高 8%，说明模块作用随架构与随机状态变化。", size=17,
             bullet=True, sb=6),
        dict(text="更换 seed（1314）后五组总体 H@10 落在 0.0215–0.0299，模块排序同样改变。", size=17,
             bullet=True, sb=6),
    ], where="p32文")
    caption(s, 6.62, "数据来源：本地实验日志（2026 年 10 月）", 32)


# ============================ 33 参数扫描 ============================
def s33(prs, dk):
    s = dk.page("2.5 复现结果（5/5）：超参扫描",
                "在 Sports/FMLPRec 上扫描 (a, b) 与 α，每组只改一个变量。",
                PART2, 34)
    add_table(s, MARGIN, TOP, 7.60, [1.3, 1.0, 3.9, 2.3, 2.3], [
        ["参数", "组数", "取值范围", "尾物品 H@10", "尾用户 H@10"],
        ["(a, b)", "12", "a∈{0.1,0.2,0.3}，b∈{0.5,…,0.8}", "0.0142–0.0157", "0.0255–0.0285"],
        ["α", "5", "0.1 ~ 0.5", "0.0157–0.0167", "0.0284–0.0311"],
    ], row_h=0.64, header_h=0.52, size=14, where="p33表")
    block(s, MARGIN, 3.90, 7.30, 2.4, [
        dict(text="(a, b) 的尾部平均最高出现在 (0.2, 0.7)，为 0.0221。", size=17, bullet=True),
        dict(text="α 的尾部平均在 0.3 附近最高；0.1 时尾用户指标更好。", size=17, bullet=True, sb=6),
        dict(text="两组扫描的极差都在 10% 量级，单 seed 下不足以确定最优区间。", size=17, bullet=True, sb=6),
        dict(text="结论：在论文给定范围内，TADA 对 (a, b) 与 α 不敏感。", size=17, bullet=True, sb=6),
    ], where="p33文")

    s.shapes.add_picture(os.path.join(CH, "parameter_ab_heatmap.png"), Inches(8.35),
                         Inches(1.90), width=Inches(4.35))
    caption(s, 3.24, "图 4：(a, b) 网格热力图（三个面板）", 33, x=8.35, w=4.4, size=14)
    s.shapes.add_picture(os.path.join(CH, "parameter_alpha_lines.png"), Inches(9.00),
                         Inches(3.62), width=Inches(3.30))
    caption(s, 5.86, "图 5：α 扫描的尾物品与尾用户 H@10", 33, x=9.00, w=3.4, size=14)


# ============================ 34 工程问题 ============================
def s34(prs, dk):
    s = dk.page("2.5 工程问题与修复：2 项",
                "两项都不涉及算法，但会直接影响能否跑通与日志是否可信。",
                PART2, 35)
    add_table(s, MARGIN, TOP, CW, [3.4, 5.0, 3.6], [
        ["现象", "原因", "处理"],
        ["FMLPRec 加载预训练权重报 Unexpected key(s)",
         "默认构建 2 层编码器，而三份权重都是 4 层", "按骨干设层数（FMLPRec = 4），以 strict=True 验证加载"],
        ["--no_filters=False 被解析为 True",
         "argparse 使用 type=bool，任何非空字符串都为真", "改为显式解析 true/false；FMLPRec 统一启用滤波层"],
    ], row_h=1.05, header_h=0.52, size=16, where="p34表")
    notes(s, 5.40, "第一项若不修，多出的两层参数会被静默忽略，后续所有对比都建立在错误结构上。", 34,
          size=16, h=0.34)

    card(s, MARGIN, 5.95, CW, 0.86, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.30, 6.02, CW - 0.6, 0.70, [
        dict(text="复现结论的前提：权重结构、参数解析、评测口径三者一致，否则指标差异无法归因于方法本身。",
             size=16, color=INK),
    ], where="p34注")


# ============================ 35 实验记录 ============================
def s35(prs, dk):
    s = dk.page("2.6 实验记录：规模与口径",
                "全部实验使用同一套数据划分与评测代码，结果取最佳检查点的测试输出。",
                PART2, 36)
    add_table(s, MARGIN, TOP, 6.30, [2.2, 1.6, 3.5], [
        ["实验类型", "组数", "内容"],
        ["主实验", "9", "3 骨干 × 3 数据集"],
        ["种子对照", "5", "GRU4Rec/Beauty"],
        ["消融（GRU4Rec）", "6", "完整 + 5 种移除"],
        ["消融（FMLPRec）", "10", "两个 seed 各 5 组"],
        ["参数扫描", "16", "(a,b) 12 组 + α 4 组"],
    ], row_h=0.62, header_h=0.48, size=16, where="p35左表")
    block(s, 7.05, TOP, 5.70, 4.2, [
        dict(text="**统一口径**", size=19, color=INK),
        dict(text="数据划分、候选构建、评测代码完全一致。", size=17, bullet=True, sb=8),
        dict(text="结果取验证集选点后的测试输出，不取训练过程指标。", size=17, bullet=True, sb=6),
        dict(text="**已知边界**", size=19, color=INK, sb=12),
        dict(text="未复现 CMR、CMRSI、RepPad、CITIES、LOAM、MELT 基线。", size=17, bullet=True, sb=8),
        dict(text="主实验为单次运行，论文为 5 次平均。", size=17, bullet=True, sb=6),
    ], where="p35右")
    caption(s, 6.30, "合计 46 组训练；每组日志记录完整命令、验证曲线与最终测试指标", 35)


# ============================ 36 图1 ============================
def s36(prs, dk):
    s = dk.page("2.6 结果图（1/4）：主结果复现",
                "每个面板对应一个数据集，实心柱为论文数据，空心柱为复刻结果。",
                PART2, 37)
    s.shapes.add_picture(os.path.join(CH, "reproduction_id1_id3.png"), Inches(1.10),
                         Inches(2.02), width=Inches(11.10))
    caption(s, 5.78, "图 6：论文数据与复刻结果的总体指标（2026 年 10 月）", 36)
    block(s, MARGIN, 6.10, CW, 0.9, [
        dict(text="三组总体指标的柱高基本一致，SASRec 与 FMLPRec 的差值在 2% 以内。", size=17, bullet=True),
        dict(text="GRU4Rec/Beauty 的四个总体指标整体偏低，是唯一需要单独排查的组合。", size=17,
             bullet=True, sb=6),
    ], where="p36文")


# ============================ 37 图2 ============================
def s37(prs, dk):
    s = dk.page("2.6 结果图（2/4）：Beauty 种子对照",
                "五个本地种子与论文数据的总体指标对比。",
                PART2, 38)
    s.shapes.add_picture(os.path.join(CH, "reproduction_beauty_seeds.png"), Inches(0.95),
                         Inches(2.02), width=Inches(7.05))
    caption(s, 6.42, "图 7：GRU4Rec/Beauty 的五次运行（seed 2025–1314）", 37, x=0.95, w=7.1)
    block(s, 8.30, 2.10, 4.40, 4.2, [
        dict(text="**读法**", size=19, color=INK),
        dict(text="横轴为运行编号，纵轴为四项总体指标。", size=16, sb=8),
        dict(text="论文数据作为基准线，其余为本地复刻。", size=16, sb=6),
        dict(text="**结论**", size=19, color=INK, sb=14),
        dict(text="五个种子的波动范围约 ±0.0014。", size=16, sb=8),
        dict(text="均值仍低于基准，说明还有系统性差距。", size=16, sb=6),
    ], where="p37文")


# ============================ 38 图3 ============================
def s38(prs, dk):
    s = dk.page("2.6 结果图（3/4）：Sports/GRU4Rec 消融",
                "移除跨序列增强或只保留主任务时，三项指标同时大幅下降；相关性候选的影响相对最小。",
                PART2, 39)
    s.shapes.add_picture(os.path.join(CH, "ablation_sports_gru4rec.png"), Inches(1.85),
                         Inches(1.88), width=Inches(9.60))
    caption(s, 6.46, "图 8：Sports/GRU4Rec 消融（2026 年 10 月）；最左两根为论文与复刻的完整模型", 38)


# ============================ 39 图4 ============================
def s39(prs, dk):
    s = dk.page("2.6 结果图（4/4）：(a, b) 与 α 扫描",
                "热力图与折线分别对应两个超参，颜色越深表示尾巴指标越高。",
                PART2, 40)
    s.shapes.add_picture(os.path.join(CH, "parameter_ab_heatmap.png"), Inches(0.80),
                         Inches(2.02), width=Inches(7.60))
    s.shapes.add_picture(os.path.join(CH, "parameter_alpha_lines.png"), Inches(8.80),
                         Inches(2.02), width=Inches(3.90))
    caption(s, 4.28, "图 9：(a, b) 网格热力图（左）与 α 折线（右）", 39, size=14)
    block(s, MARGIN, 4.66, CW, 1.6, [
        dict(text="热力图三张面板依次为尾物品、尾用户与两者的平均 H@10；最优区域集中在 a = 0.2–0.3、b = 0.6–0.7。", size=17,
             bullet=True),
        dict(text="α 折线显示尾物品在 0.3–0.4 较高，尾用户在 0.1 较高，两组指标的最优值并不一致。", size=17,
             bullet=True, sb=6),
        dict(text="整体变化幅度有限，说明在论文给定范围内 TADA 对这两个超参较鲁棒。", size=17,
             bullet=True, sb=6),
    ], where="p39文")


# ============================ 40 小结 ============================
def s40(prs, dk):
    s = dk.page("小结：复现结论",
                "汇总 46 组训练，区分复现程度与仍需排查的部分。",
                PART2, 41)
    card(s, MARGIN, 1.88, 5.90, 3.55, fill=TINT, line=LINE)
    rect(s, MARGIN + 0.02, 1.88, 5.86, 0.06, BLUE)
    block(s, MARGIN + 0.30, 2.08, 5.30, 3.2, [
        dict(text="已复现", size=22, bold=True, color=BLUE, ls=1.0),
        dict(text="主结果：九组中八组的总体偏差在 ±2.15% 以内。", size=17, bullet=True, sb=14),
        dict(text="消融：GRU4Rec 上的模块排序与论文一致。", size=17, bullet=True, sb=8),
        dict(text="参数：变化幅度有限，与论文的鲁棒性判断一致。", size=17, bullet=True, sb=8),
    ], where="p40左")
    card(s, 6.82, 1.88, 5.89, 3.55, fill=WHITE, line=LINE)
    rect(s, 6.84, 1.88, 5.85, 0.06, BLUE_LT)
    block(s, 7.12, 2.08, 5.29, 3.2, [
        dict(text="未能复现", size=22, bold=True, color=GRAY, ls=1.0),
        dict(text="GRU4Rec/Beauty 低 7.21%，五个种子仍低于论文。", size=17, bullet=True, sb=14),
        dict(text="FMLPRec 去掉共现候选反而升高，与论文方向不一致。", size=17, bullet=True, sb=8),
        dict(text="未复现任何基线方法，绝对增益无法本地校验。", size=17, bullet=True, sb=8),
    ], where="p40右")

    card(s, MARGIN, 5.62, CW, 0.86, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.30, 5.72, CW - 0.6, 0.76, [
        dict(text="结论：TADA 在 SASRec 上复现最稳定；GRU4Rec 波动较大，差距部分来自随机性，部分仍待排查。",
             size=18, color=INK),
    ], where="p40总结")


# ============================ 42 学习总结 ============================
def s42(prs, dk):
    s = dk.page("3.1 本学期的工作与产出",
                "五项工作及其对应产出文件。", PART3, 43)
    add_table(s, MARGIN, TOP, CW, [2.2, 6.5, 3.3], [
        ["方面", "工作内容", "产出"],
        ["论文复现", "精读 TADA，完成 46 组训练，修复 2 个问题", "复现记录 + 训练日志"],
        ["实验设计", "主实验、种子对照、两组消融、两组参数扫描", "实验表与图表"],
        ["推荐系统", "梳理长尾推荐与数据增强两条线的代表方法", "推荐系统笔记"],
        ["文献阅读", "精读 2 篇（ADARec、TADA），泛读同类工作", "文献阅读笔记"],
        ["前沿跟踪", "Agent、世界模型、LLM×Rec 等方向的术语与进展", "前沿术语速查"],
    ], row_h=0.70, header_h=0.54, size=17, where="p42表")
    notes(s, 6.10, "全部笔记均标注论文出处与对应代码位置，便于复查。", 42, size=17, h=0.32)


# ============================ 43 后续计划 ============================
def s43(prs, dk):
    s = dk.page("3.2 后续学习计划",
                "五项均为本学期实际受阻的环节，按优先级排序。", PART3, 44)
    items = [
        ("01", "基线复现", "补齐 CMR、RepPad 等基线的本地结果，才能校验 TADA 的相对增益。"),
        ("02", "多种子统计", "主实验统一跑 5 个种子并报告均值与标准差，减少单次运行的偶然性。"),
        ("03", "偏差归因", "针对 GRU4Rec/Beauty 的差距，核对预训练权重、实现版本与数据加载顺序。"),
        ("04", "英文阅读", "每周精读一篇原文，重点核对术语与公式推导，减少对翻译工具的依赖。"),
        ("05", "方向选择", "从“增强模块在不同骨干上作用不一致”切入，寻找可解释的改进点。"),
    ]
    y = 1.90
    for code, title, desc in items:
        card(s, MARGIN, y, CW, 0.90, fill=TINT if code in ("01", "03", "05") else WHITE,
             line=LINE)
        block(s, MARGIN + 0.25, y + 0.20, 0.7, 0.5,
              [dict(text=code, size=21, bold=True, color=BLUE_LT, ls=1.0)], where="p43号")
        block(s, MARGIN + 0.95, y + 0.06, 2.1, 0.78,
              [dict(text=title, size=20, bold=True, color=INK, ls=1.0)],
              anchor=MSO_ANCHOR.MIDDLE, where="p43题")
        block(s, MARGIN + 3.05, y + 0.06, 8.75, 0.78,
              [dict(text=desc, size=17, color=BODY, ls=1.12)],
              anchor=MSO_ANCHOR.MIDDLE, where="p43述")
        y += 0.99


def main():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    dk = Deck(prs, LOGO, total=TOTAL)
    s01(prs)
    s02(prs, dk)
    s03(prs)
    s04(prs, dk)
    s05(prs, dk)
    s06(prs, dk)
    s06b(prs, dk)
    s07(prs, dk)
    s08(prs, dk)
    s09(prs, dk)
    s10(prs, dk)
    s11(prs, dk)
    s12(prs, dk)
    s13(prs, dk)
    s14(prs, dk)
    s15(prs, dk)
    s16(prs)
    s17(prs, dk)
    s18(prs, dk)
    s19(prs, dk)
    s20(prs, dk)
    s21(prs, dk)
    s22(prs, dk)
    s23(prs, dk)
    s24(prs, dk)
    s25(prs, dk)
    s26(prs, dk)
    s27(prs, dk)
    s28(prs, dk)
    s29(prs, dk)
    s30(prs, dk)
    s31(prs, dk)
    s32(prs, dk)
    s33(prs, dk)
    s34(prs, dk)
    s35(prs, dk)
    s36(prs, dk)
    s37(prs, dk)
    s38(prs, dk)
    s39(prs, dk)
    s40(prs, dk)
    s41(prs)
    s42(prs, dk)
    s43(prs, dk)

    prs.core_properties.title = "TADA 文献阅读与代码复现"
    prs.core_properties.author = "张心雨"
    prs.core_properties.subject = "研究生组会汇报"
    prs.save(OUT)
    print("saved:", OUT, "slides:", len(prs.slides))
    for w in WARNINGS:
        print(w)


if __name__ == "__main__":
    main()
