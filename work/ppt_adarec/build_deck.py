# -*- coding: utf-8 -*-
"""构建 ADARec 组会汇报 PPT（蓝白极简 · 47 页 · 完整章节）。"""
import os
import sys
import math

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_lib import (  # noqa: E402
    Deck, block, card, rect, round_box, arrow, add_table, add_math, add_rich,
    para, textbox, FONT_CN,
    INK, BODY, GRAY, MUTED, BLUE, BLUE_MID, BLUE_LT, TINT, TINT2, LINE, WHITE,
    W, H, MARGIN, CW, WARNINGS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOGO = os.path.join(HERE, "cqu_logo_hd.png")
FIG = os.path.join(ROOT, "knowledge_base", "paper_reading", "assets", "ADARec")
IMG = os.path.join(HERE, "imgs")
OUT = os.path.join(ROOT, "information", "PPT",
                   "张心雨_ADARec文献阅读及代码复现_组会汇报_蓝白简约版.pptx")

TOTAL = 47
PART1 = "PART 1 · 文献阅读"
PART2 = "PART 2 · 代码复现"
PART3 = "PART 3 · 学习总结"
TOP = 1.88


def caption(slide, y, text, num, x=MARGIN, w=None, size=16):
    block(slide, x, y, w or CW, 0.30, [dict(text=text, size=size, color=MUTED, ls=1.0)],
          where=f"图注{num}")


def notes(slide, y, text, num, size=17, h=0.62):
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
          [dict(text="ADARec：自适应扩散增强", size=40, bold=True, color=INK, ls=1.05),
           dict(text="与分层解析混合专家", size=40, bold=True, color=INK, ls=1.05)],
          where="封面标题")
    block(s, 0.95, 4.24, 7.6, 0.80,
          [dict(text="De-collapsing User Intent: Adaptive Diffusion Augmentation "
                     "with Mixture-of-Experts for Sequential Recommendation", size=16,
                color=GRAY, ls=1.25)], where="封面英文")
    rect(s, 0.95, 5.12, 5.4, 0.012, LINE)
    block(s, 0.95, 5.30, 6.2, 0.46,
          [dict(text="**汇报人：张心雨**", size=24, color=BLUE, ls=1.0)],
          where="封面汇报人")
    block(s, 0.95, 5.86, 6.6, 0.40,
          [dict(text="文献阅读 · 代码复现 · 结果对照（AAAI 2026）", size=17,
                color=GRAY, ls=1.0)], where="封面范围")
    for (lab, val), y in zip([("论文", "AAAI 2026"), ("代码", "Cxx-0/ADARec"),
                              ("汇报范围", "阅读 + 复现")], [2.20, 3.42, 4.64]):
        block(s, 9.62, y, 3.2, 0.30, [dict(text=lab, size=15, color=BLUE_MID, ls=1.0)],
              where="封面右侧标签")
        block(s, 9.62, y + 0.30, 3.3, 0.42,
              [dict(text=val, size=21, bold=True, color=INK, ls=1.0)], where="封面右侧值")


# ============================ 2 目录 ============================
def s02(prs, dk):
    s = dk.page("汇报提纲",
                "三部分：论文方法与结果、代码复现与差异分析、本学期学习总结。",
                PART1, 2)
    items = [
        ("01", "文献阅读", "第 3–15 页", [
            "1.1 背景 · 目的 · 研究工作",
            "1.2 四个模块的公式与符号",
            "1.3 论文 RQ1–RQ5 结果",
        ]),
        ("02", "代码复现", "第 16–44 页", [
            "2.1 文件职责　2.2 模块实现",
            "2.3 参数 · 数据 · 模型 · 训练",
            "2.4 流程图与组件耦合",
            "2.5 复现结果与差异",
            "2.6 日志与结果图表",
        ]),
        ("03", "学习总结", "第 45–47 页", [
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
            ["1.1　研究背景 / 研究目的 / 研究工作",
             "1.2　ADC、HDA、HP-MoE 与总损失的公式及符号",
             "1.3　论文 RQ1–RQ5 的实验结果"],
            "对应原文 Section 3–4 与附录 Table 1 / 2 / 4 / 5")


def s16(prs):
    section(prs, 16, "02", "代码复现",
            ["2.1　文件职责　　2.2　四个模块的具体实现",
             "2.3　参数设置 / 数据处理 / 模型设定 / 模型训练",
             "2.4　训练流程与三组件耦合",
             "2.5　复现结果、差异归因与对齐版实验",
             "2.6　实验日志与结果图表"],
            "代码：Cxx-0/ADARec（官方原版）+ 本地对齐版 ADARec-2")


def s45(prs):
    section(prs, 45, "03", "学习总结",
            ["3.1　本学期的工作与产出",
             "3.2　后续学习计划"],
            "论文复现、知识梳理、文献阅读与前沿跟踪")


# ============================ 4 背景 ============================
def s04(prs, dk):
    s = dk.page("1.1 研究背景：任务与现有方法的不足",
                "任务：依据用户历史行为序列预测下一交互物品；本文聚焦数据稀疏场景。",
                PART1, 4)
    paras = [
        dict(text="**三个挑战**", size=20, color=INK),
        dict(text="① 交互样本不足，模型易过拟合训练集。", size=19, bullet=True, sb=8),
        dict(text="② 意图层次结构在稀疏数据中难以表达。", size=19, bullet=True, sb=6),
        dict(text="③ 用户活跃度呈长尾分布，加深稀疏程度。", size=19, bullet=True, sb=6),
        dict(text="**现有方法的不足**", size=20, color=INK, sb=14),
        dict(text="仅生成相似或多样的物品，未重建意图层次。", size=19, bullet=True, sb=8),
        dict(text="缺少高层意图引导，增强样本偏离真实意图。", size=19, bullet=True, sb=6),
    ]
    block(s, MARGIN, TOP, 6.55, 4.80, paras, where="p4正文")
    s.shapes.add_picture(os.path.join(FIG, "fig1.png"), Inches(7.45), Inches(1.92),
                         width=Inches(4.95))
    caption(s, 6.72, "图 1：数据稀疏对层次意图建模的阻碍（Fig.1）", 4, x=7.45, w=5.8)


# ============================ 5 目的 ============================
def s05(prs, dk):
    s = dk.page("1.1 研究目的与核心论点",
                "核心思路：增强强度随序列自适应，加噪步数对应抽象层级。",
                PART1, 5)
    paras = [
        dict(text="**要解决的问题**", size=20, color=INK),
        dict(text="在稀疏场景下由单条序列重建意图层次，缓解过拟合。", size=18, bullet=True, sb=8),
        dict(text="**核心论点**", size=20, color=INK, sb=14),
        dict(text="关键是构建意图层次结构，而非权衡相似性与多样性。", size=18, bullet=True, sb=8),
        dict(text="**方法学假设**", size=20, color=INK, sb=14),
        dict(text="去噪过程迫使模型在更高的抽象层次上推断意图。", size=18, bullet=True, sb=8),
        dict(text="**研究目标**", size=20, color=INK, sb=14),
        dict(text="标准基准与稀疏序列上均优于现有 SOTA。", size=18, bullet=True, sb=8),
    ]
    block(s, MARGIN, TOP, 6.45, 4.85, paras, where="p5正文")

    s.shapes.add_picture(os.path.join(FIG, "fig2.png"), Inches(7.35), Inches(1.92),
                         width=Inches(4.85))
    card(s, 7.35, 4.55, 5.35, 1.02, fill=TINT, line=LINE, accent=BLUE)
    block(s, 7.60, 4.65, 4.95, 0.84, [
        dict(text="高噪声步 → 粗粒度意图；低噪声步 → 细粒度意图。", size=17, color=INK),
    ], where="p5提示")
    caption(s, 5.68, "图 2：不同数据条件下的用户表示分布（Fig.2）", 5, x=7.35, w=5.4)
    block(s, 7.35, 6.02, 5.35, 0.72, [
        dict(text="稀疏条件下语义簇明显弱化，增强后分布更接近稠密情形。", size=16, color=GRAY),
    ], where="p5说明")


# ============================ 6 研究工作 ============================
def s06(prs, dk):
    s = dk.page("1.1 研究工作：骨干与三个新增模块",
                "骨干沿用 ELCRec（SASRec 变体）与 DDPM 线性噪声调度，改进集中在三个模块。",
                PART1, 6)
    cards = [
        ("ADC · 深度控制", "为每条序列确定加噪深度 T_{u}。", BLUE),
        ("HDA · 层次构建", "沿完整去噪轨迹构造意图集合 A_{u}。", BLUE_MID),
        ("HP-MoE · 专家融合", "粗/细粒度双专家经门控融合为 h_{u}。", BLUE_LT),
    ]
    x = MARGIN
    for title, desc, color in cards:
        card(s, x, 1.88, 3.86, 1.26, fill=TINT, line=LINE)
        rect(s, x + 0.02, 1.88, 0.07, 1.26, color)
        block(s, x + 0.26, 1.96, 3.45, 0.34,
              [dict(text=title, size=20, bold=True, color=INK, ls=1.0)], where="p6卡题")
        block(s, x + 0.26, 2.32, 3.40, 0.74,
              [dict(text=desc, size=18, sub=True, ls=1.16)], where="p6卡文")
        x += 4.11

    s.shapes.add_picture(os.path.join(FIG, "fig3.png"), Inches(1.46), Inches(3.34),
                         width=Inches(10.40))
    caption(s, 6.80, "图 3：ADA 框架全貌（Fig.3）", 6, x=1.46, w=10.40)


# ============================ 公式页 ============================
def formula_page(prs, dk, num, title, kicker, formulas, syms, note, fs=22):
    s = dk.page(title, kicker, PART1, num)
    fh = 0.47 * len(formulas) + 0.38
    card(s, MARGIN, 1.86, CW, fh, fill=WHITE, line=LINE, accent=BLUE)
    paras = [dict(text=f, size=fs, color=INK, math=True, ls=1.0,
                  sb=0 if i == 0 else 10) for i, f in enumerate(formulas)]
    block(s, MARGIN + 0.34, 1.96, CW - 0.6, fh - 0.18, paras, where=f"公式p{num}")

    rows = math.ceil(len(syms) / 2)
    sy = 1.86 + fh + 0.18
    sh_card = 6.44 - sy
    pitch = min(0.52, (sh_card - 0.50) / rows)
    card(s, MARGIN, sy, CW, sh_card, fill=TINT, line=LINE)
    block(s, MARGIN + 0.26, sy + 0.09, 3.0, 0.30,
          [dict(text="符号说明", size=18, bold=True, color=BLUE, ls=1.0)], where=f"符号题p{num}")
    colw = (CW - 0.90) / 2
    pt_gap = max(4.0, pitch * 72 - 18 * 1.22)
    for ci, chunk in enumerate([syms[:rows], syms[rows:]]):
        sh = textbox(s, MARGIN + 0.26 + ci * (colw + 0.42), sy + 0.46, colw,
                     sh_card - 0.52)
        tf = sh.text_frame
        for i, (sym, mean) in enumerate(chunk):
            p = para(tf, first=(i == 0), space_before=0 if i == 0 else pt_gap,
                     line_spacing=1.0)
            add_math(p, sym, 18, BLUE, bold=True)
            add_math(p, "　" + mean, 18, BODY, font=FONT_CN)

    block(s, MARGIN, 6.52, CW, 0.40,
          [dict(text=note, size=17, color=GRAY, sub=True, ls=1.0)], where=f"注释p{num}")


def s07(prs, dk):
    formula_page(
        prs, dk, 7,
        "1.2 公式（1/4）｜ADC 自适应深度控制器",
        "加噪越深，保留细节越少、越接近粗粒度意图；深度由控制器学习得到。",
        ["(1)　h_{c} = Pool( GRU( [ E_{u}^{0} ; V_{l} ] ) )",
         "(2)　logits_{u} = W_{σ} · h_{c} + b_{σ}",
         "(3)　p_{u} = Softmax( ( logits_{u} + g ) / τ_{gs} ) ，　T_{u} = argmax_{k} p_{u,k}"],
        [("E_{u}^{0}", "序列嵌入，L×D 的矩阵"),
         ("V_{l}", "序列长度的嵌入向量"),
         ("h_{c}", "GRU 汇总出的上下文向量"),
         ("W_{σ}, b_{σ}", "线性层的权重和偏置"),
         ("g", "Gumbel(0,1) 噪声，仅训练时加"),
         ("τ_{gs}", "温度系数，论文取 0.5"),
         ("p_{u}", "选到各深度的概率，K 维"),
         ("T_{u}", "最终选中的整数深度")],
        "前向以 argmax 取整数深度，反向以软概率 p_{u} 回传梯度；L_{expl} 抑制深度坍缩至最浅层。")


def s08(prs, dk):
    formula_page(
        prs, dk, 8,
        "1.2 公式（2/4）｜HDA 分层扩散增强",
        "单次加噪可学习的信息有限，因此保留完整去噪轨迹，逐层构造意图层级。",
        ["(4)　E_{u}^{k} = √ᾱ_{k} · E_{u}^{0} + √(1 − ᾱ_{k}) · ε_{k}",
         "(5)　Ê_{u}^{k} = ( E_{u}^{k} − √(1 − ᾱ_{k}) · ε_{θ}(E_{u}^{k}, k) ) / √ᾱ_{k}",
         "(6)　A_{u} = { Ê_{u}^{0}, Ê_{u}^{1}, …, Ê_{u}^{T} } ，　L_{deno} = ‖ ε_{θ}(E_{u}^{k}, k) − ε_{k} ‖²"],
        [("k", "当前加噪步数，0 ≤ k ≤ T_{u}"),
         ("ᾱ_{k}", "累积噪声系数 Π(1 − β_{i})"),
         ("β_{i}", "第 i 步的噪声方差，线性调度"),
         ("ε_{k}", "标准高斯噪声 N(0, I)"),
         ("ε_{θ}", "去噪网络，实现里是 MLP"),
         ("Ê_{u}^{k}", "第 k 层重建出的意图表示"),
         ("A_{u}", "意图层次集合，共 T_{u}+1 层"),
         ("T", "层数上限，即 ADC 给的 T_{u}")],
        "k 越大噪声越强，保留粗粒度意图；k = 0 保留全部细节。集合 A_{u} 交由 HP-MoE 逐层解析。")


def s09(prs, dk):
    formula_page(
        prs, dk, 9,
        "1.2 公式（3/4）｜HP-MoE 分层解析混合专家",
        "细粒度专家建模具体偏好，粗粒度专家建模稳定本质，由两级门控确定权重。",
        ["(7)　g_{k} = σ( MLP( [ Pool(Ê_{u}^{k}) ; emb(k) ] ) )",
         "(8)　z_{fin} = Σ_{k} g_{k} · E_{fin}(Ê_{u}^{k}) / ( Σ_{k} g_{k} + ε )",
         "(9)　z_{coar} = Σ_{k} (1 − g_{k}) · E_{coar}(Ê_{u}^{k}) / ( Σ_{k}(1 − g_{k}) + ε )",
         "(10)　gate_{u} = σ( W_{g} z_{coar} + b_{g} ) ，　h_{u} = gate_{u} ⊙ z_{fin} + (1 − gate_{u}) ⊙ z_{coar}"],
        [("g_{k}", "第 k 层归给细粒度专家的权重"),
         ("E_{fin} / E_{coar}", "细粒度 / 粗粒度专家网络"),
         ("z_{fin} / z_{coar}", "两个专家各自聚合出的表征"),
         ("gate_{u}", "第二级门控，粗、细之间的取舍"),
         ("h_{u}", "最终用户表征，送进预测层"),
         ("⊙", "逐元素相乘"),
         ("σ", "Sigmoid 函数"),
         ("ε", "防止分母为零的小常数")],
        "第一级 g_{k} 决定各层权重，第二级 gate_{u} 决定粗细表征的融合比例；L_{cont} 约束粗粒度层彼此靠近。")


def s10(prs, dk):
    s = dk.page("1.2 公式（4/4）｜总损失函数",
                "仅 L_{rec} 直接监督下一物品预测，其余三项为辅助正则。",
                PART1, 10)
    card(s, MARGIN, 1.86, CW, 1.18, fill=WHITE, line=LINE, accent=BLUE)
    block(s, MARGIN + 0.34, 1.96, CW - 0.6, 1.0, [
        dict(text="(11)　L_{tot} = L_{rec} + λ_{deno} L_{deno} + λ_{cont} L_{cont} + λ_{expl} L_{expl}",
             size=23, math=True, ls=1.0),
        dict(text="(12)　L_{expl} = Σ_{k} p̄_{k} log p̄_{k}　（最大化熵，抑制深度分布坍缩）",
             size=22, math=True, ls=1.0, sb=12),
    ], where="p10公式")

    add_table(s, MARGIN, 3.24, CW, [1.9, 6.2, 2.5], [
        ["损失项", "作用", "论文权重"],
        ["L_{rec}", "下一物品的交叉熵，主监督信号", "1（固定）"],
        ["L_{deno}", "去噪网络的噪声预测 MSE", "λ_{deno} = 0.1"],
        ["L_{cont}", "粗粒度层之间的 InfoNCE", "λ_{cont} = 0.1"],
        ["L_{expl}", "深度分布的熵正则，防止只选浅层", "λ_{expl} = 0.01"],
    ], row_h=0.50, header_h=0.52, size=19, math=True, where="p10表")

    notes(s, 6.00, "λ 为三项辅助损失的权重；权重过大会削弱主监督信号，复现实验验证了这一点（见第 31 页）。",
          10, size=18, h=0.34)


# ============================ 11 RQ1 ============================
def s11(prs, dk):
    s = dk.page("1.3 结果 RQ1：主结果（论文 Table 1）",
                "表 1 节选三个代表性基线（SASRec / ELCRec / DiffDiv），原文共 11 个基线。",
                PART1, 11)
    rows = [["数据集", "指标", "SASRec", "ELCRec", "DiffDiv", "ADARec", "提升"],
            ["Beauty", "H@5", "0.0374", "0.0529", "0.0575", "**0.0600**", "[[+4.35%]]"],
            ["Beauty", "N@5", "0.0241", "0.0355", "0.0375", "**0.0403**", "[[+4.68%]]"],
            ["Sports", "H@5", "0.0206", "0.0286", "0.0315", "**0.0341**", "[[+3.96%]]"],
            ["Sports", "N@5", "0.0135", "0.0185", "0.0211", "**0.0228**", "[[+3.64%]]"],
            ["Toys", "H@5", "0.0463", "0.0585", "0.0645", "**0.0691**", "[[+6.64%]]"],
            ["Toys", "N@5", "0.0306", "0.0403", "0.0428", "**0.0474**", "[[+9.08%]]"],
            ["Yelp", "H@5", "0.0160", "0.0248", "0.0258", "**0.0264**", "[[+2.33%]]"],
            ["Yelp", "N@5", "0.0101", "0.0153", "0.0155", "**0.0167**", "[[+3.73%]]"]]
    add_table(s, MARGIN, TOP, CW, [1.35, 1.0, 1.6, 1.6, 1.6, 1.7, 1.5], rows,
              row_h=0.45, header_h=0.45, size=19, where="p11表")
    notes(s, 6.00, "提升幅度相对次优基线计算，对应原文 Improv. 列；原文报告区间为 2.3%~9.1%。",
          11, size=18, h=0.34)
    caption(s, 6.42, "数据来源：论文 Table 1（AAAI 2026）", 11)


# ============================ 12 RQ2 ============================
def s12(prs, dk):
    s = dk.page("1.3 结果 RQ2：消融实验（论文 Table 4）",
                "在 Beauty 上逐一移除组件，NDCG@20 的降幅反映其重要性。",
                PART1, 12)
    rows = [["变体", "HR@20", "NDCG@20", "相对完整"],
            ["完整 ADARec", "**0.1187**", "**0.0568**", "—"],
            ["去掉 ADC", "0.1141", "0.0545", "[[−4.1%]]"],
            ["去掉 HDA", "0.1105", "0.0528", "[[−7.0%]]"],
            ["去掉 HP-MoE", "0.1075", "0.0505", "[[−11.1%]]"],
            ["只留 ADC + HDA", "0.1128", "0.0539", "[[−5.1%]]"],
            ["去掉 C-A Router", "0.1172", "0.0561", "[[−1.2%]]"],
            ["去掉 L_{cont}", "0.1158", "0.0553", "[[−2.6%]]"]]
    add_table(s, MARGIN, TOP, 7.4, [2.4, 1.5, 1.8, 1.8], rows, row_h=0.48,
              header_h=0.50, size=19, where="p12表")
    notes(s, 5.92, "四个数据集结论一致，此处仅列 Beauty。", 12, size=18, h=0.38)
    caption(s, 6.30, "数据来源：论文 Table 4（AAAI 2026）", 12)

    card(s, 8.32, TOP, 4.39, 4.40, fill=TINT, line=LINE, accent=BLUE)
    block(s, 8.58, 1.98, 3.90, 4.15, [
        dict(text="原文结论", size=19, bold=True, color=BLUE, ls=1.0),
        dict(text="HP-MoE 降幅最大，是层次解析的主要来源。", size=16, sb=10),
        dict(text="HDA 次之，其作用为提供结构化输入。", size=16, sb=8),
        dict(text="仅保留 HP-MoE 时约等于 ELCRec：缺少 HDA 的层次输入，解析器难以发挥作用。", size=16, sb=8),
        dict(text="C-A Router 与 L_{cont} 影响最小，但不属可选项。", size=16, sb=8),
    ], where="p12侧栏")


# ============================ 13 RQ3 ============================
def s13(prs, dk):
    s = dk.page("1.3 结果 RQ3：计算效率（论文 Table 5）",
                "单位：训练时间为秒、推理时间为毫秒；均在相同硬件条件下测量。",
                PART1, 13)
    rows = [["模型", "Beauty 训练", "Beauty 推理", "Sports 训练", "Sports 推理"],
            ["SASRec", "2.15", "3.1", "6.51", "9.5"],
            ["ELCRec", "3.81", "5.9", "11.28", "16.1"],
            ["DiffDiv", "7.55", "14.8", "20.10", "31.5"],
            ["GlobalDiff", "8.90", "21.2", "23.50", "45.8"],
            ["**ADARec**", "**6.21**", "**10.5**", "**17.95**", "**24.9**"]]
    add_table(s, MARGIN, TOP, CW, [1.9, 2.2, 2.2, 2.2, 2.2], rows, row_h=0.56,
              header_h=0.58, size=19, where="p13表")
    block(s, MARGIN, 5.24, CW, 1.45, [
        dict(text="相对 ELCRec：训练耗时约为 1.6~1.8 倍，源于双专家两次前向；推理约为 1.5~1.8 倍。", size=19,
             bullet=True),
        dict(text="相对扩散基线：训练快 20%~40%、推理快 30%~50%，整体架构较轻。", size=19,
             bullet=True, sb=8),
        dict(text="推理只使用未加噪的 raw 分支，扩散与自适应仅在训练期生效。", size=19,
             bullet=True, sb=8),
    ], where="p13文")
    caption(s, 6.60, "数据来源：论文 Table 5（AAAI 2026）", 13)


# ============================ 14 RQ4 ============================
def s14(prs, dk):
    s = dk.page("1.3 结果 RQ4：超参设置与敏感性（论文 Fig.4）",
                "左表为原文 Implementation Details，右图为两个辅助损失权重的敏感性曲线。",
                PART1, 14)
    add_table(s, MARGIN, TOP, 6.30, [3.0, 3.2], [
        ["超参", "论文取值"],
        ["优化器 / weight decay", "Adam / 1×10⁻⁴"],
        ["最大序列长度", "50"],
        ["Gumbel 温度 τ_{gs}", "0.5"],
        ["InfoNCE 温度 τ", "0.07"],
        ["最大扩散深度 T_{max}", "{4, 6, 8, 10, 12}"],
        ["λ_{deno} / λ_{cont}", "{0.01 … 0.15}"],
        ["λ_{expl}", "{0.001 … 0.1}"],
        ["硬件", "20 GB RTX 3090"],
    ], row_h=0.44, header_h=0.46, size=17, sub=True, where="p14表")

    s.shapes.add_picture(os.path.join(FIG, "fig4.png"), Inches(7.12), Inches(1.90),
                         width=Inches(5.50))
    block(s, 7.12, 5.05, 5.50, 1.30, [
        dict(text="最优取值：λ_{expl} = 0.01、λ_{deno} = 0.1。", size=18, sub=True),
        dict(text="最优点附近曲线平缓，小幅偏差影响有限。", size=18, sub=True, sb=6),
    ], where="p14文")
    caption(s, 6.44, "数据来源：论文 Table 5 与 Fig.4（AAAI 2026）", 14)


# ============================ 15 RQ5 ============================
def s15(prs, dk):
    s = dk.page("1.3 结果 RQ5：稀疏序列（论文 Table 2）",
                "稀疏场景是本方法的主要收益来源：序列越短，层次重建的收益越明显。",
                PART1, 15)
    cd = CategoryChartData()
    cd.categories = ["Yelp", "Toys", "Sports", "Beauty"]
    cd.add_series("NDCG@5 提升", (2.5, 10.0, 10.6, 19.7))
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.62), Inches(1.88),
                            Inches(6.35), Inches(4.35), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = "稀疏场景（≤5 条）NDCG@5 提升"
    tp = ch.chart_title.text_frame.paragraphs[0]
    tp.font.size = Pt(18)
    tp.font.name = FONT_CN
    tp.font.color.rgb = GRAY
    ch.font.size = Pt(18)
    ch.font.name = FONT_CN
    plot = ch.plots[0]
    plot.gap_width = 55
    ser = plot.series[0]
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = BLUE
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.font.size = Pt(18)
    dl.font.bold = True
    dl.font.color.rgb = BLUE
    dl.number_format = '0.0"%"'
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    va = ch.value_axis
    va.visible = False
    va.has_major_gridlines = False
    ca = ch.category_axis
    ca.format.line.color.rgb = LINE
    ca.tick_labels.font.size = Pt(20)
    ca.tick_labels.font.color.rgb = BODY
    ca.has_major_gridlines = False

    block(s, 7.25, 1.90, 5.45, 4.5, [
        dict(text="**三个数据集提升 10.0%~19.7%**", size=20, color=INK),
        dict(text="Beauty、Sports、Toys 在序列截短后提升最明显，增益来自层次信息补全。", size=18, sb=6),
        dict(text="**Yelp 提升仅 2.5%**", size=20, color=INK, sb=14),
        dict(text="原文解释为 Yelp 本身缺乏明显的层次结构。", size=18, sb=6),
        dict(text="**待核实**", size=20, color=BLUE, sb=14),
        dict(text="原文未设 ELCRec 稀疏对照，无法区分增益来自扩散增强还是双专家集成。", size=18, sb=6),
    ], where="p15文")
    caption(s, 6.42, "数据来源：论文 Table 2（AAAI 2026）；提升为相对最强基线的 NDCG@5 涨幅", 15)


# ============================ 17 文件职责 ============================
def s17(prs, dk):
    s = dk.page("2.1 文件职责：六个源码文件的分工",
                "官方仓库 Cxx-0/ADARec 的源码结构，按 main.py 的调用顺序阅读。",
                PART2, 17)
    rows = [["文件", "职责", "核心类 / 函数"],
            ["main.py", "解析参数、组装模型、调度双专家、训练至早停", "main()"],
            ["datasets.py", "切分历史与标签，生成对比学习副本",
             "RecWithContrastiveLearningDataset"],
            ["models.py", "SASRec 主干、自适应扩散分支、KMeans", "SASRecModel"],
            ["modules.py", "注意力、编码层、深度控制器、对比损失", "AdaptiveDepthController"],
            ["diffusion_utils.py", "噪声调度与闭式前向加噪", "forward_diffusion"],
            ["trainers.py", "损失组合、训练循环、验证与早停", "ELCRecTrainer / DualExpertTrainer"]]
    add_table(s, MARGIN, TOP, CW, [2.1, 5.6, 4.4], rows, row_h=0.62,
              header_h=0.54, size=17, where="p17表")
    notes(s, 6.10, "双专家并非独立模型：同一套代码运行两份，深度档位分别为 [4,5] 与 [9,10]，预测取平均。",
          17, size=17, h=0.62)


# ============================ 18 ADC 实现 ============================
def s18(prs, dk):
    s = dk.page("2.2 模块实现（1/4）：ADC 深度控制器",
                "实现位置：models.py L162 调用，modules.py L276–312 定义。",
                PART2, 18)
    block(s, MARGIN, TOP, CW, 3.95, [
        dict(text="**调用入口**", size=20, color=INK),
        dict(text="probs = self.controller(sequence_emb.detach(), sequence_lengths) → (B, 2)；detach 阻断梯度。",
             size=19, bullet=True, sb=8),
        dict(text="**内部三步**", size=20, color=INK, sb=12),
        dict(text="① GRU 编码序列，取 last_hidden.squeeze(0) → (B, D)。", size=19, bullet=True, sb=8),
        dict(text="② 长度嵌入 clamp(lengths, max=512) → (B, D)。", size=19, bullet=True, sb=6),
        dict(text="③ 拼接 (B, 2D) → Linear → logits (B, 2) → softmax → probs。", size=19, bullet=True, sb=6),
        dict(text="**probs 的三个去向**", size=20, color=INK, sb=12),
        dict(text="前向加权求和；反向计算 −entropy(probs)；InstanceCL 分支丢弃。", size=19, bullet=True, sb=8),
    ], where="p18文")
    card(s, MARGIN, 6.02, CW, 0.78, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.30, 6.10, CW - 0.6, 0.64, [
        dict(text="与论文的差异：使用普通 softmax，无 Gumbel 噪声、无 argmax，属相邻深度的软混合。",
             size=18, color=INK),
    ], where="p18差异")


# ============================ 19 HDA 实现 ============================
def s19(prs, dk):
    s = dk.page("2.2 模块实现（2/4）：HDA 分层扩散增强",
                "实现位置：diffusion_utils.py forward_diffusion，models.py L181–194。",
                PART2, 19)
    block(s, MARGIN, TOP, CW, 4.0, [
        dict(text="**前向加噪**（forward_diffusion，仅两个纯函数）", size=20, color=INK),
        dict(text="sqrt_alpha = sqrt(alphas_cumprod[t]).view(-1, 1, 1)。", size=19, bullet=True, sb=8),
        dict(text="E_t = sqrt_alpha · E_0 + sqrt(1 − alpha) · noise，noise 独立采样。", size=19,
             bullet=True, sb=6),
        dict(text="t 为 per-sample（torch.full((B,), depth − 1)），同批共享 depth。", size=19,
             bullet=True, sb=6),
        dict(text="**adaptive 分支的生成路径**", size=20, color=INK, sb=14),
        dict(text="input_ids → 位置编码 → sequence_emb → controller(detach) → probs。", size=19,
             bullet=True, sb=8),
        dict(text="对每个 depth 执行 forward_diffusion，按权重加权求和。", size=19, bullet=True, sb=6),
        dict(text="→ item_encoder（Transformer）→ 取末层 → adaptive_sequence_output。", size=19,
             bullet=True, sb=6),
    ], where="p19文")
    card(s, MARGIN, 5.88, 5.95, 0.90, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.26, 5.98, 5.55, 0.74, [
        dict(text="差异：无独立去噪网络，加噪后直接再编码。", size=17, color=INK),
    ], where="p19差异")
    card(s, 6.68, 5.88, 6.04, 0.90, fill=TINT, line=LINE)
    block(s, 6.94, 5.98, 5.64, 0.74, [
        dict(text="两路关系：raw 与 adaptive 并行，共享 sequence_emb（加噪前 detach）。", size=17, color=INK),
    ], where="p19并行")


# ============================ 20 HP-MoE 实现 ============================
def s20(prs, dk):
    s = dk.page("2.2 模块实现（3/4）：HP-MoE 双专家",
                "实现位置：main.py L178–184 构造，trainers.py 负责训练调度。",
                PART2, 20)
    block(s, MARGIN, TOP, CW, 4.0, [
        dict(text="**双专家构造**（--dual_expert 启用）", size=20, color=INK),
        dict(text="levels_e1 = [4, 5]（浅层）→ model_e5；levels_e2 = [9, 10]（深层）→ model_e10。", size=19,
             bullet=True, sb=8),
        dict(text="每个专家是完整的 SASRecModel，含独立 embedding、Encoder 与 controller。", size=19,
             bullet=True, sb=6),
        dict(text="**训练**", size=20, color=INK, sb=14),
        dict(text="每个 batch 分别训练 model_e5 与 model_e10，各自独立优化器、独立反向传播。", size=19,
             bullet=True, sb=8),
        dict(text="**推理融合**", size=20, color=INK, sb=14),
        dict(text="rating_pred = (rating_pred_e5 + rating_pred_e10) / 2.0，等权平均。", size=19,
             bullet=True, sb=8),
    ], where="p20文")
    card(s, MARGIN, 5.86, 5.95, 0.92, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.26, 5.96, 5.55, 0.76, [
        dict(text="差异：无门控网络，等权平均，更接近双模型集成。", size=17, color=INK),
    ], where="p20差异")
    card(s, 6.68, 5.86, 6.04, 0.92, fill=WHITE, line=LINE)
    block(s, 6.94, 5.96, 5.64, 0.76, [
        dict(text="风险：双专家无参数共享，显存翻倍，且无跨专家知识迁移。", size=17, color=INK),
    ], where="p20风险")


# ============================ 21 损失实现 ============================
def s21(prs, dk):
    s = dk.page("2.2 模块实现（4/4）：损失函数组合",
                "实现位置：trainers.py _train_one_expert。",
                PART2, 21)
    card(s, MARGIN, 1.86, CW, 0.74, fill=WHITE, line=LINE, accent=BLUE)
    block(s, MARGIN + 0.34, 1.97, CW - 0.6, 0.54, [
        dict(text="joint_loss = rec × 1.0 + deno × 1e−4 + expl × 1e−4 + cl", size=21, math=True, ls=1.0),
    ], where="p21公式")

    add_table(s, MARGIN, 2.78, CW, [1.5, 6.6, 3.4], [
        ["损失项", "代码实现", "说明"],
        ["L_{rec}", "raw 与 adaptive 分支各自计算交叉熵", "主监督信号"],
        ["L_{deno}", "MSE(adaptive_output, raw.detach())", "表示重构，非噪声预测"],
        ["L_{expl}", "−mean(entropy(probs))", "负熵探索"],
        ["L_{cl}", "InstanceCL(NCE) + IntentCL(PCL) + 簇中心距离", "对比学习"],
    ], row_h=0.56, header_h=0.52, size=18, sub=True, where="p21表")

    block(s, MARGIN, 5.62, CW, 1.15, [
        dict(text="双专家各自独立反向传播，共享数据但参数不共享。", size=18, bullet=True),
        dict(text="与论文的差异：L_{deno} 为表示重构损失，而非噪声预测 MSE，是后续超参失效的原因之一。",
             size=18, bullet=True, sb=8, sub=True),
    ], where="p21注")


# ============================ 22 参数设置 1 ============================
def s22(prs, dk):
    s = dk.page("2.3 参数设置（1/2）：符号约定与默认值",
                "符号含义与命令行默认值；数据集差异见表格末行。",
                PART2, 22)
    add_table(s, MARGIN, TOP, CW, [1.6, 6.4, 3.6], [
        ["符号", "含义", "默认值"],
        ["B", "batch_size", "256"],
        ["T", "max_seq_length", "50"],
        ["D", "hidden_size", "128"],
        ["H", "num_attention_heads", "2"],
        ["L", "head_size = D / H", "64"],
        ["K", "num_intent_clusters", "512（Sports/Toys）/ 256（Beauty/Yelp）"],
        ["M", "controller_num_levels", "2（硬编码）"],
        ["T_{max}", "diffusion_t_max", "50"],
    ], row_h=0.44, header_h=0.46, size=18, sub=True, where="p22表")
    notes(s, 5.98, "其他设置：patience = 40、delta = 0；Adam betas = (0.9, 0.98)、eps = 1e−8；β 由 1e−4 线性增至 0.02。",
          22, size=17, h=0.62)


# ============================ 23 参数设置 2 ============================
def s23(prs, dk):
    s = dk.page("2.3 参数设置（2/2）：论文设定与本地命令",
                "训练时显式传参以对齐论文；唯一无法对齐的是 Gumbel-Softmax。",
                PART2, 23)
    add_table(s, MARGIN, TOP, CW, [2.6, 2.9, 2.9, 3.6], [
        ["参数", "论文设定", "本地命令", "说明"],
        ["batch_size", "256", "256", "对齐"],
        ["hidden_size", "256 / 128", "256 / 128", "对齐"],
        ["num_hidden_layers", "1 / 2", "1 / 2", "对齐"],
        ["weight_decay", "1×10⁻⁴", "0", "**无法对齐，会崩**"],
        ["λ_{deno} / λ_{expl}", "0.1 / 0.01", "1e−4 / 1e−4", "**无法对齐**"],
        ["Gumbel 温度", "0.5", "普通 softmax", "**代码不支持**"],
        ["diffusion_t_max", "50", "50", "对齐"],
    ], row_h=0.46, header_h=0.48, size=17, sub=True, where="p23表")
    notes(s, 5.88, "本地硬件：RTX 5060 Laptop 8 GB / 32 GB 内存 / Core Ultra 9 275HX；论文为 20 GB RTX 3090。"
                   "另注意：双专家实际加噪深度仍硬编码为 [4,5] / [9,10]，diffusion_t_max 只控制噪声调度的步数。",
          23, size=17, h=0.90)


# ============================ 24 数据处理 ============================
def s24(prs, dk):
    s = dk.page("2.3 数据处理：切分、样本与评测协议",
                "实现位置：datasets.py RecWithContrastiveLearningDataset。",
                PART2, 24)
    block(s, MARGIN, TOP, CW, 4.74, [
        dict(text="**数据划分**（item_list 切分）", size=20, color=INK),
        dict(text="valid_rating_matrix 取 item_list[:-2]，仅含训练物品。", size=19, bullet=True, sb=8),
        dict(text="test_rating_matrix 取 item_list[:-1]，含训练与验证物品。", size=19, bullet=True, sb=6),
        dict(text="**序列切分与标签**", size=20, color=INK, sb=14),
        dict(text="交互序列 → input_ids（历史）+ target_pos（正样本）+ target_neg（负样本）。", size=19,
             bullet=True, sb=8),
        dict(text="同时生成两份增强副本，供对比学习使用。", size=19, bullet=True, sb=6),
        dict(text="**Padding**：左填充至 max_seq_length，以 mask 标记有效位置。", size=19, bullet=True, sb=12),
        dict(text="**负采样**：候选集为 100 个随机负样本 + 1 个正样本；负物品不从训练历史中采样。", size=19,
             bullet=True, sb=12),
        dict(text="**评测协议**：--do_eval 必须切到 test_rating_matrix，否则验证物品泄漏（第 38 页）。", size=19,
             bullet=True, sb=12),
    ], where="p24文")


# ============================ 25 模型设定 ============================
def s25(prs, dk):
    s = dk.page("2.3 模型设定：主干、扩散分支与双专家",
                "实现位置：models.py SASRecModel 与 KMeans 聚类器。",
                PART2, 25)
    block(s, MARGIN, TOP, CW, 4.74, [
        dict(text="**SASRec 主干**（继承 ELCRec）", size=20, color=INK),
        dict(text="物品嵌入 + 位置嵌入 → 多层 Transformer Encoder → 序列表示；含 Pre-LN、残差与因果 mask。",
             size=18, bullet=True, sb=8),
        dict(text="**自适应扩散分支**（enable_adaptive_diffusion）", size=20, color=INK, sb=14),
        dict(text="raw：干净嵌入 → Encoder → raw_output。", size=18, bullet=True, sb=8),
        dict(text="adaptive：加噪嵌入 → Encoder → adaptive_output。", size=18, bullet=True, sb=6),
        dict(text="两路共享 sequence_emb，controller 输入处 detach。", size=18, bullet=True, sb=6),
        dict(text="**双专家模式**（--dual_expert）", size=20, color=INK, sb=12),
        dict(text="model_e5（[4,5]）与 model_e10（[9,10]）各自为完整模型，融合 fused = (pred_e5 + pred_e10) / 2。",
             size=18, bullet=True, sb=8),
        dict(text="**KMeans 聚类器**：维护可学习簇中心（意图原型），按 EMA 更新。", size=18, bullet=True, sb=10),
    ], where="p25文")


# ============================ 26 模型训练 ============================
def s26(prs, dk):
    s = dk.page("2.3 模型训练：循环、早停与续训",
                "实现位置：trainers.py DualExpertTrainer 与 ELCRecTrainer。",
                PART2, 26)
    block(s, MARGIN, TOP, CW, 4.6, [
        dict(text="**单专家训练循环**（_train_one_expert）", size=20, color=INK),
        dict(text="forward → raw_loss + adaptive_loss → joint_loss → backward → step。", size=19,
             bullet=True, sb=8),
        dict(text="**双专家调度**（DualExpertTrainer）", size=20, color=INK, sb=14),
        dict(text="每个 batch 分别训练 model_e5 与 model_e10，独立优化器、独立反向。", size=19,
             bullet=True, sb=8),
        dict(text="**验证与早停**", size=20, color=INK, sb=14),
        dict(text="每 epoch 评测 e5 / e10 / fused 三组指标（各 HIT@5/20、NDCG@5/20）。", size=19,
             bullet=True, sb=8),
        dict(text="双独立 EarlyStopper，监控各专家 raw NDCG@20，patience = 40、delta = 0；两者均触发后结束。",
             size=19, bullet=True, sb=6),
        dict(text="**断点续训**（--resume）：快照含模型、优化器与早停器状态，weights_only = False。", size=19,
             bullet=True, sb=14),
    ], where="p26文")


# ============================ 27 流程 ============================
def s27(prs, dk):
    s = dk.page("2.4 训练执行流程：从入口到测试",
                "入口为 main.py，流程为：解析参数 → 读取数据 → 构建模型 → 训练 → 早停 → 测试。",
                PART2, 27)
    steps = [("main.py", "解析命令行参数"), ("datasets.py", "切分序列"),
             ("models.py", "构建骨干与专家"), ("trainers.py", "逐 epoch 训练"),
             ("最终测试", "加载权重并评测")]
    bw, gap = 2.06, 0.42
    x = MARGIN
    for i, (name, desc) in enumerate(steps):
        solid = i in (0, 4)
        round_box(s, x, 1.92, bw, 1.10, fill=BLUE if solid else WHITE, line=BLUE_LT)
        block(s, x + 0.10, 2.04, bw - 0.20, 0.34,
              [dict(text=name, size=19, bold=True, color=WHITE if solid else INK, ls=1.0)],
              align=PP_ALIGN.CENTER, where="p27流程名")
        block(s, x + 0.16, 2.44, bw - 0.32, 0.46,
              [dict(text=desc, size=17, color=WHITE if solid else GRAY, ls=1.05)],
              align=PP_ALIGN.CENTER, where="p27流程文")
        if i < len(steps) - 1:
            arrow(s, x + bw + 0.07, 2.34, 0.28, 0.28)
        x += bw + gap

    card(s, MARGIN, 3.32, CW, 1.55, fill=TINT, line=LINE, accent=BLUE)
    block(s, MARGIN + 0.30, 3.44, CW - 0.6, 1.3, [
        dict(text="**双专家训练方式**", size=19, color=INK, ls=1.0),
        dict(text="每个 batch 依次训练 e5 与 e10：优化器、反向传播与早停均独立，两者均早停后结束。", size=18, sb=6),
        dict(text="hidden 128 对应 Sports / Toys，256 对应 Beauty；缺失项一律使用代码默认值。", size=18, sb=4),
    ], where="p27说明")

    items = [("batch_size", "256"), ("max_seq_length", "50"), ("epochs", "400"),
             ("patience", "40"), ("hidden_size", "128 / 256"),
             ("显卡", "RTX 5060 8 GB")]
    bw2, gap2 = 3.86, 0.30
    for i, (k, v) in enumerate(items):
        r_, c_ = divmod(i, 3)
        x = MARGIN + c_ * (bw2 + gap2)
        y = 5.06 + r_ * 0.86
        card(s, x, y, bw2, 0.72, fill=WHITE, line=LINE, radius=0.1)
        block(s, x + 0.22, y + 0.08, bw2 - 0.4, 0.26,
              [dict(text=k, size=15, color=GRAY, ls=1.0)], where="p27参数名")
        block(s, x + 0.22, y + 0.35, bw2 - 0.4, 0.30,
              [dict(text=v, size=19, bold=True, color=INK, ls=1.0)], where="p27参数值")

    block(s, MARGIN, 6.70, CW, 0.34, [
        dict(text="训练期验证用 valid_rating_matrix（监控 raw NDCG@20），最终测试用 test_rating_matrix。",
             size=15, color=MUTED),
    ], where="p27注")


# ============================ 28 耦合 ============================
def s28(prs, dk):
    s = dk.page("2.4 三组件耦合：共用一个 probs",
                "论文中 ADC、HDA、HP-MoE 为独立模块，代码中三者依赖同一概率向量。",
                PART2, 28)
    boxes = [(MARGIN, 1.86, 2.45, 0.90, "sequence_emb", "序列嵌入 (B,T,D)"),
             (3.42, 1.86, 2.55, 0.90, "controller", "GRU + Linear"),
             (6.35, 1.86, 2.30, 0.90, "probs (B,2)", "softmax 概率"),
             (9.18, 1.86, 3.45, 0.90, "HDA 加噪加权", "Σ 各深度 × 概率")]
    for x, y, w, h, t1, t2 in boxes:
        round_box(s, x, y, w, h, fill=WHITE, line=BLUE_LT)
        block(s, x + 0.08, y + 0.10, w - 0.16, 0.30,
              [dict(text=t1, size=17, bold=True, color=INK, ls=1.0)],
              align=PP_ALIGN.CENTER, where="p28框1")
        block(s, x + 0.08, y + 0.45, w - 0.16, 0.28,
              [dict(text=t2, size=14, color=GRAY, ls=1.0)],
              align=PP_ALIGN.CENTER, where="p28框2")
    for x in (3.03, 6.04, 8.72):
        arrow(s, x, 2.18, 0.32, 0.28)
    arrow(s, 7.40, 2.80, 0.26, 0.24, direction="down")

    card(s, MARGIN, 3.10, 5.30, 0.98, fill=TINT2, line=None)
    block(s, MARGIN + 0.24, 3.18, 4.9, 0.82, [
        dict(text="移除任一组件，其余两个随之失效：probs 为 None 时探索损失归零，HDA 亦失去分层输入。",
             size=16, color=INK),
    ], where="p28拆")
    card(s, 6.05, 3.10, 6.66, 0.98, fill=TINT2, line=None)
    block(s, 6.29, 3.18, 6.2, 0.82, [
        dict(text="probs 同时承担两种职责：前向作为层级权重，反向经负熵作为探索信号。", size=17, color=INK),
    ], where="p28probs")

    rows = [["组件的论文职责", "代码实现"],
            ["ADC 选深度", "仅保留相邻两档（num_levels=2），自适应空间被压缩"],
            ["HDA 建层次", "加噪后直接编码，无去噪网络，层次仅剩两档"],
            ["HP-MoE 门控", "固定 0.5/0.5 等权平均，无学习得到的门控"]]
    add_table(s, MARGIN, 4.24, CW, [3.3, 8.7], rows, row_h=0.48, header_h=0.48,
              size=18, where="p28表")
    block(s, MARGIN, 6.18, CW, 0.42, [
        dict(text="结论：决定指标的是 SASRec 骨干与双专家集成，扩散相关组件对最终结果贡献有限。", size=17,
             color=GRAY),
    ], where="p28结")


# ============================ 29 复现 RQ1 ============================
def s29(prs, dk):
    s = dk.page("2.5 复现结果（1/6）：RQ1 主结果对照",
                "代码默认参数、双专家 fused 指标；Beauty 与 Sports 的差值均在 0.002 以内。",
                PART2, 29)
    rows = [["数据集", "指标", "论文 ADARec", "本地复现", "差值"],
            ["Beauty", "H@5", "0.0600", "0.0587", "[[−0.0013]]"],
            ["Beauty", "N@5", "0.0403", "0.0392", "[[−0.0011]]"],
            ["Beauty", "H@20", "0.1187", "0.1169", "[[−0.0018]]"],
            ["Beauty", "N@20", "0.0568", "0.0557", "[[−0.0011]]"],
            ["Sports", "H@5", "0.0341", "0.0340", "[[−0.0001]]"],
            ["Sports", "N@5", "0.0228", "0.0224", "[[−0.0004]]"],
            ["Sports", "H@20", "0.0745", "0.0751", "[[+0.0006]]"],
            ["Sports", "N@20", "0.0341", "0.0340", "[[−0.0001]]"]]
    add_table(s, MARGIN, TOP, 7.6, [1.3, 1.0, 1.9, 1.7, 1.6], rows,
              row_h=0.45, header_h=0.45, size=19, where="p29表")

    card(s, 8.52, TOP, 4.19, 4.05, fill=TINT, line=LINE, accent=BLUE)
    block(s, 8.78, 1.98, 3.70, 3.8, [
        dict(text="差异来源", size=19, bold=True, color=BLUE, ls=1.0),
        dict(text="主干与训练流程未改动，仅修正评测协议。", size=16, sb=10),
        dict(text="早停分别在第 324 / 239 个 epoch。", size=16, sb=8),
        dict(text="单次运行，未做多随机种子平均，差值含波动。", size=16, sb=8),
        dict(text="八项指标七项略低、一项略高，无系统性偏差。", size=16, sb=8),
    ], where="p29侧栏")
    caption(s, 6.10, "数据来源：本地训练日志（2026 年 9 月，RTX 5060 8GB），对照论文 Table 1", 29)


# ============================ 30 复现 RQ2 ============================
def s30(prs, dk):
    s = dk.page("2.5 复现结果（2/6）：RQ2 去 ADC 消融",
                "模型 4：--ablate_adc，固定深度 [5] / [10]，最终测试 fused 指标。",
                PART2, 30)
    add_table(s, MARGIN, TOP, CW, [3.0, 1.5, 1.5, 1.5, 1.5, 1.6], [
        ["变体", "H@5", "N@5", "H@20", "N@20", "早停 epoch"],
        ["完整模型（Beauty-1）", "0.0587", "0.0392", "0.1169", "0.0557", "324"],
        ["去 ADC（Beauty-4）", "**0.0594**", "**0.0401**", "**0.1179**", "**0.0566**", "233"],
        ["去 HDA（Beauty-5）", "0.0576", "0.0390", "0.1170", "0.0558", "164"],
        ["伪消融（Beauty-3）", "0.0588", "0.0398", "0.1177", "0.0565", "262"],
    ], row_h=0.50, header_h=0.50, size=18, where="p30表")

    block(s, MARGIN, 4.44, CW, 2.3, [
        dict(text="**未复现论文的下降**：去掉 ADC 后指标反而微升 0.001 量级，与论文 −4.1% 的结论相反。", size=19,
             color=INK),
        dict(text="① 本地 ADC 为相邻双深度软混合（num_levels=2，[4,5] / [9,10]），自适应空间极小，远弱于论文 0~k 的连续 Gumbel-Softmax。",
             size=18, bullet=True, sb=10),
        dict(text="② 代码把 ADC / HDA / C-A Router 挂在同一个 probs 上，拆任一个都会连带废掉其余。", size=18,
             bullet=True, sb=6),
        dict(text="③ probs 的唯一消费方是 HDA 加噪，去掉 HDA 后 probs 为 None，探索损失归零，ADC 不再被训练。",
             size=18, bullet=True, sb=6),
    ], where="p30文")
    caption(s, 6.62, "数据来源：本地训练日志（2026 年 9 月），对照论文 Table 4", 30)


# ============================ 31 复现 RQ3 ============================
def s31(prs, dk):
    s = dk.page("2.5 复现结果（3/6）：RQ3 超参敏感性",
                "在 Toys 上对比代码默认参数与论文参数，NDCG@5 的差异随 τ 与 weight_decay 放大。",
                PART2, 31)
    add_table(s, MARGIN, TOP, CW, [3.4, 1.1, 1.8, 2.0, 1.6, 1.5], [
        ["配置", "τ", "weight_decay", "deno / expl", "NDCG@5", "相对默认"],
        ["默认（Toys-4）", "1.0", "0", "1e−4 / 1e−4", "**0.0474**", "—"],
        ["论文 τ=0.07（Toys-2）", "0.07", "0", "0.1 / 0.01", "0.0049", "[[−89.7%]]"],
        ["论文 τ=0.5（Toys-3）", "0.5", "0", "0.1 / 0.01", "0.0182", "[[−61.6%]]"],
        ["τ=1.0，大辅助权重（Toys-5）", "1.0", "0", "0.1 / 0.01", "0.0463", "[[−2.3%]]"],
        ["τ=1.0，加权重衰减（Toys-6）", "1.0", "1e−4", "1e−4 / 1e−4", "0.0280", "[[−41%]]"],
    ], row_h=0.52, header_h=0.50, size=17, where="p31表")
    block(s, MARGIN, 5.24, CW, 1.30, [
        dict(text="**默认参数性能最优**：deno = 1e−4、expl = 1e−4、τ = 1.0 在 Toys 上取得最佳结果。", size=18,
             bullet=True),
        dict(text="**论文参数套用后骤降**：τ 由 1.0 降至 0.5 即下降 61.6%，降至 0.07 进一步恶化至 −89.7%。", size=18,
             bullet=True, sb=6),
        dict(text="**辅助权重不是主因**：放大 1000 倍仅降 2.3%；weight_decay 由 0 改为 1e−4 单独即降 41%。", size=18,
             bullet=True, sb=6),
    ], where="p31注")
    caption(s, 6.58, "数据来源：本地训练日志（2026 年 9–10 月），对照论文 Implementation Details", 31)


# ============================ 32 复现 RQ5 ============================
def s32(prs, dk):
    s = dk.page("2.5 复现结果（4/6）：RQ5 稀疏序列",
                "把历史截断到 5 条（--shorten_seq_to 5），最终测试 fused 指标。",
                PART2, 32)
    add_table(s, MARGIN, TOP, 7.4, [1.6, 1.2, 1.6, 1.6, 1.6], [
        ["数据集", "指标", "论文", "本地", "差值"],
        ["Sports", "H@5", "0.0305", "0.0306", "+0.0001"],
        ["Sports", "N@5", "0.0199", "0.0204", "+0.0005"],
        ["Sports", "H@20", "0.0659", "0.0681", "+0.0022"],
        ["Sports", "N@20", "0.0298", "0.0309", "+0.0011"],
        ["Yelp", "H@5", "0.0200", "0.0195", "−0.0005"],
        ["Yelp", "N@5", "0.0124", "0.0123", "−0.0001"],
        ["Yelp", "H@20", "0.0559", "0.0552", "−0.0007"],
        ["Yelp", "N@20", "0.0225", "0.0222", "−0.0003"],
    ], row_h=0.42, header_h=0.44, size=18, where="p32表")
    card(s, 8.32, TOP, 4.39, 1.95, fill=TINT, line=LINE, accent=BLUE)
    block(s, 8.58, 2.06, 3.90, 1.70, [
        dict(text="结论", size=18, bold=True, color=BLUE, ls=1.0),
        dict(text="两个数据集的四项指标差值均在 0.0022 以内，稀疏复现成立。", size=16, sb=8),
    ], where="p32侧栏")
    card(s, 8.32, 3.95, 4.39, 2.05, fill=WHITE, line=LINE)
    block(s, 8.58, 4.06, 3.90, 1.80, [
        dict(text="补充说明", size=18, bold=True, color=GRAY, ls=1.0),
        dict(text="未设置 ELCRec 稀疏对照，无法区分增益来自扩散增强还是双专家集成。", size=16, sb=8),
    ], where="p32侧栏2")
    caption(s, 6.10, "数据来源：本地训练日志（2026 年 9 月），对照论文 Table 2", 32)


# ============================ 33 汇总 1 ============================
def s33(prs, dk):
    s = dk.page("2.5 复现结果（5/6）：Beauty 全部训练记录",
                "官方原版代码，双专家模式，epochs = 400，按最终测试 fused 指标汇总。",
                PART2, 33)
    add_table(s, MARGIN, TOP, CW, [2.4, 4.6, 1.3, 1.4, 1.4, 1.4], [
        ["实验", "关键设置", "早停", "H@5", "N@5", "N@20"],
        ["Beauty-1", "完整模型（代码默认）", "324", "0.0587", "0.0392", "0.0557"],
        ["Beauty-3", "伪消融（probs 固定 0.5）", "262", "0.0588", "0.0398", "0.0565"],
        ["Beauty-4", "初版去 ADC", "233", "0.0594", "0.0401", "0.0566"],
        ["Beauty-5", "初版去 HDA", "164", "0.0576", "0.0390", "0.0558"],
        ["Beauty-6", "论文参数（τ = 0.07）", "102", "0.0235", "0.0151", "0.0221"],
        ["Beauty-7", "严格去 ADC", "231", "0.0580", "0.0393", "0.0559"],
        ["Beauty-8", "严格去 HDA", "290", "0.0568", "0.0382", "0.0548"],
        ["Beauty-9", "严格去 MoE（单专家）", "227", "0.0575", "0.0388", "0.0548"],
        ["Beauty-10", "三消融全开", "258", "0.0598", "0.0403", "0.0571"],
    ], row_h=0.42, header_h=0.46, size=17, where="p33表")
    caption(s, 6.38, "数据来源：本地训练日志（2026 年 9–10 月）；对应论文 Table 1 / Table 4", 33)


# ============================ 34 汇总 2 ============================
def s34(prs, dk):
    s = dk.page("2.5 复现结果（6/6）：Toys / Sports / Yelp 训练记录",
                "同一批训练中的其余数据集，含稀疏设置与论文参数组。",
                PART2, 34)
    add_table(s, MARGIN, TOP, CW, [2.3, 5.2, 1.2, 1.4, 1.4, 1.4], [
        ["实验", "关键设置", "早停", "H@5", "N@5", "N@20"],
        ["Toys-4", "完整（代码默认）", "358", "0.0687", "0.0474", "0.0649"],
        ["Toys-1", "论文参数 + T_{max} = 12", "179", "0.0070", "0.0052", "0.0084"],
        ["Toys-3", "论文参数（τ = 0.5）", "280", "0.0281", "0.0182", "0.0281"],
        ["Toys-5", "τ=1.0 / 大辅助权重", "286", "0.0673", "0.0463", "0.0616"],
        ["Toys-6", "τ=1.0 / wd=1e−4", "325", "0.0426", "0.0280", "0.0410"],
        ["Sports-1", "完整（代码默认）", "239", "0.0338", "0.0224", "0.0339"],
        ["Sports-2", "稀疏（max_seq_length=5）", "163", "0.0308", "0.0201", "0.0306"],
        ["Sports-3", "稀疏（shorten_seq_to 5）", "179", "0.0306", "0.0204", "0.0309"],
        ["Yelp-2", "稀疏（max_seq_length=5）", "117", "0.0223", "0.0137", "0.0246"],
        ["Yelp-3", "稀疏（shorten_seq_to 5）", "103", "0.0195", "0.0123", "0.0222"],
    ], row_h=0.38, header_h=0.42, size=16, where="p34表")
    caption(s, 6.24, "数据来源：本地训练日志（2026 年 9–10 月）；Toys 论文基线 N@5 = 0.0474，Sports = 0.0228", 34)


# ============================ 35 归因 ============================
def s35(prs, dk):
    s = dk.page("2.5 参数失效的归因：三个发现",
                "默认参数可复现，换成论文参数即失效；消融也做不出论文的下降幅度。",
                PART2, 35)
    cards = [
        ("发现一：默认参数可复现",
         "Beauty 0.0392、Sports 0.0224、Toys 0.0474，与论文 0.0403 / 0.0228 / 0.0474 基本一致。",
         "主干（SASRec + 双专家集成）与评测协议正确。"),
        ("发现二：论文参数直接失效",
         "Beauty 0.0392 → 0.0151；Toys 0.0474 → 0.0052。",
         "主因为 weight_decay = 1e−4 与过小的 InfoNCE 温度 τ = 0.07。"),
        ("发现三：消融做不出论文降幅",
         "严格去 ADC 后 NDCG@20 为 0.0559，与完整模型 0.0557 几乎相同；去 HDA / MoE 仅降 1.6%。",
         "代码中三组件共用 probs，自适应空间被压缩（见第 28 页）。"),
    ]
    y = 1.88
    for title, body, tag in cards:
        card(s, MARGIN, y, CW, 1.46, fill=TINT, line=LINE, accent=BLUE)
        block(s, MARGIN + 0.32, y + 0.12, CW - 0.7, 1.24, [
            dict(text="**" + title + "**", size=19, color=INK, ls=1.0),
            dict(text=body, size=17, sb=6),
            dict(text=tag, size=15, color=GRAY, sb=4),
        ], where="p35卡")
        y += 1.56
    caption(s, 6.62, "对应论文 Table 1 / Table 4，以及本地 Beauty-6、Toys-1、Beauty-7/8/9 五组实验", 35)


# ============================ 36 对齐版实验 ============================
def s36(prs, dk):
    s = dk.page("2.5 补充：ADARec-2 对齐版的实验",
                "按论文补全 5 处语义后重跑，检验超参与消融结论是否随之改变。",
                PART2, 36)
    add_table(s, MARGIN, TOP, 6.00, [2.2, 3.9], [
        ["补全项", "ADARec-2 的做法"],
        ["ADC", "gumbel_softmax，τ=0.5，hard"],
        ["HDA", "DenoisingNetwork + Eq.5 重建"],
        ["HP-MoE", "逐层 g_k，按 Eq.8/9 聚合"],
        ["L_{deno}", "噪声预测 MSE"],
        ["深度采样", "[min, max] 内采样单一 T_u"],
    ], row_h=0.50, header_h=0.48, size=16, sub=True, where="p36左表")

    add_table(s, 6.92, TOP, 5.79, [2.7, 1.2, 1.2], [
        ["实验", "H@5", "N@5"],
        ["Sports-3　默认参数", "0.0331", "0.0222"],
        ["Sports-1　论文超参", "0.0070", "0.0046"],
        ["Sports-2　放宽 τ / T_{max}", "0.0150", "0.0116"],
        ["Beauty-3　完整对齐版", "0.0598", "0.0403"],
        ["Beauty-1　去 ADC", "0.0580", "0.0400"],
        ["Beauty-2　去 HDA", "0.0575", "0.0396"],
    ], row_h=0.46, header_h=0.48, size=16, where="p36右表")
    block(s, MARGIN, 5.30, CW, 0.95, [
        dict(text="Sports-3 与 Beauty-3 在默认参数下与论文一致（论文 Beauty N@5 = 0.0403）。", size=17,
             bullet=True),
        dict(text="论文超参在 Sports-1 上仍失效（N@5 = 0.0046），说明补全语义并未消除该问题。", size=17,
             bullet=True, sb=6),
    ], where="p36注")
    caption(s, 6.36, "数据来源：本地训练日志（2026 年 10 月）；左表为 5 处语义补全，右表为六组实验结果", 36)


# ============================ 37 对齐版结论 ============================
def s37(prs, dk):
    s = dk.page("2.5 补充：ADARec-2 的结论变化",
                "语义对齐后，部分旧结论被推翻，部分结论保持不变。",
                PART2, 37)
    block(s, MARGIN, TOP, CW, 4.15, [
        dict(text="**① 语义对齐 + 默认参数即可复现**", size=19, color=INK),
        dict(text="Beauty-3 的 N@5 = 0.0403，与论文完全一致；补全的 Gumbel、去噪网络与门控路由在默认超参下未带来可观测增益。",
             size=17, bullet=True, sb=8),
        dict(text="**② 论文超参仍失效，但主因是 weight_decay 与 τ**", size=19, color=INK, sb=14),
        dict(text="旧归因「L_deno 为表示重构」被推翻：改为噪声预测 MSE 后，崩溃依旧。", size=17, bullet=True, sb=8),
        dict(text="Toys-5（τ=1.0、wd=0、大辅助权重）正常，N@5 = 0.0463；Toys-6（τ=1.0、wd=1e−4）即降 41%。",
             size=17, bullet=True, sb=6),
        dict(text="**③ HDA 消融仍做不出论文降幅**", size=19, color=INK, sb=14),
        dict(text="Beauty-2 去 HDA 后 N@20 = 0.0567，相对完整仅降 0.70%，远小于论文 −7.0%。", size=17,
             bullet=True, sb=8),
    ], where="p37文")
    card(s, MARGIN, 6.12, CW, 0.76, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.30, 6.20, CW - 0.6, 0.60, [
        dict(text="结论：可复现的增益主要来自 SASRec 骨干与双专家集成，扩散与门控的边际贡献接近于 0。", size=17,
             color=INK),
    ], where="p37结")


# ============================ 38 工程问题 ============================
def s38(prs, dk):
    s = dk.page("2.5 工程与协议问题：4 项修复",
                "前三项为工程问题，第四项为评测协议问题，是指标偏低的主要原因。",
                PART2, 38)
    add_table(s, MARGIN, TOP, CW, [3.2, 5.5, 3.4], [
        ["现象", "原因", "处理"],
        ["训练约 4.5 小时后内存溢出", "每批调用 train_matrix.toarray()，单批约 17.9 MiB，占用主机内存而非显存",
         "按批读取稀疏行，避免整块转稠密"],
        ["启动即报 NameError", "变量名拼写前后不一致", "统一变量名"],
        ["权重保存后无法加载", "两个专家 checkpoint 后缀重复", "为 e5 / e10 分别加后缀"],
        ["--do_eval 指标异常偏低（0.0266）", "评测仍使用 valid_rating_matrix，验证集物品泄漏",
         "改用 test_rating_matrix 后恢复至 0.0587"],
    ], row_h=0.90, header_h=0.54, size=17, where="p38表")
    notes(s, 6.18, "第 4 项影响最大：未修复时整体指标系统性偏低，易误判为方法无效。", 38, size=18, h=0.38)


# ============================ 39 日志 ============================
def s39(prs, dk):
    s = dk.page("2.6 实验日志：记录方式与训练分布",
                "每轮训练按数据集分别记录，指标逐 epoch 追加。",
                PART2, 39)
    block(s, MARGIN, TOP, 5.95, 4.55, [
        dict(text="**记录内容**", size=20, color=INK),
        dict(text="每次运行单独记录运行输出、错误输出与指标追加，首行记录完整命令。", size=18,
             bullet=True, sb=8),
        dict(text="每个 epoch 记录 e5 / e10 / fused 三组四项指标，并记录早停与最终测试。", size=18,
             bullet=True, sb=6),
        dict(text="**早停监控**", size=20, color=INK, sb=14),
        dict(text="各专家 raw NDCG@20，双独立 EarlyStopper，patience = 40。", size=18, bullet=True, sb=8),
        dict(text="两专家均触发后才结束训练。", size=18, bullet=True, sb=6),
    ], where="p39左")

    add_table(s, 6.78, TOP, 5.92, [1.9, 1.0, 3.0], [
        ["数据集", "组数", "主要设置"],
        ["Beauty", "9", "默认 / 消融 / 论文参数"],
        ["Sports", "3", "默认 / 两种稀疏"],
        ["Toys", "6", "默认 / 论文参数 / 折中"],
        ["Yelp", "2", "两种稀疏"],
        ["ADARec-2", "6", "Sports 3 + Beauty 3"],
    ], row_h=0.56, header_h=0.50, size=16, where="p39表")
    caption(s, 5.92, "合计 26 组训练（官方原版 20 组 + 对齐版 6 组）", 39, x=6.78, w=5.92)


# ============================ 40 图表 1 ============================
def s40(prs, dk):
    s = dk.page("2.6 结果图（1/4）：主结果复现",
                "完整模型（默认参数）与论文 Table 1 的四项指标对照，三个数据集走势一致。",
                PART2, 40)
    s.shapes.add_picture(os.path.join(IMG, "s39_ee1b20ba.png"), Inches(0.90), Inches(1.96),
                         width=Inches(11.55))
    caption(s, 5.10, "图 4：主结果复现对比（本地训练日志 vs 论文 Table 1，2026 年 9 月）", 40)
    block(s, MARGIN, 5.48, CW, 1.25, [
        dict(text="三个数据集的 HIT 与 NDCG 柱高基本重合，最大偏差为 Beauty 的 N@5（−0.0011）。", size=18,
             bullet=True),
        dict(text="Toys 的两项指标略高于论文，说明默认参数在该数据集上并非严格最优。", size=18,
             bullet=True, sb=6),
    ], where="p40文")


# ============================ 41 图表 2 ============================
def s41(prs, dk):
    s = dk.page("2.6 结果图（2/4）：稀疏序列复现",
                "序列长度截断至 5 后，本地结果与论文 Table 2 的对照。",
                PART2, 41)
    s.shapes.add_picture(os.path.join(IMG, "s40_901e7712.png"), Inches(2.40), Inches(1.98),
                         width=Inches(8.55))
    caption(s, 5.34, "图 5：稀疏序列复现对比（--shorten_seq_to 5，2026 年 9 月）", 41)
    block(s, MARGIN, 5.74, CW, 1.05, [
        dict(text="Sports：四项指标差值 0.0001~0.0022，N@20 略高于论文。", size=18, bullet=True),
        dict(text="Yelp：四项指标差值 0.0001~0.0007，整体略低于论文，但量级一致。", size=18, bullet=True, sb=6),
    ], where="p41文")


# ============================ 42 图表 3 ============================
def s42(prs, dk):
    s = dk.page("2.6 结果图（3/4）：消融与超参对照",
                "左图为 Beauty 消融，右图为论文超参与代码默认参数的对照。",
                PART2, 42)
    s.shapes.add_picture(os.path.join(IMG, "s41_45e2763c.png"), Inches(1.10), Inches(1.96),
                         width=Inches(5.20))
    s.shapes.add_picture(os.path.join(IMG, "s41_0291a45a.png"), Inches(6.95), Inches(1.96),
                         width=Inches(5.20))
    caption(s, 4.66, "图 6：Beauty 消融（左）与论文 / 默认超参对照（右）", 42)
    block(s, MARGIN, 5.06, CW, 1.55, [
        dict(text="消融：三种严格消融的降幅均在 1.6% 以内，去 ADC 甚至微升，与论文 −4.1%~−11.1% 不一致。", size=18,
             bullet=True),
        dict(text="超参：论文参数组的指标明显低于默认参数组，τ 与 weight_decay 是主要分界。", size=18,
             bullet=True, sb=6),
    ], where="p42文")


# ============================ 43 图表 4 ============================
def s43(prs, dk):
    s = dk.page("2.6 结果图（4/4）：训练结果汇总",
                "各实验最终 fused H@20，红线为论文基线；默认参数组贴近基线，论文参数组明显偏离。",
                PART2, 43)
    s.shapes.add_picture(os.path.join(IMG, "s42_15cbcc11.png"), Inches(2.45), Inches(1.92),
                         width=Inches(8.45))
    caption(s, 6.74, "图 7：16 组实验的 fused H@20 汇总（2026 年 9–10 月）", 43)


# ============================ 44 复现结论 ============================
def s44(prs, dk):
    s = dk.page("小结：复现结论",
                "汇总 26 组训练，区分论文贡献与工程实现带来的收益。",
                PART2, 44)
    card(s, MARGIN, 1.88, 5.90, 3.55, fill=TINT, line=LINE)
    rect(s, MARGIN + 0.02, 1.88, 5.86, 0.06, BLUE)
    block(s, MARGIN + 0.30, 2.08, 5.30, 3.2, [
        dict(text="已复现", size=22, bold=True, color=BLUE, ls=1.0),
        dict(text="主结果：Beauty / Sports 与论文差值在 0.002 以内。", size=17, bullet=True, sb=14),
        dict(text="稀疏结果：Sports、Yelp 基本一致。", size=17, bullet=True, sb=8),
        dict(text="对齐版 ADARec-2 同样可复现，Beauty N@5 = 0.0403 与论文相同。", size=17,
             bullet=True, sb=8),
    ], where="p44左")
    card(s, 6.82, 1.88, 5.89, 3.55, fill=WHITE, line=LINE)
    rect(s, 6.84, 1.88, 5.85, 0.06, BLUE_LT)
    block(s, 7.12, 2.08, 5.29, 3.2, [
        dict(text="未能复现", size=22, bold=True, color=GRAY, ls=1.0),
        dict(text="消融幅度：论文为 7%~11%，本地仅 1.6%，移除 ADC 后指标回升。", size=17, bullet=True, sb=14),
        dict(text="效率实验：未单独测量，仅核对原文数值。", size=17, bullet=True, sb=8),
        dict(text="论文超参：直接套用导致指标骤降，本地仅采用代码默认值。", size=17, bullet=True, sb=8),
    ], where="p44右")

    card(s, MARGIN, 5.62, CW, 0.86, fill=TINT2, line=None, accent=BLUE)
    block(s, MARGIN + 0.30, 5.72, CW - 0.6, 0.76, [
        dict(text="结论：可复现的增益主要来自 SASRec 骨干与双专家集成；扩散、自适应深度与门控在公开代码中贡献接近 0。",
             size=18, color=INK),
    ], where="p44总结")


# ============================ 46 学习总结 ============================
def s46(prs, dk):
    s = dk.page("3.1 本学期的工作与产出",
                "五项工作及其对应产出文件。", PART3, 46)
    add_table(s, MARGIN, TOP, CW, [2.2, 6.5, 3.3], [
        ["方面", "工作内容", "产出"],
        ["论文复现", "精读 ADARec，完成 26 组训练，修复 4 个问题", "复现记录 + 训练日志"],
        ["推荐系统", "梳理发展脉络、经典与现代方法、实验范式", "推荐系统笔记（56 条）"],
        ["Python", "从代码中提取语法点，补充序列、图、扩散写法", "语法笔记（标行号）"],
        ["文献阅读", "精读 1 篇、泛读 21 篇（13 篇扩散 SR + 8 篇生成式）", "文献阅读笔记（21 篇）"],
        ["前沿跟踪", "Agent、世界模型、LLM×Rec；补充 HSTU、OneRec", "前沿术语速查"],
    ], row_h=0.70, header_h=0.54, size=18, where="p46表")
    notes(s, 6.08, "全部笔记均标注论文出处与对应代码行号。", 46, size=18, h=0.38)


# ============================ 47 后续计划 ============================
def s47(prs, dk):
    s = dk.page("3.2 后续学习计划",
                "五项均为本学期实际受阻的环节，按优先级排序。", PART3, 47)
    items = [
        ("01", "英文阅读", "阅读英文论文仍依赖翻译工具。计划每周精读一篇原文，重点核对关键术语。"),
        ("02", "专业术语", "建立个人术语表，新术语先查证再记录。"),
        ("03", "文献阅读", "保持每周 2~3 篇，重点转向扩散增强与对比学习方向。"),
        ("04", "复现频率", "每篇论文至少完成一次代码跑通，记录论文描述与实现不一致之处。"),
        ("05", "方向选择", "以消融无法复现的问题为切入点，先完善实验设计，再考虑投稿。"),
    ]
    y = 1.90
    for code, title, desc in items:
        card(s, MARGIN, y, CW, 0.90, fill=TINT if code in ("01", "03", "05") else WHITE,
             line=LINE)
        block(s, MARGIN + 0.25, y + 0.20, 0.7, 0.5,
              [dict(text=code, size=21, bold=True, color=BLUE_LT, ls=1.0)], where="p47号")
        block(s, MARGIN + 0.95, y + 0.06, 2.1, 0.78,
              [dict(text=title, size=20, bold=True, color=INK, ls=1.0)],
              anchor=MSO_ANCHOR.MIDDLE, where="p47题")
        block(s, MARGIN + 3.05, y + 0.06, 8.75, 0.78,
              [dict(text=desc, size=18, color=BODY, ls=1.12)],
              anchor=MSO_ANCHOR.MIDDLE, where="p47述")
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
    s41(prs, dk)
    s42(prs, dk)
    s43(prs, dk)
    s44(prs, dk)
    s45(prs)
    s46(prs, dk)
    s47(prs, dk)

    prs.core_properties.title = "ADARec 文献阅读与代码复现"
    prs.core_properties.author = "张心雨"
    prs.core_properties.subject = "研究生组会汇报"
    prs.save(OUT)
    print("saved:", OUT, "slides:", len(prs.slides))
    for w in WARNINGS:
        print(w)


if __name__ == "__main__":
    main()
