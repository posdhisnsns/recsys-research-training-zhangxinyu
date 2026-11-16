# -*- coding: utf-8 -*-
"""绘图基础库：蓝白极简风格的元素封装（文本框、卡片、表格、公式、流程图）。"""
import re
import unicodedata

from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree

# ---------------- 配色 ----------------
INK      = RGBColor(0x10, 0x24, 0x3E)
BODY     = RGBColor(0x24, 0x34, 0x4D)
GRAY     = RGBColor(0x56, 0x68, 0x82)
MUTED    = RGBColor(0x8C, 0x9C, 0xB2)
BLUE     = RGBColor(0x1B, 0x4B, 0x97)
BLUE_MID = RGBColor(0x2F, 0x6A, 0xBE)
BLUE_LT  = RGBColor(0x6E, 0x9B, 0xD8)
TINT     = RGBColor(0xF5, 0xF9, 0xFE)
TINT2    = RGBColor(0xE9, 0xF1, 0xFB)
LINE     = RGBColor(0xD5, 0xE3, 0xF6)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
ROW_ALT  = RGBColor(0xF3, 0xF8, 0xFE)

FONT_CN = "微软雅黑"
FONT_MATH = "Cambria Math"

W, H = 13.333, 7.5
MARGIN = 0.62
CW = W - 2 * MARGIN

WARNINGS = []


# ---------------- 估算 ----------------
def _is_wide(ch):
    return unicodedata.east_asian_width(ch) in ("W", "F")


def est_lines(text, width_in, size_pt):
    usable = width_in * 72.0
    lines, cur = 1, 0.0
    for ch in text:
        if ch == "\n":
            lines += 1
            cur = 0.0
            continue
        if _is_wide(ch):
            w = size_pt * 1.0
        elif ch == " ":
            w = size_pt * 0.28
        elif ch.isdigit():
            w = size_pt * 0.56
        else:
            w = size_pt * 0.52
        if cur + w > usable:
            lines += 1
            cur = w
        else:
            cur += w
    return lines


def est_height(paras, width_in):
    total = 0.0
    for text, size, sb, ls in paras:
        total += est_lines(text, width_in, size) * size * ls * 1.22 / 72.0 + sb / 72.0
    return total


# ---------------- 文本 ----------------
TOKEN = re.compile(r"\*\*(.+?)\*\*|\[\[(.+?)\]\]")


def _strip(text):
    return TOKEN.sub(lambda m: m.group(1) or m.group(2), text)


def _plain_math(text):
    """去掉 _{ } ^{ } 标记，用于高度估算。"""
    out = []
    for seg, kind in parse_math(_strip(text)):
        out.append(seg)
    return "".join(out)


def _split_runs(text):
    out, pos = [], 0
    for m in TOKEN.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False, False))
        out.append((m.group(1), True, False) if m.group(1) is not None
                   else (m.group(2), False, True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False, False))
    return out


def _run(p, text, size, color, font, bold=False):
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    rPr = r._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", font)
    return r


def parse_math(text):
    """把 _{...} ^{...} 拆成 (片段, 类型)，支持嵌套大括号（内层拍平）。"""
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c in "_^" and i + 1 < len(text) and text[i + 1] == "{":
            depth, j = 0, i + 1
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner = text[i + 2:j]
            inner = inner.replace("_{", "").replace("^{", "").replace("}", "")
            out.append((inner, "sub" if c == "_" else "sup"))
            i = j + 1
        else:
            out.append((c, "n"))
            i += 1
    merged = []
    for seg, kind in out:
        if merged and merged[-1][1] == kind == "n":
            merged[-1] = (merged[-1][0] + seg, "n")
        else:
            merged.append((seg, kind))
    return merged


def _emit(p, text, size, color, bold, font):
    """输出一段文本：CJK 用中文字体，其他用 font。"""
    if font is None:
        buf, cur = "", None
        for ch in text:
            k = "cn" if _is_wide(ch) else "en"
            if cur is None or k == cur:
                buf += ch
                cur = k
            else:
                _run(p, buf, size, color, FONT_CN if cur == "cn" else FONT_MATH, bold)
                buf, cur = ch, k
        if buf:
            _run(p, buf, size, color, FONT_CN if cur == "cn" else FONT_MATH, bold)
    else:
        _run(p, text, size, color, font, bold)


def add_math(p, text, size, color=INK, bold=False, font=None):
    for seg, kind in parse_math(text):
        if kind == "n":
            _emit(p, seg, size, color, bold, font)
        else:
            r_probe = _emit
            # 先记录 run 数，再给新增 run 设置 baseline
            n0 = len(p.runs)
            _emit(p, seg, size, color, bold, font)
            for run in list(p.runs)[n0:]:
                run._r.get_or_add_rPr().set("baseline", "-25000" if kind == "sub" else "30000")


def add_rich(p, text, size, color=BODY, bold=False, font=None, sub=False):
    for seg, b, acc in _split_runs(text):
        if sub:
            add_math(p, seg, size, BLUE if acc else color, bold or b, font=font or FONT_CN)
        else:
            _emit(p, seg, size, BLUE if acc else color, bold or b, font or FONT_CN)


# ---------------- 图元 ----------------
def _flat(sp):
    """去掉主题样式引用，避免继承阴影/描边。"""
    st = sp._element.find(qn("p:style"))
    if st is not None:
        sp._element.remove(st)
    sp.shadow.inherit = False


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP, wrap=True):
    sh = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.word_wrap = wrap
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    tf.auto_size = None
    return sh


def para(tf, first=False, align=PP_ALIGN.LEFT, space_before=0, line_spacing=1.22):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(0)
    p.line_spacing = line_spacing
    return p


def block(slide, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, where=""):
    sh = textbox(slide, x, y, w, h, anchor=anchor)
    tf = sh.text_frame
    est = []
    for i, spec in enumerate(paras):
        size = spec.get("size", 20)
        p = para(tf, first=(i == 0), align=spec.get("align", align),
                 space_before=spec.get("sb", 0), line_spacing=spec.get("ls", 1.22))
        if spec.get("bullet"):
            _run(p, "▪  ", size, BLUE_LT, FONT_CN, True)
            text = spec["text"]
        else:
            text = spec["text"]
        if spec.get("math"):
            add_math(p, text, size, spec.get("color", INK), spec.get("bold", False))
        else:
            add_rich(p, text, size, spec.get("color", BODY), spec.get("bold", False),
                     sub=spec.get("sub", False))
        est.append((_plain_math(text) + ("▪  " if spec.get("bullet") else ""), size,
                    spec.get("sb", 0), spec.get("ls", 1.22)))
    need = est_height(est, w)
    if need > h + 0.05:
        WARNINGS.append(f"[溢出] {where}: {need:.2f}in / {h:.2f}in  «{_plain_math(paras[0]['text'])[:22]}»")
    return sh


def card(slide, x, y, w, h, fill=TINT, line=LINE, line_w=1.0, radius=0.05, accent=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.adjustments[0] = radius
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_w)
    _flat(sh)
    sh.text_frame.word_wrap = True
    if accent:
        bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                     Inches(0.075), Inches(h))
        bar.adjustments[0] = 0.4
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent
        bar.line.fill.background()
        _flat(bar)
    return sh


def rect(slide, x, y, w, h, fill, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
    _flat(sh)
    return sh


def round_box(slide, x, y, w, h, fill=WHITE, line=BLUE_LT, radius=0.1):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.adjustments[0] = radius
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(1.1)
    _flat(sh)
    return sh


def arrow(slide, x, y, w, h, direction="right", fill=BLUE_LT):
    shp = MSO_SHAPE.RIGHT_ARROW if direction == "right" else MSO_SHAPE.DOWN_ARROW
    sh = slide.shapes.add_shape(shp, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    _flat(sh)
    return sh


# ---------------- 表格 ----------------
def _cell_borders(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    for i, tag in enumerate(("a:lnL", "a:lnR", "a:lnT", "a:lnB")):
        el = etree.Element(qn(tag))
        el.set("w", "0")
        etree.SubElement(el, qn("a:noFill"))
        tcPr.insert(i, el)


def add_table(slide, x, y, w, col_ratios, rows, row_h=0.44, header_h=None, size=19,
              header_fill=BLUE, header_color=WHITE, aligns=None, math=False,
              sub=False, where=""):
    nrow, ncol = len(rows), len(rows[0])
    header_h = header_h or row_h
    heights = [header_h] + [row_h] * (nrow - 1)
    gf = slide.shapes.add_table(nrow, ncol, Inches(x), Inches(y), Inches(w),
                                Inches(sum(heights)))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")

    total = sum(col_ratios)
    for i, r in enumerate(col_ratios):
        tbl.columns[i].width = Inches(w * r / total)
    for r in range(nrow):
        tbl.rows[r].height = Inches(heights[r])

    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.09)
            cell.margin_right = Inches(0.09)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else (WHITE if r % 2 else ROW_ALT)
            _cell_borders(cell)
            tf = cell.text_frame
            tf.word_wrap = True
            color = header_color if r == 0 else BODY
            for li, line in enumerate(str(val).split("\n")):
                p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
                p.alignment = (aligns[c] if aligns
                               else (PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER))
                p.line_spacing = 1.0
                p.space_before = Pt(0)
                if math:
                    add_math(p, _strip(line), size, color, bold=(r == 0))
                else:
                    add_rich(p, line, size, color, bold=(r == 0), sub=sub)
    return gf


# ---------------- 页面骨架 ----------------
class Deck:
    def __init__(self, prs, logo_path, total=25):
        self.prs = prs
        self.logo = logo_path
        self.total = total

    def page(self, title, kicker=None, part=None, num=None):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        rect(s, MARGIN, 0.50, 0.085, 0.40, BLUE)
        block(s, MARGIN + 0.22, 0.28, 10.2, 0.80,
              [dict(text=title, size=28, bold=True, color=INK, ls=1.0)],
              anchor=MSO_ANCHOR.MIDDLE, where=f"标题p{num}")
        s.shapes.add_picture(self.logo, Inches(11.30), Inches(0.39), width=Inches(1.41))
        rect(s, MARGIN, 1.17, CW, 0.012, LINE)
        if kicker:
            block(s, MARGIN, 1.24, CW, 0.42,
                  [dict(text=kicker, size=20, color=GRAY, ls=1.0, sub=True)],
                  anchor=MSO_ANCHOR.MIDDLE, where=f"副标题p{num}")
        if part:
            block(s, MARGIN, 7.03, 7.0, 0.3,
                  [dict(text=part, size=13, color=MUTED, ls=1.0)],
                  anchor=MSO_ANCHOR.MIDDLE, where=f"页脚{num}")
        if num:
            block(s, 10.9, 7.03, 1.8, 0.3,
                  [dict(text=f"{num:02d} / {self.total:02d}", size=13, color=MUTED, ls=1.0)],
                  align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, where=f"页码{num}")
        rect(s, 0, 7.46, W, 0.04, BLUE)
        return s
