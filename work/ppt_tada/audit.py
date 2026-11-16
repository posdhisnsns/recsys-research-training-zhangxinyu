# -*- coding: utf-8 -*-
"""检查排版：底部空白过大 / 元素重叠 / 出界。"""
from pptx import Presentation
from pptx.util import Emu

PATH = r"information\PPT\张心雨_TADA文献阅读及代码复现_组会汇报.pptx"
p = Presentation(PATH)
W, H = 13.333, 7.5

# 装饰性/页眉页脚元素：不参与空白与重叠判断
def is_chrome(sh):
    l, t = sh.left / 914400, sh.top / 914400
    w, h = sh.width / 914400, sh.height / 914400
    if t < 1.30 and h < 1.0:              # 顶部标题、校徽、竖条
        return True
    if t > 6.95:                          # 页脚与底部线条
        return True
    if h < 0.06:                          # 分隔细线
        return True
    if w > 13.2 or h > 7.4:               # 全屏底色
        return True
    return False

for i, s in enumerate(p.slides, 1):
    boxes = []
    for sh in s.shapes:
        try:
            l, t = sh.left / 914400, sh.top / 914400
            w, h = sh.width / 914400, sh.height / 914400
        except TypeError:
            continue
        boxes.append((sh, l, t, w, h))

    content = [(sh, l, t, w, h) for sh, l, t, w, h in boxes if not is_chrome(sh)]
    bottom = max((t + h for _, _, t, _, h in content), default=0)
    issues = []
    if bottom < 6.15:
        issues.append(f"底部空白 {7.0-bottom:.1f}in")
    for _, l, t, w, h in content:
        if t + h > 7.2:
            issues.append(f"出界 bottom={t+h:.2f}")
            break
    # 重叠：只看文本/图片/表格，忽略卡片背景与上面的文字（卡片内文字必然相交）
    for a in range(len(content)):
        sa, la, ta, wa, ha = content[a]
        for b in range(a + 1, len(content)):
            sb, lb, tb, wb, hb = content[b]
            ox = min(la + wa, lb + wb) - max(la, lb)
            oy = min(ta + ha, tb + hb) - max(ta, tb)
            if ox > 0.25 and oy > 0.25:
                # 卡片与其上文字属正常包含关系，跳过
                inside = (la <= lb and ta <= tb and la + wa >= lb + wb and ta + ha >= tb + hb)
                inside2 = (lb <= la and tb <= ta and lb + wb >= la + wa and tb + hb >= ta + ha)
                if inside or inside2:
                    continue
                issues.append(f"重叠 {sa.name[:12]}×{sb.name[:12]} ({ox:.1f}x{oy:.1f})")
    if issues:
        print(f"p{i:02d}: " + " | ".join(issues[:4]))
