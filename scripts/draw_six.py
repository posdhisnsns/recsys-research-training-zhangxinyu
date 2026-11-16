import sys, io, secrets, shutil, datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

src = r"F:\recsys-research-training-zhangxinyu\information\你说的队\项目统计表.xlsx"
dst = r"F:\recsys-research-training-zhangxinyu\information\你说的队\项目统计表_抽选结果.xlsx"

wb = openpyxl.load_workbook(src, data_only=False)
ws = wb["Sheet1"]

rows = [r for r in range(1, ws.max_row + 1)
        if ws.cell(r, 2).fill is not None and ws.cell(r, 2).fill.fill_type == "solid"]
names = {r: ws.cell(r, 2).value for r in rows}
assert len(rows) == 8, rows

rng = secrets.SystemRandom()
picked = rng.sample(rows, 6)          # 6 of 8, uniform, no repeats
picked_sorted = sorted(picked)

print("候选8人：", [names[r] for r in rows])
print("抽中6人（按原表顺序）：", [names[r] for r in picked_sorted])
print("抽中顺序：", [names[r] for r in picked])
print("未抽中2人：", [names[r] for r in rows if r not in picked])

# 在副本里标注结果，不动原文件
hdr = ws.cell(1, 6, "抽选结果")
hdr.font = Font(bold=True)
hdr.alignment = Alignment(horizontal="center")
ws.column_dimensions[get_column_letter(6)].width = 14

green = PatternFill("solid", fgColor="FFC6EFCE")
for r in rows:
    c = ws.cell(r, 6, "抽中" if r in picked else "未抽中")
    c.alignment = Alignment(horizontal="center")
    if r in picked:
        c.fill = green
        c.font = Font(bold=True, color="FF006100")

note = ws.cell(ws.max_row + 2, 1,
    f"抽选说明：从标黄的8人中随机抽取6人（不放回，等概率）。抽选时间 {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
wb.save(dst)
print("saved:", dst)
