import sys, io, openpyxl
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = r"F:\recsys-research-training-zhangxinyu\information\你说的队\项目统计表_抽选结果.xlsx"
wb = openpyxl.load_workbook(p, data_only=False)
ws = wb["Sheet1"]
for r in range(1, ws.max_row + 1):
    print([ws.cell(r, c).value for c in range(1, 7)])
print("dims", ws.dimensions, "sheets", wb.sheetnames)
