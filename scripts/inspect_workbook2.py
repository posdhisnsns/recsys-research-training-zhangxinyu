import sys, io, openpyxl
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

path = r"F:\recsys-research-training-zhangxinyu\information\你说的队\项目统计表.xlsx"
wb = openpyxl.load_workbook(path, data_only=False)
ws = wb["Sheet1"]

def fill_key(c):
    f = c.fill
    if f is None or f.fill_type != "solid":
        return None
    fg = f.fgColor
    if fg.type == "theme":
        return f"theme{fg.theme},tint{round(fg.tint,2)}"
    return f"{fg.type}:{fg.rgb or fg.indexed}"

print("--- all cells ---")
for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
    cells = []
    for c in row:
        cells.append(f"{c.coordinate}[{fill_key(c)}]={c.value!r}")
    print(" | ".join(cells))

highlighted = []
for r in range(1, ws.max_row + 1):
    if fill_key(ws.cell(r, 2)) is not None:
        highlighted.append((r, ws.cell(r, 1).value, ws.cell(r, 2).value, ws.cell(r, 3).value,
                            ws.cell(r, 4).value, ws.cell(r, 5).value))
print("\n--- highlighted rows (col B filled) ---")
for h in highlighted:
    print(h)
print("count:", len(highlighted))

print("\n--- header-ish row 1 & widths ---")
print([ws.cell(1, i).value for i in range(1, 6)])
