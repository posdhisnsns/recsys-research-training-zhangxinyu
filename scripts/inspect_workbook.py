import openpyxl

path = r"F:\recsys-research-training-zhangxinyu\information\你说的队\项目统计表.xlsx"
wb = openpyxl.load_workbook(path, data_only=False)
print("sheets:", wb.sheetnames)
for ws in wb.worksheets:
    print("=" * 60)
    print("sheet:", ws.title, ws.dimensions, "max_row", ws.max_row, "max_col", ws.max_column)
    print("merged:", ws.merged_cells.ranges)
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 60), max_col=min(ws.max_column, 25)):
        vals = []
        for c in row:
            fill = c.fill
            fg = None
            if fill is not None and fill.fill_type == "solid":
                fg = fill.fgColor.rgb if fill.fgColor.type == "rgb" else (str(fill.fgColor.theme) + "/" + str(fill.fgColor.tint) if fill.fgColor.type == "theme" else str(fill.fgColor.indexed))
            v = c.value
            if v is not None or fg:
                vals.append(f"{c.coordinate}={v!r}" + (f" [fill {fill.fill_type}:{fg}]" if fg else ""))
        if vals:
            print(" | ".join(vals))
