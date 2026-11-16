import io
from pptx import Presentation
from pptx.util import Emu
p = Presentation(r"information\PPT\张心雨_ADARec文献阅读及代码复现_组会汇报.pptx")
out = io.StringIO()
out.write(f"slide_size {p.slide_width} {p.slide_height} {p.slide_width/914400:.2f} {p.slide_height/914400:.2f}\n")
out.write(f"num_slides {len(p.slides)}\n")
for i, s in enumerate(p.slides, 1):
    out.write("="*90+"\n")
    out.write(f"SLIDE {i} layout={s.slide_layout.name}\n")
    for sh in s.shapes:
        t = ""
        if sh.has_text_frame:
            t = " || ".join(x.text for x in sh.text_frame.paragraphs if x.text.strip())
        out.write(f"  [{sh.shape_type}] name={sh.name!r} id={sh.shape_id} L={Emu(sh.left).inches:.2f} T={Emu(sh.top).inches:.2f} W={Emu(sh.width).inches:.2f} H={Emu(sh.height).inches:.2f}\n     TEXT: {t}\n")
open(r"work\ppt_adarec\dump.txt","w",encoding="utf-8").write(out.getvalue())
print("done")
