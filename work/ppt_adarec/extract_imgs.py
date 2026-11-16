import sys
from pptx import Presentation
from pptx.util import Emu
import os, hashlib
p = Presentation(r"information\PPT\张心雨_ADARec文献阅读及代码复现_组会汇报.pptx")
os.makedirs(r"work\ppt_adarec\imgs", exist_ok=True)
for i, s in enumerate(p.slides, 1):
    for sh in s.shapes:
        if sh.shape_type == 13:
            img = sh.image
            h = hashlib.md5(img.blob).hexdigest()[:8]
            fn = os.path.join(r"work\ppt_adarec\imgs", f"s{i:02d}_{h}.{img.ext}")
            if not os.path.exists(fn):
                open(fn,'wb').write(img.blob)
            print(f"s{i:02d} {img.size[0]}x{img.size[1]} {img.ext} {len(img.blob)//1024}KB -> {fn}")
