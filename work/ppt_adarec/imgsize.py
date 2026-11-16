from PIL import Image
for n in ["fig1","fig2","fig3","fig4","fig5","fig6"]:
    p = rf"knowledge_base\paper_reading\assets\ADARec\{n}.png"
    im = Image.open(p); print(n, im.size, round(im.size[0]/im.size[1],3))
for f in ["s04_0e90614f","s05_24afa779","s06_7714d148"]:
    im = Image.open(rf"work\ppt_adarec\imgs\{f}.png"); print(f, im.size, round(im.size[0]/im.size[1],3))
