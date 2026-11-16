from PIL import Image, ImageFilter
src = r"experiment\ELSE\LDAS\cqu_logo.png"
im = Image.open(src)
print(im.mode, im.size, im.info.get("transparency"))
rgba = im.convert("RGBA")
bg = Image.new("RGBA", rgba.size, (255,255,255,255))
flat = Image.alpha_composite(bg, rgba).convert("RGB")
w, h = flat.size
im2 = flat.resize((w*4, h*4), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=2.2, percent=60, threshold=2))
im2.save(r"work\ppt_adarec\cqu_logo_hd.png")
print("ok", im2.size)
