try:
    import Image
    import ImageDraw
    import ImageFont
    import ImageOps

except ImportError:
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont
    from PIL import ImageOps

from string import ascii_lowercase
import glob
import pathlib

fonts = glob.glob("fonts/for_ukr/*.ttf")
#fonts = ['fonts\\Action Man Bold.ttf']
image = Image.open('pictures/out.jpg').convert('RGBA')

for c in ascii_lowercase:
    pathlib.Path('data/lower/%s' % c).mkdir(parents=True, exist_ok=True)
    for font in fonts:
        text_to_show = c
        txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
        d = ImageDraw.Draw(txt)
        fnt = ImageFont.truetype(font, 50)
        d.text((20, 20), text_to_show, font=fnt, fill=(255, 255, 255, 255))
        out = Image.alpha_composite(image, txt)
        if out.mode == 'RGBA':
            r, g, b, a = out.split()
            rgb_image = Image.merge('RGB', (r, g, b))

            inverted_image = ImageOps.invert(rgb_image)

            r2, g2, b2 = inverted_image.split()

            final_image = Image.merge('RGBA', (r2, g2, b2, a))

            # final_transparent_image.save('image_samples_with_fonts/'+font[6:-4]+'.bmp')
            final_image.save('data/lower/%s/' % c + font[14:-4] + '.bmp')




