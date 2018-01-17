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


import glob
fonts = glob.glob("fonts/train/*.ttf")
#fonts = ['fonts\\Action Man Bold.ttf']
text_to_show = "Lorem Ipsum is simply dummy text \n of the printing and typesetting industry.\n Lorem Ipsum has been the industry's \n standard dummy text ever since the 1500s,\n when an unknown printer took a galley\n  of type and scrambled it to make a type \n specimen book. It has survived not only five centuries, \n but also the leap into electronic typesetting,\n remaining essentially unchanged. \n It was popularised in the 1960s with the release of \n Letraset sheets containing Lorem Ipsum passages,\n and more recently with desktop publishing software like \n Aldus PageMaker including versions of Lorem Ipsum."

image = Image.open('pictures/out.jpg').convert('RGBA')


for font in fonts:
    txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(txt)
    fnt = ImageFont.truetype(font, 15)
    d.text((20, 10), text_to_show, font=fnt, fill=(255,255,255, 255))
    out = Image.alpha_composite(image, txt)
    if out.mode == 'RGBA':
        r, g, b, a = out.split()
        rgb_image = Image.merge('RGB', (r, g, b))

        inverted_image = ImageOps.invert(rgb_image)

        r2, g2, b2 = inverted_image.split()

        final_transparent_image = Image.merge('RGBA', (r2, g2, b2, a))

        #final_transparent_image.save('image_samples_with_fonts/'+font[6:-4]+'.bmp')
        final_transparent_image.save('image_samples_with_fonts/train/'+font[12:-4]+'.bmp')



