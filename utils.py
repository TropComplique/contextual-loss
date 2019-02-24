from PIL import Image, ImageDraw


def show(images):
    """
    Shows a list of images.
    Images can be of different sizes.
    """
    width = max(i.size[0] for i in images)
    height = sum(i.size[1] for i in images)
    background = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(background, 'RGB')

    offset = 0
    for i in images:
        _, h = i.size
        background.paste(i, (0, offset))
        offset += h

    return background
