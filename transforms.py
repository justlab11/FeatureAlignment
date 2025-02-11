from PIL import Image

class EnsureThreeChannelsPIL(object):
    def __call__(self, img: Image.Image):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img