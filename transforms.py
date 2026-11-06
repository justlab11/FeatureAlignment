from PIL import Image

class EnsureThreeChannelsPIL(object):
    def __call__(self, img: Image.Image):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

def train_augmentations(kind: str, dataset_name: str, size: int) -> list:
    """
    PIL augmentations for the training images, applied to target and source alike so they add variation within each
    domain without adding a difference between them. Geometric only: photometric changes would blur the colour and
    texture shift that ShiftNet is there to learn.
    """
    from torchvision import transforms
    if kind == "none":
        return []
    if kind != "geometric":
        raise ValueError(f"Unknown augment option: {kind}")
    name = dataset_name.lower()
    if "mnist" in name or "svhn" in name:  # digits: small shifts and tilts; no flips (2/5, 6/9)
        return [transforms.RandomAffine(degrees=8, translate=(2 / size, 2 / size), fill=0)]
    return [  # objects: crops and horizontal flips keep the class
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
