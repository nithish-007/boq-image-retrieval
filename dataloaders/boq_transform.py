import torch
from torchvision.transforms import transforms as T

def get_boq_transform(transform_config):
    transform = T.Compose([
        T.Resize((transform_config["size"], transform_config["size"])),
        T.ToTensor()
    ])
    return transform

class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'
    
def augmentation(cfg: dict):
    transform_list = []

    if "affine" in cfg:
        affine = cfg["affine"]
        transform_list.append(
            T.RandomAffine(
                degrees=affine.get("rotate", 0),
                translate=affine.get("translate_percent", (0, 0)),
                scale=affine.get("scale", (1.0, 1.0)),
            )
        )

    if "horizontal_flip" in cfg and cfg["horizontal_flip"].get("p", 0.5) > 0:
        transform_list.append(
            T.RandomHorizontalFlip(p=cfg["horizontal_flip"]["p"])
        )

    if "random_brightness_contrast" in cfg:
        rbc = cfg["random_brightness_contrast"]
        transform_list.append(
            T.ColorJitter(
                brightness=rbc["brightness_limit"],
                contrast=rbc["contrast_limit"]
            )
        )

    if "gauss_noise" in cfg:
        gn = cfg["gauss_noise"]
        std = ((gn["var_limit"][0] + gn["var_limit"][1]) / 2) ** 0.5 / 255.0
        transform_list.append(
            AddGaussianNoise(0., std)
        )

    if "rotation" in cfg:
        rot = cfg["rotation"]
        transform_list.append(
           T.RandomRotation(degrees=rot.get("degrees", 10))
        )

    if "perspective" in cfg:
        persp = cfg["perspective"]
        transform_list.append(
            T.RandomPerspective(distortion_scale=persp.get("scale", 0.5), p=persp.get("p", 0.5))
        )

    if "random_erasing" in cfg:
        re = cfg["random_erasing"]
        transform_list.append(
            T.RandomErasing(p=re.get("p", 0.5), scale=re.get("scale", (0.02, 0.33)))
        )

    # Optional: Simulated JPEG compression via resize up + down
    if "compression" in cfg:
        compress = cfg["compression"]
        transform_list.append(
            T.Resize((compress.get("height", 256), compress.get("width", 256)))
        )
        transform_list.append(
            T.Resize((compress.get("orig_height", 224), compress.get("orig_width", 224)))
        )

    # Always convert to tensor last
    transform_list.append(T.ToTensor())

    return T.Compose(transform_list)
