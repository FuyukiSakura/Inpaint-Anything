import albumentations
from albumentations.core.transforms_interface import DualTransform

try:
    from albumentations import to_tuple
except ImportError:
    def to_tuple(x, low=0):
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (low, x)

try:
    import imgaug.augmenters as iaa
    _HAS_IMGAUG = True
except ImportError:
    _HAS_IMGAUG = False


class IAAAffine2(DualTransform):
    def __init__(
        self,
        scale=(0.7, 1.3),
        translate_percent=None,
        translate_px=None,
        rotate=0.0,
        shear=(-0.1, 0.1),
        order=1,
        cval=0,
        mode="reflect",
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.scale = dict(x=scale, y=scale)
        self.translate_percent = to_tuple(translate_percent, 0)
        self.translate_px = to_tuple(translate_px, 0)
        self.rotate = to_tuple(rotate)
        self.shear = dict(x=shear, y=shear)
        self.order = order
        self.cval = cval
        self.mode = mode

    def apply(self, img, **params):
        if not _HAS_IMGAUG:
            return img
        aug = iaa.Affine(
            self.scale, self.translate_percent, self.translate_px,
            self.rotate, self.shear, self.order, self.cval, self.mode,
        )
        return aug(image=img)

    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)

    def get_transform_init_args_names(self):
        return ("scale", "translate_percent", "translate_px", "rotate", "shear", "order", "cval", "mode")


class IAAPerspective2(DualTransform):
    def __init__(self, scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5,
                 order=1, cval=0, mode="replicate"):
        super().__init__(always_apply, p)
        self.scale = to_tuple(scale, 1.0)
        self.keep_size = keep_size
        self.cval = cval
        self.mode = mode

    def apply(self, img, **params):
        if not _HAS_IMGAUG:
            return img
        aug = iaa.PerspectiveTransform(self.scale, keep_size=self.keep_size, mode=self.mode, cval=self.cval)
        return aug(image=img)

    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)

    def get_transform_init_args_names(self):
        return ("scale", "keep_size")
