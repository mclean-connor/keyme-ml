# import needed packages
import os
import random
import glob
import numpy as np
from PIL import Image

# project imports
from models.segmentation.U2Net import config
from models.utils import load_image_from_s3

# global cache for training data
cache = None

# aborting wget leaves .tmp files everywhere >:(
def clean_dataloader():
    for tmp_file in glob.glob("*.tmp"):
        os.remove(tmp_file)


def format_input(input_image):
    assert (
        input_image.size == config.default_in_shape[:2]
        or input_image.shape == config.default_in_shape
    )
    inp = np.array(input_image)
    if inp.shape[-1] == 4:
        input_image = input_image.convert("RGB")
    return np.expand_dims(np.array(input_image) / 255.0, 0)


def get_image_mask_pair(in_img, in_resize=None, out_resize=None, augment=True):

    img = load_image_from_s3(in_img)
    if not img:
        raise ValueError(f"Could not load image from {in_img}")

    # mask is first half of the image, image is second half
    mask = img.crop((0, 0, img.width // 2, img.height))
    img = img.crop((img.width // 2, 0, img.width, img.height))

    if in_resize:
        img = img.resize(in_resize[:2], Image.BICUBIC)

    if out_resize:
        mask = mask.resize(out_resize[:2], Image.BICUBIC)

    # the paper specifies the only augmentation done is horizontal flipping.
    if augment and bool(random.getrandbits(1)):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return (
        np.asarray(img, dtype=np.float32),
        np.expand_dims(np.asarray(mask, dtype=np.float32), -1),
    )


def load_training_batch(
    images,
    batch_size=config.batch_size,
    in_shape=config.default_in_shape,
    out_shape=config.default_out_shape,
):
    global cache
    if cache is None:
        cache = images

    imgs = random.choices(cache, k=batch_size)
    image_list = [
        get_image_mask_pair(img, in_resize=in_shape, out_resize=out_shape)
        for img in imgs
    ]

    tensor_in = np.stack([i[0] / 255.0 for i in image_list])  # type: ignore
    tensor_out = np.stack([i[1] / 255.0 for i in image_list])  # type: ignore

    return (tensor_in, tensor_out)
