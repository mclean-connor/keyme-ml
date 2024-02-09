# import needed packages
import os
import random
import glob
import numpy as np
from PIL import Image
from typing import List
import tensorflow as tf

from models.base_model import BaseModelConfig


# aborting wget leaves .tmp files everywhere >:(
def clean_dataloader():
    for tmp_file in glob.glob("*.tmp"):
        os.remove(tmp_file)


def format_input(input_image: Image.Image, config: BaseModelConfig):
    assert (
        input_image.size == config.default_in_shape[:2]
        or input_image.size == config.default_in_shape
    )
    inp = np.array(input_image)
    if inp.shape[-1] == 4:
        input_image = input_image.convert("RGB")
    return np.expand_dims(np.array(input_image) / 255.0, 0)


def get_image_mask_pair(in_img: str, in_resize=None, out_resize=None, augment=True):
    img = load_image_tf(in_img)  # type: ignore

    # img = Image.open(in_img).convert("RGB")  # type: ignore
    if img is None:
        raise ValueError(f"Could not load image from {in_img}")

    # mask is first half of the image, image is second half
    mask, input_img = tf.split(img, num_or_size_splits=2, axis=1)

    if in_resize:
        input_img = tf.image.resize(
            input_img, in_resize[:2], method=tf.image.ResizeMethod.BICUBIC
        )
        # img = img.resize(in_resize[:2], Image.BICUBIC)

    if out_resize:
        mask = tf.image.resize(
            mask, out_resize[:2], method=tf.image.ResizeMethod.BICUBIC
        )
        # mask = mask.resize(out_resize[:2], Image.BICUBIC)

    # the paper specifies the only augmentation done is horizontal flipping.
    if augment and bool(random.getrandbits(1)):
        input_img = tf.image.flip_left_right(input_img)
        mask = tf.image.flip_left_right(mask)
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return (
        tf.cast(input_img, tf.float32),
        tf.expand_dims(tf.cast(mask, tf.float32), -1),
    )


def load_image_tf(file_path, set_shape=True):
    # Read the file
    img_raw = tf.io.read_file(file_path)
    # Decode the image
    img = tf.image.decode_image(
        img_raw, channels=3
    )  # Use channels=1 for grayscale, channels=3 for RGB

    if set_shape:
        # this is needed for tf dataset precompiling
        # only use if the image is resized to a specific shape before training
        img.set_shape(
            [None, None, 3]
        )  # If the height and width are variable, but the number of channels is known

    return img


def load_training_images(image_path: str, config: BaseModelConfig):
    # imgs = random.choices(images, k=config.batch_size)
    X, y = get_image_mask_pair(
        image_path,
        in_resize=config.default_in_shape,
        out_resize=config.default_out_shape,
    )

    return tf.cast(X, tf.float32) / 255.0, tf.cast(y, tf.float32) / 255.0  # type: ignore
