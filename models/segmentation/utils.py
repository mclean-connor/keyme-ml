# import needed packages
import os
import random
import glob
import numpy as np
from PIL import Image
from typing import List
import tensorflow as tf
import tensorflow_addons as tfa
import math

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


def get_image_mask_pair(
    in_img: str,
    in_resize: tuple,
    out_resize: tuple,
    augment=True,
    expand_dims=True,
):
    img = load_image_tf(
        in_img
    )  # Replace `load_image_tf` with your actual function to load an image

    if img is None:
        raise ValueError(f"Could not load image from {in_img}")

    # mask is first half of the image, image is second half
    mask, input_img = tf.split(img, num_or_size_splits=2, axis=1)

    if in_resize:
        input_img = tf.image.resize(
            input_img, in_resize[:2], method=tf.image.ResizeMethod.BICUBIC
        )

    if out_resize:
        mask = tf.image.resize(
            mask, out_resize[:2], method=tf.image.ResizeMethod.BICUBIC
        )

    if augment:
        # Horizontal flipping
        if random.random() < 0.5:
            input_img = tf.image.flip_left_right(input_img)
            mask = tf.image.flip_left_right(mask)

        # Random rotation
        if random.random() < 0.5:
            # Choose a random angle between -15 and 15 degrees
            angle = (
                random.uniform(-15, 15) * math.pi / 180
            )  # Convert degrees to radians

            # Rotate both the input image and mask by the same angle
            input_img = tfa.image.rotate(
                input_img, angles=angle, interpolation="BILINEAR"
            )
            mask = tfa.image.rotate(mask, angles=angle, interpolation="NEAREST")

        # add random gaussian noise
        if random.random() < 0.5:
            noise = tf.random.normal(
                tf.shape(input_img), mean=0.0, stddev=0.1, dtype=tf.float32
            )
            input_img = tf.clip_by_value(input_img + noise, 0, 1) # type: ignore

        # TODO: need to validate these augmentations
        # # Random brightness with minor adjustment
        # if random.random() < 0.5:
        #     input_img = tf.image.random_brightness(input_img, max_delta=0.05)

        # # Random contrast with minor adjustment
        # if random.random() < 0.5:
        #     input_img = tf.image.random_contrast(input_img, lower=0.95, upper=1.05)

        # # Random saturation with minor adjustment
        # if random.random() < 0.5:
        #     input_img = tf.image.random_saturation(input_img, lower=0.95, upper=1.05)

        # # Custom crop and scale
        # if random.random() < 0.5:
        #     combined = tf.concat(
        #         [input_img, mask], axis=-1
        #     )  # Combine image and mask to apply the same crop

        #     crop_size = [
        #         int(in_resize[0] * 0.8),
        #         int(in_resize[1] * 0.8),
        #         combined.shape[-1],  # type: ignore
        #     ]  # 80% crop

        #     combined_cropped = tf.image.random_crop(combined, size=crop_size)

        #     # Separating the cropped image and mask, each with 3 channels
        #     input_img_cropped, mask_cropped = tf.split(
        #         combined_cropped, [3, 3], axis=-1
        #     )

        #     # Resizing back to desired size
        #     input_img = tf.image.resize(
        #         input_img_cropped, in_resize[:2], method=tf.image.ResizeMethod.BICUBIC
        #     )
        #     mask = tf.image.resize(
        #         mask_cropped, out_resize[:2], method=tf.image.ResizeMethod.BICUBIC
        #     )

    return (
        tf.cast(input_img, tf.float32),
        (
            tf.expand_dims(tf.cast(mask, tf.float32), -1)
            if expand_dims
            else tf.cast(mask, tf.float32)
        ),
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
        img.set_shape(  # type: ignore
            [None, None, 3]
        )  # If the height and width are variable, but the number of channels is known

    return img


def load_training_images(image_path: str, config: BaseModelConfig, expand_dims=True):
    X, y = get_image_mask_pair(
        image_path,
        in_resize=config.default_in_shape,
        out_resize=config.default_out_shape,
        expand_dims=expand_dims,
    )

    return tf.cast(X, tf.float32) / 255.0, tf.cast(y, tf.float32) / 255.0  # type: ignore
