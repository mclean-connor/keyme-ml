import tensorflow as tf
from tf2cv.models.resnetd import resnetd50b
from tf2cv.models.icnet import get_icnet
import cv2


def icnet_builder(
    pretrained_backbone: bool = False,
    classes: int = 2,
    aux: bool = True,
    data_format: str = "channels_last",
    **kwargs,
) -> tf.keras.Model:

    backbone1 = resnetd50b(
        pretrained=pretrained_backbone,
        ordinary_init=False,
        bends=None,
        data_format=data_format,
    ).features
    for i in range(len(backbone1) - 3):
        # backbone1.children.pop()
        del backbone1.children[-1]
    backbone2 = resnetd50b(
        pretrained=pretrained_backbone,
        ordinary_init=False,
        bends=None,
        data_format=data_format,
    ).features
    # backbone2.children.pop()
    del backbone2.children[-1]
    for i in range(3):
        # backbone2.children.pop(0)
        del backbone2.children[0]
    backbones = (backbone1, backbone2)
    backbones_out_channels = (512, 2048)

    return get_icnet(
        backbones=backbones,
        backbones_out_channels=backbones_out_channels,
        classes=classes,
        aux=aux,
        data_format=data_format,
        **kwargs,
    )


base_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


class ICNetLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        num_classes,
        ignore_label: int = -1,
        lambda_values=(0.16, 0.4, 1.0),  # lambda 1, 2, 3
        name="ICNetLoss",
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.lambda_values = lambda_values  # Using the provided lambda values

    # prepare the ground truth mask by resizing to that of the prediction and cast to float
    def prepare_label(
        self, mask, new_shape: tuple, num_classes: int, one_hot: bool = True
    ):
        # As labels are integer numbers, need to use NN interp.
        input_batch = tf.image.resize(
            mask, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        mask = tf.image.rgb_to_grayscale(input_batch)
        threshold = 0.5
        # since we are using binary mask, we can use a threshold to convert to 0s and 1s
        mask = tf.cast(mask > threshold, dtype=tf.float32)  # type: ignore
        input_batch = tf.squeeze(mask, axis=3)
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)

        return input_batch

    def get_mask(self, gt, num_classes, ignore_label):
        # this is in shape (None, h, w, 1)
        """Form mask with option to ignore a label."""
        less_equal_class = tf.less_equal(gt, num_classes - 1)
        not_equal_ignore = tf.not_equal(gt, ignore_label)
        mask = tf.logical_and(less_equal_class, not_equal_ignore)
        indices = tf.squeeze(tf.where(mask), 1)

        # print indices values
        return indices

    def calculate_loss(self, y_true, output, loc):
        # add dimention to the prediction
        raw_prediction = tf.reshape(output, [-1, self.num_classes])
        label = self.prepare_label(
            y_true,
            tf.stack(tf.shape(output)[1:3]),  # type: ignore
            num_classes=2,
            one_hot=False,
        )

        label = tf.reshape(
            label,
            [
                -1,
            ],
        )

        # get loss and reduce mean
        loss = base_loss(label, raw_prediction)
        reduced_loss = tf.reduce_mean(loss)
        return reduced_loss

    def call(self, y_true, y_preds):
        # the logits we want to use are the 1,2,3 index of the predictions
        low_res_logits = y_preds[3]
        mid_res_logits = y_preds[2]
        high_res_logits = y_preds[1]

        # same mask applies to all logits
        # calulate the loss for each resolution
        low_res_loss = self.calculate_loss(y_true, low_res_logits, 1)
        mid_res_loss = self.calculate_loss(y_true, mid_res_logits, 2)
        high_res_loss = self.calculate_loss(y_true, high_res_logits, 3)

        # sum the losses and apply the lambda values
        total_loss = (
            self.lambda_values[0] * low_res_loss
            + self.lambda_values[1] * mid_res_loss
            + self.lambda_values[2] * high_res_loss
        )
        return total_loss
