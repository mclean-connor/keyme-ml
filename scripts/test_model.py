from models import models
import argparse
import cv2

parser = argparse.ArgumentParser(description="Make prediction with a model.")
# add arge for model name
parser.add_argument(
    "--model", "-m", type=str, help="The model to test", choices=models.keys()
)
# add arg for weights path
parser.add_argument(
    "--weights", "-w", type=str, help="The weights to load", default=None
)
# add arg for image path
parser.add_argument("--image", "-i", type=str, help="The image to test")
# add arg for output path
parser.add_argument(
    "--output", "-o", type=str, help="The output path", default="output.png"
)

args = parser.parse_args()


def predict(model_name: str, image_path: str, weights_path: str, output_path: str):
    """
    Make a prediction with the given model
    """
    # load the model... specify that we are not training.
    model = models[model_name](training=False)

    # make the prediction... specify the weights to load
    mask = model.predict(image_path, weights_path)
    # save the prediction
    cv2.imwrite(output_path, mask)


if __name__ == "__main__":
    # make sure required arguments are provided
    if not args.model or not args.image or not args.weights:
        parser.print_help()
        exit(1)

    predict(args.model, args.image, args.weights, args.output)
