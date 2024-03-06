import argparse

# import training script
from models.training import train_new_model

# parse the arguments
parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("--model", "-m", type=str, help="The model to train")


def main(model: str):
    # run the training script
    train_new_model(model)


if __name__ == "__main__":
    args = parser.parse_args()
    # make sure all the arguments are valid
    if args.model is None:
        raise ValueError("Model name is required")

    main(args.model)
