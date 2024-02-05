import boto3
from PIL import Image
import io
import onnx
import tf2onnx
from tensorflow import keras
import subprocess
import json
from typing import List, Union


def list_s3_dvc_files(bucket: str, prefix: str) -> List[str]:
    """
    List all files in the DVC repository.
    """
    data_path = prefix.split("/")[-1]

    dvc_files = []
    result = subprocess.run(
        ["dvc", "list", ".", f"training-data/{data_path}", "--dvc-only", "--json"],
        capture_output=True,
        text=True,
    )

    # Parse the JSON output
    files = json.loads(result.stdout)
    dvc_files = [f's3://{bucket}/{prefix}{file["path"]}' for file in files]

    return dvc_files


def list_s3_objects(bucket: str, prefix: str) -> List[str]:
    """
    List all image files in the S3 bucket under the specified prefix.
    """
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    all_images = []
    for page in page_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith((".png", ".jpg", ".jpeg")):
                    all_images.append(f"s3://{bucket}/{key}")

    return all_images


def load_image_from_s3(s3_path: str) -> Union[Image.Image, None]:
    # Parse the S3 URL
    s3_path = s3_path.replace("s3://", "")
    parts = s3_path.split("/")
    bucket_name = parts[0]
    key = "/".join(parts[1:])

    # Create an S3 client
    s3 = boto3.client("s3")

    # Get the image object from S3
    s3_response = s3.get_object(Bucket=bucket_name, Key=key)

    # if the response is not successful, return None
    if s3_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        return None

    # Read the image data in bytes
    image_data = s3_response["Body"].read()

    # Convert bytes data to a PIL Image
    image = Image.open(io.BytesIO(image_data))

    return image


def tf2_to_onnx(model: keras.Model, onnx_path: str) -> None:
    # Convert the TensorFlow model to ONNX
    print("Converting the model to ONNX")
    print("cman", model)
    onnx_model = tf2onnx.convert.from_keras(model)
    # Save the ONNX model
    onnx.save_model(onnx_model, onnx_path)


if __name__ == "__main__":
    print(list_s3_dvc_files("keyme-data", "training-data/bitting_left"))
