# install the required packages with poetry
# if poetry is not installed... add it with pip
if ! command -v poetry &> /dev/null
then
    pip install poetry
fi

poetry install

# initialize dvc
dvc init

# pull the latest data from the remote storage
dvc pull
