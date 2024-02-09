# install the required packages with poetry
# if poetry is not installed... add it with pip
if ! command -v poetry &> /dev/null
then
    pip install poetry
fi

poetry install

# activate the virtual environment
source $(poetry env info --path)/bin/activate

# pull the latest data from the remote storage
dvc pull

# run the prefect agent in the background if args are passed for --start-agent
if [ "$1" == "--start-agent" ]
then
    prefect agent start -p 'default-agent-pool' &
fi
