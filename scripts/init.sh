# make sure docker is installed and running
# if not install it and start the service
if ! command -v docker &> /dev/null
then
    # install docker
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce
    sudo systemctl start docker
fi

# make sure docker compose is installed
if ! command -v docker-compose &> /dev/null
then
    # install docker-compose
    sudo apt-get update
    sudo apt-get install -y docker-compose
fi

# make sure nvidia-docker2 is installed
if ! command -v nvidia-docker &> /dev/null
then
    # install nvidia-docker2
    sudo apt-get update
    sudo apt install -y nvidia-docker2
    sudo sustemctl restart docker
fi

# check if pyenv is installed.
# if not install it and use python 3.11 (latest version)
if ! command -v pyenv &> /dev/null
then
    # install pyenv
    curl https://pyenv.run | bash
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    source ~/.bashrc
    pyenv install 3.11
    pyenv use 3.11
fi

# install the required packages with poetry
# if poetry is not installed... add it with pip
if ! command -v poetry &> /dev/null
then
    # install pipx
    sudo apt install pipx
    pipx ensurepath
    pipx install poetry
fi

poetry install

# activate the virtual environment
poetry env use 3.11

# pull the latest data from the remote storage
dvc pull
