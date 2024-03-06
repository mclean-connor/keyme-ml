##

# KeyMe ML

## Description

This repository contains code for training KeyMe ML models. The app is built around poetry and pyproject for easy install and uses [Modal](https://modal.com/) for remote training, deployment, and CI/CD.

## Models package

The primary package can be found under the `models` directory. There you can find all models and training methods.

## Data version control

For training locally, data version control can be used... TODO: add instructions

### Environment setup

`.env.example` contains the required environment variable. Copy the file as `.env` and fill in the required values.

### Weights and biases logging

All training runs are logged with [wandb](https://wandb.ai/). Each training run will create a project with name {model_name}. Subsequent training runs can be found under the project and are saparated by timestamp. To set up wandb run the following `wandb login` followed by entering following prompts.

### Dependencies and local training

A setup script has been proviced at `scripts/init.sh'/ simply run the script with the command below:

```bash
./scripts/init.sh
```

If run successfully, skip to step 3 below. For manual setup, begin from step 1.

1. Ensure you have [Poetry](https://python-poetry.org/) installed, along with a matching Python version (recommended to use pyenv for managing Python versions. It will sync with .python-version file).
2. Run the following command to install the dependencies locally:
   ```bash
   poetry install
   ```
3. A dict of the trainable `model_name`s is listed under `models/__init__.py` as `models`. To train a model locally you can run:
   ```bash
   poetry run scripts/train_model.py --model {YOUR_MODEL_NAME}
   ```

### Training remotly and deploying with Modal

Similar to how we train locally, we can also utilize cloud training resources provided by modal.

1. Configure Modal by running the following command:
   ```bash
   modal token set {your_token}
   ```
2. To test training remotly on modal run:
   ```bash
   modal run scripts/train_remote.py --model {YOUR_MODEL_NAME}
   ```
3. After confirming training runs successfully, the app can be deployed to modal with
   ```bash
   modal deploy scripts/train_remote.py
   ```
   Any subsequent calls will update the deployment.
4. Deployments can be invoked through `HTTPS` or a `Python` script.
   - For HTTPS request, the URL takes the format `https://{workspace name}-{modal env name}--{app name slug}-{function name slug}.modal.run?model={model name}&run=remote`
     - NOTE: all slug replace `_` with `-`
   - Docs for running with python and additional notes on HTTPS can be found [here](https://modal.com/docs/guide/trigger-deployed-functions)

### Continuous integration and deployment

CI/CD is integrated with github workflows. See `.github/workflows/ci-cd.yml` for the definition. To integrate, complete the following:

1. Add each of the `.env.example` variables as well as `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to your github repository secrets manager.
2. Configure github workflows for your repository.
3. Each time a PR against `main` is closed and merge, the CI/CD pipeline will run and deploy to Modal.

## Docker

The entire application can also be built and run using Docker. A Docker Compose file is included for running the application in a Docker environment.

### Building and Running with Docker

1. Ensure you have Docker installed on your system.
2. Navigate to the root directory of the project.
3. Run the following command to build and start the containers:
   ```bash
   docker-compose up --build
   ```

## License

TBA

## Author

[Connor McLean](https://github.com/mclean-connor)
