# add the prefect training deployment
prefect deployment build train_model.py:train_model -n train_model
prefect deployment apply train_model-deployment.yaml

# add the prefect dvc deployment
prefect deployment build train_model.py:sync_dvc_data -n sync_dvc
prefect deployment apply sync_dvc_data-deployment.yaml

# run the prefect agent in the background
prefect agent start -p 'default-agent-pool' &
