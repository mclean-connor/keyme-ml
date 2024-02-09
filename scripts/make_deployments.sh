# remove all files from the deployment_files directory
rm -rf prefect_flows/deployment_files/*

# add the prefect training deployment
prefect deployment build ./prefect_flows/train_model.py:train_new_model -n train_new_model --output prefect_flows/deployment_files/train-new-model-deployment.yaml --apply

# add the prefect dvc deployment
prefect deployment build ./prefect_flows/train_model.py:sync_dvc_data -n sync_dvc --output prefect_flows/deployment_files/sync_dvc_data-deployment.yaml --apply

# run the prefect agent in the background
# prefect agent start -p 'default-agent-pool' &
