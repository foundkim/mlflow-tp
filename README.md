# UV

source .venv/bin/activate
uv add mlflow
uv add --dev ipykernel

# Git

git tag <tag-name>
git push origin --tags

# Start MLFlow server

`mlflow server --host 127.0.0.1 --port 8080`

# Local 

`./run-pipeline.sh data/raw/iris.csv data/processed`

# MLFLow Project

`pyproject2conda yaml -f pyproject.toml >> python_env.yaml`

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:8080"
export MLFLOW_EXPERIMENT_NAME="iris_experiment"

mlflow run https://github.com/Orianne-B/mlops-with-mlflow.git -P input_data=/home/administrateur/exo/mlops-with-mlflow/data/raw/iris.csv -P processed_data_folder=/home/administrateur/exo/mlops-with-mlflow/data/processed --env-manager=local
```

# MLFLow Serve

`mlflow models serve -m runs:/8faa59a98d314ce5b9adbd1c8e01cc1d/model -p 8081 --env-manager=local`

https://mlflow.org/docs/latest/deployment/deploy-model-locally/

`curl http://127.0.0.1:8081/invocations -H "Content-Type:application/json"  --data '{"inputs": [{"sepal_lenght":6.1,"sepal_width":3.0,"petal_lenght":4.6,"petal_width":1.4}]}'`

Poissible aussi de faire requete en python, cf documentation

# FastAPI

`python3 app.py`

Doc
- lifespan : chose à faire au démarage et à la fin de l'application, gestion des dépendances synchrones

# Docker

```bash
cd fastapi-app
sudo docker build -t mlflow-service:latest .
sudo docker run --env-file .env --name mlflow-container -p 5050:5089 -d mlflow-service 
sudo docker ps
sudo docker inspect <container_name>
```

<IPAddress>:5050

```bash
docker stop mlflow-container
docker rm mlflow-container
```

## Access MLFLow server

Restart MLFLow server
mlflow server --host 0.0.0.0 --port 8080

ip add -> get ip address

`sudo docker exec -it mlflow-container bash`

# GitHub actions

https://docs.github.com/fr/actions/writing-workflows/workflow-syntax-for-github-actions


# Azure Dev Ops

dev.azure.com/<"nom de l'organisation">

https://learn.microsoft.com/fr-fr/azure/machine-learning/how-to-setup-mlops-azureml?view=azureml-api-2&tabs=azure-shell

https://learn.microsoft.com/fr-fr/azure/machine-learning/how-to-devops-machine-learning?view=azureml-api-2&tabs=arm

# Example Antoine

https://github.com/hanabi70/m2i_formation/tree/mlflow
