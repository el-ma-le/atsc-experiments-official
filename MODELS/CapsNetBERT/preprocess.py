import yaml
from data_process.data_process import data_process
import mlflow

config = yaml.safe_load(open('config.yml'))

#mlflow.set_tracking_uri("")
#mlflow.set_experiment("capsnetbert")

#with mlflow.start_run():    
 #   mlflow.log_artifact("config.yml")
data_process(config)
