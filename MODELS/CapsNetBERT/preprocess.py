import yaml
from data_process.data_process import data_process
import mlflow

config = yaml.safe_load(open('config.yml'))

data_process(config)
