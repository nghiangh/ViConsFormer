import yaml
import argparse
from typing import Text
from task import Task

def main(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    main_task = Task(config)
    main_task.training()
    main_task.get_predictions()
    
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    main(args.config)