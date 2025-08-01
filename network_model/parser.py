import yaml


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def parse_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)
