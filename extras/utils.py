import yaml


def read_yaml(file):
    """ read_yaml """
    with open(file, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return config
