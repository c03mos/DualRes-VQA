import yaml


def get_yaml_data(yaml_file):
    yaml_file = open(yaml_file, "r", encoding="utf-8")
    file_data = yaml_file.read()
    yaml_file.close()
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return data