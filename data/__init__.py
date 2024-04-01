import sys

sys.path.append("../")
from extras.utils import read_yaml


DATA_INFO = read_yaml("../data/conf/data.yaml")
ROLE = read_yaml("../data/conf/role.yaml")
TEMPLATE = read_yaml("../data/conf/templates.yaml")
