import re
import sys

ids = []
for line in sys.stdin:
    result = re.findall(r'#ID#[0-9]+?#', line)
    if len(result) > 0:
        ids.extend([int(r[4:-1]) for r in result])

for id_ in sorted(ids):
    print(id_)
