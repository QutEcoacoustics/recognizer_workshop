import os
import sys
import json

sys.path.append(os.getcwd())
from baw_api_utils import baw_api_utils as bau

# todo: refactor this messy "cisticola"config setup
config_file = 'config/config.json'

print('cwd')
print(os.getcwd())

with open(config_file) as f:
  dataset_config = json.load(f)
  config = dataset_config['config']

creds = bau.Credentials(logins_file=config['paths']['logins'], pw=config['credentials_password'])

toolboxes = {}

def get_toolbox_for_instance(instance=config['host'], username=config['username']):
    global toolboxes
    toolbox_id = f'{instance}_{username}'
    if toolbox_id in toolboxes:
        return toolboxes[toolbox_id]
    else:
        tb = bau.ApiToolbox(host=instance, username=username, credentials=creds, parallel=False)
        toolboxes[toolbox_id] = tb
        return tb
