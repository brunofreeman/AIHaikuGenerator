from sip_lstm import train_lstm
import json

json_data = json.load(open('sip_trainer_config.json'))

max_steps = json_data['max_steps']
log_every = json_data['log_every']
save_every = json_data['save_every']

train_lstm(max_steps, log_every, save_every)