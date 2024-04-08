import json
import random
import os

def get_random_config():
    layers = [1,2]
    max_hidden = 100
    min_hidden = 10
    num_layers = random.choice(layers)
    
    nn_config = {}
    
    for i in range(num_layers):
        size = random.randint(min_hidden,max_hidden)
        nn_config["layer_{}".format(i)] = size
    
    return nn_config

outpath = "configs_2layer/"
num_configs = 100
seed = 42

random.seed(seed)

for i in range(num_configs):
    nn_config = get_random_config()
    with open(os.path.join(outpath,"config_{}".format(i)),'w') as f:
        json.dump(nn_config, f)

