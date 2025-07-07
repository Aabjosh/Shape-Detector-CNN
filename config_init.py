# simple file to create a default json

# imports
import json

# file constant
FILE = "config.json"

default_values = {
    "batch_size": 2,
    "epochs": 30,
    "learning_rate": 0.0005    
}

current_values = {
    "batch_size": 2,
    "epochs": 30,
    "learning_rate": 0.0005    
}

write_data = {
    "default": default_values,
    "current": current_values
}

# write to json
with open(FILE, "w") as file:
    json.dump(write_data, file, indent=4)