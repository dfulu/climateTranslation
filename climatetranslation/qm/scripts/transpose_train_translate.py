import os
import argparse
from climatetranslation.unit.utils import get_config
import copy
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
args = parser.parse_args()
config = get_config(args.config)

############################################
print(f"{'-'*8}TRANSPOSING ZARR FILE{'-'*8}")
############################################

script = "/home/dfulu/repos/climateTranslation/climatetranslation/get_data/rechunk_zarr_for_time.py"
inzarr1 = config["data_zarr_a"]
inzarr2 = config["data_zarr_b"]
outzarr1 = inzarr1.replace("_zarr", "_transposed_zarr") 
outzarr2 = inzarr2.replace("_zarr", "_transposed_zarr")

os.system(
f"""
    python {script} \
        --inputzarr  {inzarr1}  {inzarr2} \
        --outputzarr  {outzarr1} {outzarr2} \
        --max_mem 40 \
        --n_workers 1 \
        --config {args.config}
"""
)

###################################
print(f"{'-'*8}TRAINING QM{'-'*8}")
###################################

train_config = copy.deepcopy(config)
train_config["data_zarr_a"] = outzarr1
train_config["data_zarr_b"] = outzarr2

with open(r'temp_train_config.yaml', 'w') as file:
    yaml.dump(train_config, file)

script = "/home/dfulu/repos/climateTranslation/climatetranslation/qm/train.py"

os.system(
f"""
    python {script} \
        --config temp_train_config.yaml
"""
)
    
###################################
print(f"{'-'*8}TRANSLATING{'-'*8}")
###################################

script = "/home/dfulu/repos/climateTranslation/climatetranslation/qm/translate.py"

os.system(
f"""
    python {script} \
        --config temp_train_config.yaml \
        --output_zarr {config['out_zarr_a']}\
        --x2x a2b
"""
)

os.system(
f"""
    python {script} \
        --config temp_train_config.yaml \
        --output_zarr {config['out_zarr_b']}\
        --x2x b2a
"""
)

##########################
# clear-up
##########################
# remove transposed files and temp config file
#os.remove("temp_train_config.yaml")
#os.system(f"rm -rf {outzarr1}")
#os.system(f"rm -rf {outzarr2}")