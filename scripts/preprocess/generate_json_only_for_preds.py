
import json
from tqdm import tqdm

path_to_big_json = "sampled_DOVE_100_resamplings_gpqa.json"

with open(path_to_big_json, "r") as f:
    data = json.load(f)

new_json_path = "slim_json_for_preds_100_resamplings_gpqa.json"
data_for_new_json = {}

for sampling in data:
    data_for_new_json[sampling] = {}
    current_sampling = data[sampling]
    source = current_sampling["source"]
    new_source = []
    for dp in source:
        if dp[-1]['role'] == 'assistant':
            dp = dp[:-1]
        new_source.append(dp)
    data_for_new_json[sampling]["source"] = new_source

print("Saving slim json for preds")
with open(new_json_path, "w") as f:
    str_for_file = json.dumps(data_for_new_json, indent=2)
    f.write(str_for_file)

