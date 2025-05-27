path = "predictions/Gpt-4o_predictions.json"
import json
import tiktoken

with open(path) as f:
    data = json.load(f)

enc = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
counter = 0
above_20 = 0
for resampling in data:
    for pred in data[resampling]:
        current = len(enc.encode(pred))
        if current > 30:
            above_20 += 1
        counter += current

print(above_20, "above 20")
print("total tokens", counter)