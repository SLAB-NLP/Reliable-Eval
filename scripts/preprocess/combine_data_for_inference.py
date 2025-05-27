
from argparse import ArgumentParser
import json
import os
import random
random.seed(42)


def combine_all_data_for_predictions(data_dir, output_path, num_samples):
    all_data = {}
    print("Combining all data from {}".format(data_dir))

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), "r") as f:
                data = json.load(f)
                all_data.update(data)

    print("sampling {} samples from {} total samples".format(num_samples, len(all_data)))
    sample_keys_from_data(all_data, num_samples, output_path)


def sample_keys_from_data(all_data, num_samples, output_path, include_keys=None):
    list_of_keys = list(all_data.keys())
    list_of_keys = [key for key in list_of_keys if "placeCorrectChoice" not in key]

    # sample num_samples keys, without replacement
    data_for_output = {}

    if include_keys is not None:
        print("data will include those in {}".format(include_keys))
        for key in include_keys:
            data_for_output[key] = all_data[key]

        num_samples = num_samples - len(include_keys)
        print("will sample only {} samples".format(num_samples))
        list_of_keys = [key for key in list_of_keys if key not in include_keys]

    random_keys = random.sample(list_of_keys, num_samples)
    for key in random_keys:
        data_for_output[key] = all_data[key]
    with open(output_path, "w") as f:
        json.dump(data_for_output, f, indent=2)
    print("data saved to", output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    combine_all_data_for_predictions(args.dir, args.out, args.num_samples)
