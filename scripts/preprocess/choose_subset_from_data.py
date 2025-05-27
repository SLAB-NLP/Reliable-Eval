from argparse import ArgumentParser
import json

from scripts.preprocess.combine_data_for_inference import sample_keys_from_data

def sample_subset(path_to_data, out, num_samples, partial_data):

    print("taking a sample of {} samples from {}".format(num_samples, path_to_data))
    with open(path_to_data, 'rt') as f:
        data = json.load(f)

    include_keys = None
    if partial_data:
        print("samples will include those in {}".format(partial_data))
        with open(partial_data, 'rt') as f:
            partial_data = json.load(f)
        include_keys = set(partial_data.keys())

    sample_keys_from_data(data, num_samples, out, include_keys)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--full_data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--partial_data", default="")
    args = parser.parse_args()

    sample_subset(args.full_data, args.out, args.num_samples, args.partial_data)