"""Sample theorem records and convert them to inference requests."""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--batch-id', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input.open(encoding='utf-8') as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    selected = random.Random(args.seed).sample(
        list(enumerate(records)),
        args.sample_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as output_file:
        for sample_index, (dataset_index, record) in enumerate(selected):
            request = {
                'request_id': f'{args.batch_id}:{sample_index}',
                'theorem_id': record['id'],
                'source': 'dataset',
                'attempt': 0,
                'theorem': record['theorem'],
                'dataset_index': dataset_index,
            }
            output_file.write(json.dumps(request) + '\n')

    print(
        f'Sampled {len(selected):,} of {len(records):,} records from '
        f'{args.input} with seed {args.seed}.',
        flush=True,
    )


if __name__ == '__main__':
    main()
