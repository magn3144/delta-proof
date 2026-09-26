"""Summarize the JSONL protocol emitted by AlphaProof inference."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-results', type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input.open(encoding='utf-8') as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    results = [record for record in records if record['type'] == 'result']
    if len(results) != args.expected_results:
        raise ValueError(
            f'Expected {args.expected_results} results, found {len(results)}.'
        )

    statuses = {
        status: sum(result['status'] == status for result in results)
        for status in ('proved', 'failed', 'rejected')
    }
    summary = {
        'total': len(results),
        **statuses,
        'success_rate': statuses['proved'] / len(results),
        'total_search_seconds': sum(
            result['duration_seconds'] for result in results
        ),
        'simulations_allocated': sum(
            result['simulations_allocated'] for result in results
        ),
        'simulations_used': sum(
            result['simulations_used'] for result in results
        ),
    }
    args.output.write_text(
        json.dumps(summary, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
