"""Read new-run status files; no training, log regexes, or automatic conclusions."""
import argparse
import json
from pathlib import Path


def collect(root):
    rows = []
    for path in sorted(root.rglob('status.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema_version') == 1:
            rows.append(dict(payload, status_path=str(path)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path('data/runs'))
    parser.add_argument('--out', type=Path, help='New JSON file; existing files are never overwritten')
    args = parser.parse_args()
    rows = collect(args.root)
    print('variant seed finished R@20 N@20 R@50 status')
    for row in rows:
        metric = row.get('valid_result') or {}
        print(row['variant'], row['seed'], row['finished'],
              *(metric.get(k, '-') for k in ('recall@20', 'ndcg@20', 'recall@50')),
              row['status_path'])
    print('Descriptive summary only; no equivalence/significance inference.')
    if args.out:
        with args.out.open('x', encoding='utf-8') as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
