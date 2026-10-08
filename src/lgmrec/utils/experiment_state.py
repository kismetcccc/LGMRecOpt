"""Content identity for repeatable checkpoint evaluation and log comparisons."""
import hashlib
import json
from pathlib import Path


def fingerprint_inputs(config):
    directory = Path(config['data_path']) / config['dataset']
    hashes = {}
    for key in ('inter_file_name', 'vision_feature_file', 'text_feature_file'):
        digest = hashlib.sha256()
        with (directory / config[key]).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        hashes[key] = digest.hexdigest()
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return hashes, signature
