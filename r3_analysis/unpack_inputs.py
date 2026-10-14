"""Verify and unpack the compact R3 test-prediction bundle."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=root / 'results/r3/recompute_inputs.zip')
    p.add_argument('--output', type=Path, default=Path('reproduced/r3_inputs'))
    args = p.parse_args()
    expected = (root / 'results/r3/RECOMPUTE_INPUTS.sha256').read_text().split()[0]
    if hashlib.sha256(args.bundle.read_bytes()).hexdigest() != expected:
        raise SystemExit('bundle SHA256 mismatch')
    with zipfile.ZipFile(args.bundle) as archive:
        for line in archive.read('SHA256SUMS.txt').decode().splitlines():
            sha, name = line.split(maxsplit=1)
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise SystemExit('invalid archive member')
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != sha:
                raise SystemExit(f'archive checksum mismatch: {name}')
            out = args.output / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
    print('verified compact inputs:', args.output)


if __name__ == '__main__':
    main()
