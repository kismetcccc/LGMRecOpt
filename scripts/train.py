"""Source-checkout entry; installed users may use lgmrec-train."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lgmrec.cli import main

if __name__ == '__main__':
    main()
