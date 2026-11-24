"""Put the bundled mac_slam checkout on sys.path so `Src.*` imports resolve."""

from pathlib import Path
import sys

MACSLAM_PATH = Path(__file__).resolve().parent / "mac_slam"

if str(MACSLAM_PATH) not in sys.path:
    sys.path.insert(0, str(MACSLAM_PATH))
