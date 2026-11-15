"""Download the Telco Customer Churn dataset and verify its checksum.

Run from the repository root:

    python data/fetch_data.py

The file is the public IBM sample dataset (Apache-2.0 licensed repository).
I pin the download to one specific commit and check the SHA-256 hash, so
everyone who runs this script trains on exactly the same bytes.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

# The commit hash in the URL freezes the file version.
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_CSV = RAW_DIR / "telco_customer_churn.csv"


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hash of a file as a hex string."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_CSV.exists() and sha256_of(RAW_CSV) == EXPECTED_SHA256:
        print(f"Dataset already present and verified: {RAW_CSV.name}")
        return 0

    print("Downloading dataset ...")
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    RAW_CSV.write_bytes(content)

    actual = sha256_of(RAW_CSV)
    if actual != EXPECTED_SHA256:
        # A different hash means the file changed or the download was cut short.
        print("Checksum mismatch. The downloaded file is not the expected one.")
        print(f"  expected: {EXPECTED_SHA256}")
        print(f"  actual  : {actual}")
        return 1

    print(f"Saved {len(content):,} bytes to data/raw/{RAW_CSV.name}")
    print(f"SHA-256 verified: {actual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
