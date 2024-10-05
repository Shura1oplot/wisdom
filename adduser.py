#!/usr/bin/env python3

import sys
import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).parent.absolute()

PASSWORD_SALT = os.environ["PASSWORD_SALT"]


def main(argv=sys.argv):
    username = argv[0]
    password = argv[1]

    hash_ = hashlib.sha512(
        (password + PASSWORD_SALT).encode("utf-8")).hexdigest()

    with open(BASE_DIR / "users.txt", "w", encoding="utf-8") as fp:
        fp.write(f"{username}:{hash_}\n")


if __name__ == "__main__":
    sys.exit(main())
