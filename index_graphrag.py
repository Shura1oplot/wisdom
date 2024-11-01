#!/usr/bin/env python3

import sys
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete

from tqdm import tqdm


load_dotenv()


BASE_DIR = Path(__file__).parent.absolute()
DATABASE_PATH = Path(os.environ["DATABASE_PATH"])
LIGHTRAG_WORKING_DIR = Path(os.environ["LIGHTRAG_WORKING_DIR"])


def main(argv=sys.argv):
    try:
        with open(BASE_DIR / "documents.pickle", "rb") as fp:
            documents = pickle.load(fp)

    except FileNotFoundError:
        documents = []

    print(documents[0])

    return

    rag = LightRAG(
        working_dir=LIGHTRAG_WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete
        # llm_model_func=gpt_4o_complete
    )

    for doc in tqdm(documents):
        rag.insert(doc)


if __name__ == "__main__":
    sys.exit(main())
