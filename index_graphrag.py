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
GRAPHRAG_WORKING_DIR = Path(os.environ["GRAPHRAG_WORKING_DIR"])


def main(argv=sys.argv):
    with open(BASE_DIR / "documents.pickle", "rb") as fp:
        documents = pickle.load(fp)

    graphrag = LightRAG(
        working_dir=GRAPHRAG_WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete)

    for doc in tqdm(documents):
        graphrag.insert(doc.get_content())


if __name__ == "__main__":
    sys.exit(main())
