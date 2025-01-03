#!/usr/bin/env python3

import sys
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import Settings

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.openai import OpenAI

from llama_parse import LlamaParse

import chromadb

from tqdm import tqdm


load_dotenv()


BASE_DIR = Path(__file__).parent.absolute()
DATABASE_PATH = Path(os.environ["DATABASE_PATH"])

PRIMARY_FORMAT = os.environ["PRIMARY_FORMAT"]
FILE_EXTENSIONS = os.environ["FILE_EXTENSIONS"]

LLM_GPT_4_MINI = "gpt-4o-mini-2024-07-18"

MODEL_EMBEDDING = "text-embedding-3-large"
EMBED_MODEL_DIM = 3072
EMBED_BATCH_SIZE = 256

TEMPERATURE_DEFAULT = 0.0
MAX_TOKENS_DEFAULT = 4096


def main(argv=sys.argv):
    Settings.embed_model = OpenAIEmbedding(
        model=MODEL_EMBEDDING,
        embed_batch_size=EMBED_BATCH_SIZE)
    Settings.llm=OpenAI(
        model=LLM_GPT_4_MINI,
        temperature=TEMPERATURE_DEFAULT,
        max_tokens=MAX_TOKENS_DEFAULT)

    print("Parsing files...")

    try:
        with open(BASE_DIR / "documents.pickle", "rb") as fp:
            documents = pickle.load(fp)

    except FileNotFoundError:
        documents = []

    docs_to_remove = []

    for doc in documents:
        file_path = DATABASE_PATH / doc.metadata["file_path"]

        if not file_path.exists():
            docs_to_remove.append(doc)

    doc_ids_to_remove = set()

    for doc in docs_to_remove:
        doc_ids_to_remove.add(doc.id_)

    documents = [doc for doc in documents
                 if doc.id_ not in doc_ids_to_remove]

    files_to_remove = set()

    for doc in docs_to_remove:
        files_to_remove.add(doc.metadata["file_path"])

    print("Files removed:", len(files_to_remove))

    # TODO: check file modification date

    files_to_exclude = set()

    for doc in documents:
        file_path = DATABASE_PATH / doc.metadata["file_path"]
        file_path = file_path.absolute()
        files_to_exclude.add(str(file_path))

    files_to_exclude = list(files_to_exclude)

    print("Files to exclude:", len(files_to_exclude))

    parser_ppt = LlamaParse(
        result_type="markdown",
        # language="ru",  # "en" or "ru""
        parsing_instruction="""\
You are parsing a slide of the presentation developed by a \
management consultant. DO extract all text from the slide. \
DO NOT skip or omit any text. Do you best to understand charts \
and diagrams and translate them to markdown tables.\
""",
        page_separator="\n\n---\n\n",
        # page_prefix="\n",
        page_suffix="\n\nSlide number: {pageNumber}\n",
        verbose=False)

    parser_doc = LlamaParse(
        result_type="markdown",
        # language="ru",  # "en" or "ru""
        # parsing_instruction="",
        # page_separator="\n\n---\n\n",
        # page_prefix="\n",
        # page_suffix="\n\nPage: {pageNumber}\n",
        verbose=False)

    if PRIMARY_FORMAT == "ppt": 
        parser_pdf = parser_ppt
    else:
        parser_pdf = parser_doc

    file_exts = [f".{x}" for x in FILE_EXTENSIONS.split(",")]

    try:
        new_documents = SimpleDirectoryReader(
            input_dir=str(DATABASE_PATH),
            recursive=True,
            required_exts=file_exts,
            file_extractor={".pdf":  parser_pdf,
                            ".ppt":  parser_ppt,
                            ".pptx": parser_ppt,
                            ".txt":  parser_doc,
                            ".doc":  parser_doc,
                            ".docx": parser_doc},
            exclude=files_to_exclude,
        ).load_data(show_progress=True)  # num_workers=4

    except ValueError as e:
        print(e)  # ValueError: No files found in ...
        new_documents = []

    for doc in new_documents:
        file_path = Path(doc.metadata["file_path"])
        doc.metadata["file_path"] \
            = str(file_path.relative_to(DATABASE_PATH)).replace("\\", "/")
        doc.excluded_llm_metadata_keys.remove("file_name")

    new_files = set()

    for doc in new_documents:
        new_files.add(doc.metadata["file_path"])

    print("New files:", len(new_files))

    documents.extend(new_documents)

    with open(BASE_DIR / "documents.pickle", "wb") as fp:
        pickle.dump(documents, fp)

    all_files = set()

    for doc in documents:
        all_files.add(doc.metadata["file_path"])

    print("Total files parsed:", len(all_files))

    print("Indexing documents...")

    new_index = not (BASE_DIR / "chroma_db").exists()

    chroma_db = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
    chroma_collection = chroma_db.get_or_create_collection("wisdom")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if new_index:
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True)

        print("Done!")

        return

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context)

    print("Remove index for deleted files")

    for doc_id in tqdm(doc_ids_to_remove):
        index.delete_ref_doc(doc_id, delete_from_docstore=True)

    print("Add index for new files")

    for doc in tqdm(new_documents):
        index.insert(doc)

    print("Done!")


if __name__ == "__main__":
    sys.exit(main())
