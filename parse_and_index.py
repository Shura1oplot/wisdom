#!/usr/bin/env python3

import sys
import os
import pickle
from pathlib import Path
from textwrap import dedent

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


def main(argv=sys.argv):
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        embed_batch_size=256)
    Settings.llm=OpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=4096)

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

    parser = LlamaParse(
        result_type="markdown",
        # language="ru",  # "en" or "ru""
        parsing_instruction=dedent("""
            You are parsing a slide of the presentation developed by a \
            management consultant. DO extract all text from the slide. \
            DO NOT skip or omit any text. Do you best to understand charts \
            and diagrams and translate them to markdown tables.
        """.strip()),
        # page_separator="\n---\n",
        # page_prefix="\n",
        page_suffix="\n\nSlide: {pageNumber}",
        verbose=False)

    try:
        new_documents = SimpleDirectoryReader(
            input_dir=str(DATABASE_PATH),
            recursive=True,
            required_exts=[".pdf", ".ppt", ".pptx"],
            file_extractor={".pdf": parser,
                            ".ppt": parser,
                            ".pptx": parser},
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

