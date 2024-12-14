#!/usr/bin/env python3

import sys
import os
import hashlib
from pathlib import Path
import pickle

from typing import List

from dotenv import load_dotenv

from llama_index.core import Settings

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.cohere import Cohere
import llama_index.llms.cohere.utils as cohere_utils

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

from llama_index.core.types import BaseOutputParser

from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from llama_index.postprocessor.cohere_rerank import CohereRerank

import chromadb

import Stemmer

import gradio as gr


################################################################################


load_dotenv()


################################################################################


BASE_DIR = Path(__file__).parent.absolute()
DATABASE_PATH = Path(os.environ["DATABASE_PATH"])

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

PASSWORD_SALT = os.environ["PASSWORD_SALT"]

LLM_GPT_4 = "gpt-4o-2024-11-20"
LLM_CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
LLM_COHERE_COMMAND_R_PLUS = "command-r-plus-08-2024"
RERANK_COHERE = "rerank-v3.5"

MODEL_EMBEDDING = "text-embedding-3-large"
EMBED_MODEL_DIM = 3072
EMBED_BATCH_SIZE = 256

LLM_ENHANCE_QUERY = LLM_CLAUDE_SONNET

TEMPERATURE_DEFAULT = 0.0
MAX_TOKENS_DEFAULT = 4096

MOCK_LLM_MAX_TOKENS = 50

GRADIO_CONCURRENCY_LIMIT = 5

GRADIO_ROOT_PATH = os.environ.get("GRADIO_ROOT_PATH", "/wisdom")

RESTRICTED_FILE_PATH = os.environ["RESTRICTED_FILE_PATH"]

### DEFAULTS ###

DEFAULT_SIM_TOP_K = int(os.environ["DEFAULT_SIM_TOP_K"])
DEFAULT_RERANK_TOP_N = int(os.environ["DEFAULT_RERANK_TOP_N"])
DEFAULT_ENHANCE_QUERY = bool(int(os.environ["DEFAULT_ENHANCE_QUERY"]))

DEFAULT_LLM_PROVIDER = os.environ["DEFAULT_LLM_PROVIDER"]
DEFAULT_LLM_MODEL = {
    "openai": LLM_GPT_4,
    "anthropic": LLM_CLAUDE_SONNET,
    "cohere": LLM_COHERE_COMMAND_R_PLUS,
}[DEFAULT_LLM_PROVIDER]


################################################################################


def _path_cohere_utils():
    cohere_utils.COMMAND_MODELS[LLM_COHERE_COMMAND_R_PLUS] = 128_000
    cohere_utils.ALL_AVAILABLE_MODELS[LLM_COHERE_COMMAND_R_PLUS] = 128_000
    cohere_utils.CHAT_MODELS[LLM_COHERE_COMMAND_R_PLUS] = 128_000
    cohere_utils.FUNCTION_CALLING_MODELS.add(LLM_COHERE_COMMAND_R_PLUS)


_path_cohere_utils()


################################################################################


def load_prompt(type_, role, lang):
    file_path = BASE_DIR / "prompts" / f"{type_}_{role}_{lang}.txt"

    with open(file_path, encoding="utf-8") as fp:
        return fp.read()


class OutputParser(BaseOutputParser):

    def parse(self, output):
        for tag in ("refined_answer",
                    "уточненный_ответ",
                    "response",
                    "ответ",
                    "enhanced_query",
                    "улучшенный_запрос"):
            if f"<{tag}>" not in output:
                continue

            orig_output = output

            if f"</{tag}>" in output:
                _, output = output.split(f"<{tag}>", 2)
                output, _ = output.split(f"</{tag}>", 2)
                output = output.strip()

            else:
                _, output = output.split(f"<{tag}>", 2)
                output = output.strip()

            if not output:
                output = orig_output
                continue

        return output


class NonExistingFilesFilterPostprocessor(BaseNodePostprocessor):
    @classmethod
    def class_name(cls) -> str:
        return "NonExistingFilesFilterPostprocessor"

    def _postprocess_nodes(self, nodes, query_bundle=None):
        new_nodes = []

        for node in nodes:
            path = DATABASE_PATH / node.node.metadata["file_path"]

            if path.exists():
                new_nodes.append(node)

        return new_nodes


class FilePathFilterPostprocessor(BaseNodePostprocessor):
    ignore_file_paths: List[str] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "FilePathFilterPostprocessor"

    def _postprocess_nodes(self, nodes, query_bundle=None):
        new_nodes = []

        for node in nodes:
            if node.node.metadata["file_path"] \
                    not in self.ignore_file_paths:
                new_nodes.append(node)

        return new_nodes


async def index_query(index,
                      nodes,
                      query,
                      instructions,
                      query_enhance,
                      similarity_top_k,
                      rerank_top_n,
                      model,
                      exclude_files_str,
                      include_restricted):
    ### Checks ###

    if not query:
        return ""

    if similarity_top_k is None:
        raise gr.Error("'Index similarity top_k' should be greater than 0")

    similarity_top_k = int(similarity_top_k)

    if rerank_top_n is None:
        raise gr.Error("'Index similarity top_k' should be greater than 0")

    rerank_top_n = int(rerank_top_n)

    if not model:
        raise gr.Error("'Model' cannot be empty")

    ### Parameters ###

    language = "en"

    abc_ru = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    abc_ru = set(abc_ru) | set(abc_ru.upper())

    for c in query:
        if c in abc_ru:
            language = "ru"
            break

    ### Prepare templates ###

    text_qa_template = ChatPromptTemplate([
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=load_prompt("text_qa", "system", language),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=load_prompt("text_qa", "user", language))])

    text_qa_template = text_qa_template.partial_format(
        instructions=instructions)

    refine_template = ChatPromptTemplate([
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=load_prompt("refine", "system", language)),
        ChatMessage(
            role=MessageRole.USER,
            content=load_prompt("refine", "user", language))])

    refine_template = refine_template.partial_format(
        instructions=instructions)

    q_enhance_template = ChatPromptTemplate([
        ChatMessage(role=MessageRole.SYSTEM,
                    content=load_prompt("q_enhance", "system", language)),
        ChatMessage(role=MessageRole.USER,
                    content=load_prompt("q_enhance", "user", language))])

    ### Query enhance ###

    enhanced_query = None

    if query_enhance:
        llm = OpenAI(model=LLM_ENHANCE_QUERY,
                     temperature=TEMPERATURE_DEFAULT,
                     max_tokens=MAX_TOKENS_DEFAULT,
                     output_parser=OutputParser())

        enhanced_query = await llm.apredict(
            q_enhance_template,
            query_str=query)
        query = enhanced_query

    ### Query ###

    cohere_rerank = CohereRerank(model=RERANK_COHERE,
                                 top_n=rerank_top_n)

    non_existing_files_filter = NonExistingFilesFilterPostprocessor()

    ignore_file_paths = []
    ignore_file_paths.extend(exclude_files_str.split("\n"))

    if not include_restricted:
        with open(RESTRICTED_FILE_PATH, encoding="utf-8") as fp:
            ignore_file_paths.extend(fp.read().strip().split("\n"))

    ignore_file_paths = [x.strip() for x in ignore_file_paths]
    ignore_file_paths = [x for x in ignore_file_paths
                         if x and not x.startswith("#")]
    ignore_file_paths = [x.replace("\\", "/") for x in ignore_file_paths]

    file_path_filter = FilePathFilterPostprocessor(
        ignore_file_paths=ignore_file_paths)

    language_full = {
        "ru": "russian",
        "en": "english",
    }[language]

    vector_retriever = index.as_retriever(
        similarity_top_k=similarity_top_k,
        embed_model=OpenAIEmbedding(
            model=MODEL_EMBEDDING,
            embed_batch_size=EMBED_BATCH_SIZE))

    bm25_retriever = BM25Retriever(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer(language_full),
        language=language_full)

    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=1,
        mode="relative_score")

    # Anthropic
    if model.startswith("claude-"):
        llm = Anthropic(
            model=model,
            temperature=TEMPERATURE_DEFAULT,
            max_tokens=MAX_TOKENS_DEFAULT,
            output_parser=OutputParser())

    # Cohere
    elif model.startswith("command-"):
        llm = Cohere(
            model=model,
            temperature=TEMPERATURE_DEFAULT,
            max_tokens=MAX_TOKENS_DEFAULT,
            output_parser=OutputParser())

    # OpenAI: gpt-4o
    elif model.startswith("gpt-"):
        llm = OpenAI(
            model=model,
            temperature=TEMPERATURE_DEFAULT,
            max_tokens=MAX_TOKENS_DEFAULT,
            output_parser=OutputParser())

    else:
        raise ValueError(model)

    synthesizer = CompactAndRefine(
        llm=llm,
        text_qa_template=text_qa_template,
        refine_template=refine_template)

    query_engine = RetrieverQueryEngine(
        retriever=fusion_retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[non_existing_files_filter,
                             file_path_filter,
                             cohere_rerank])

    response_obj = await query_engine.aquery(query)
    response = response_obj.response

    ### Mentioned files ###

    mentioned_files = set()

    if response_obj is not None:
        for node in response_obj.source_nodes:
            if node.node.metadata["file_name"] in response:
                mentioned_files.add(node.node.metadata["file_path"])

    mentioned_files_table = [[x] for x in mentioned_files]
    mentioned_files_table.sort()

    mentioned_files_str = "\n".join(sorted(mentioned_files))

    ### End ###

    return (response,
            mentioned_files_table,
            mentioned_files_str,
            None,
            enhanced_query)


def auth(username, password):
    users = {}

    with open(BASE_DIR / "users.txt", "r", encoding="utf-8") as fp:
        for line in fp:
            uname, hash_ = line.split(":", 2)
            users[uname] = hash_.rstrip()

    if username not in users:
        return False

    hash1 = users[username]

    hash2 = hashlib.sha256(
        (password + PASSWORD_SALT).encode("utf-8")).hexdigest()

    return hash1 == hash2


################################################################################


def main(argv=sys.argv):
    # Settings.embed_model = OpenAIEmbedding(
    #     model=MODEL_EMBEDDING,
    #     embed_batch_size=EMBED_BATCH_SIZE)

    chroma_db = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
    chroma_collection = chroma_db.get_or_create_collection("wisdom")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context)

    with open(BASE_DIR / "documents.pickle", "rb") as fp:
        nodes = pickle.load(fp)
    
    ############################################################################

    with gr.Blocks() as demo:
        with gr.Tab("General"):
            with gr.Row():
                with gr.Column():  # Request and response
                    with gr.Row():
                        out_response = gr.TextArea(
                            label="AI response")

                    in_query = gr.Textbox(
                        label="User's query")

                    in_q_enhance = gr.Checkbox(
                        label=("Optimize query (may increase the number of "
                               "documents but decrease relevance)"),
                        value=DEFAULT_ENHANCE_QUERY)

                    in_instructions = gr.Textbox(
                        label="Additional instructions",
                        value="Provide at least 10 presentations.")

                    in_model = gr.Dropdown(
                        label="Model",
                        choices=[("GPT-4o",        LLM_GPT_4),
                                 ("Claude Sonnet", LLM_CLAUDE_SONNET),
                                 ("Command R+",    LLM_COHERE_COMMAND_R_PLUS)],
                        value=DEFAULT_LLM_MODEL)

                    btn_submit = gr.Button("Submit")

                with gr.Column():  # Documents
                    out_docs = gr.Dataframe(
                        # label="Documents",
                        headers=["Path"],
                        datatype=["str"])

                    out_file = gr.File(label="Download")

            with gr.Accordion("Examples", open=False):
                gr.Examples(
                    examples=[
                        ["Find presentations about safety stock management.",
                         "Provide at least 10 presentations."],
                        ["Найди материалы по стратегическим фреймворкам.",
                         "Приведи не менее 10 презентаций."],
                        ["What metrics and KPIs do exist in logistics?",
                         ""]],
                    inputs=[in_query, in_instructions])

        with gr.Tab("Filter"):
            out_file_paths = gr.TextArea(
                label="Found files (last query)")

            in_filter_file_paths = gr.TextArea(
                label="Exclude files (next query)")

            with gr.Row():
                btn_filter_copy = gr.Button("Copy to ignore")
                btn_filter_clear = gr.Button("Clear")

        with gr.Tab("Options"):
            with gr.Group():
                in_similarity_top_k = gr.Number(
                    label="Index similarity top_k",
                    minimum=1,
                    maximum=300,
                    value=DEFAULT_SIM_TOP_K)

                in_rerank_top_n = gr.Number(
                    label="Index rerank top_n",
                    minimum=1,
                    maximum=50,
                    value=DEFAULT_RERANK_TOP_N)

        with gr.Tab("Debug"):
            out_enhanced_prompt = gr.TextArea(
                label="Enhanced prompt")

            in_include_restricted = gr.Checkbox(
                label="Option 2",
                value=False)

        ########################################################################

        async def fn_submit(*args):
            return await index_query(index, nodes, *args)

        btn_submit.click(
            fn=fn_submit,
            inputs=[in_query,
                    in_instructions,
                    in_q_enhance,
                    in_similarity_top_k,
                    in_rerank_top_n,
                    in_model,
                    in_filter_file_paths,
                    in_include_restricted],
            outputs=[out_response,
                     out_docs,
                     out_file_paths,
                     out_file,
                     out_enhanced_prompt])

        def fn_out_docs_select_callback(evt: gr.SelectData):
            return str(DATABASE_PATH / evt.value.replace("\\", "/"))

        out_docs.select(
            fn=fn_out_docs_select_callback,
            outputs=[out_file])

        def fn_filter_copy(in_filter_file_paths, out_file_paths):
            lst = in_filter_file_paths.split("\n")
            lst.extend(out_file_paths.split("\n"))
            lst = [x.strip() for x in lst]
            lst = [x for x in lst if x]
            lst = list(set(lst))
            lst.sort()
            return "\n".join(lst)

        btn_filter_copy.click(
            fn=fn_filter_copy,
            inputs=[in_filter_file_paths,
                    out_file_paths],
            outputs=[in_filter_file_paths])

        def fn_filter_clear():
            return ""

        btn_filter_clear.click(
            fn=fn_filter_clear,
            inputs=[],
            outputs=[in_filter_file_paths])

    ############################################################################

    demo.queue(default_concurrency_limit=GRADIO_CONCURRENCY_LIMIT)

    demo.launch(allowed_paths=[DATABASE_PATH],
                root_path=GRADIO_ROOT_PATH,
                auth=auth)


if __name__ == "__main__":
    sys.exit(main())
