#!/usr/bin/env python3

import sys
import os
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import Settings

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

from llama_index.core import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.callbacks import TokenCountingHandler

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

from llama_index.core.types import BaseOutputParser

from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from llama_index.postprocessor.cohere_rerank import CohereRerank

import chromadb

import tiktoken

import gradio as gr


################################################################################


load_dotenv()


################################################################################


BASE_DIR = Path(__file__).parent.absolute()
DATABASE_PATH = Path(os.environ["DATABASE_PATH"])

COST_THRESHOLD = float(os.environ["COST_THRESHOLD"])


################################################################################


def load_prompt(type_, role, lang):
    file_path = BASE_DIR / "prompts" / f"{type_}_{role}_{lang}.txt"

    with open(file_path, encoding="utf-8") as fp:
        return fp.read()


def calculate_cost(model, token_counter):
    # https://openai.com/api/pricing/
    # https://www.anthropic.com/pricing#anthropic-api
    pricing = {"gpt-4o-2024-08-06":          [2.500, 10.000],
               "gpt-4o-mini-2024-07-18":     [0.150,  0.600],
               "claude-3-haiku-20240307":    [0.250,  1.250],
               "claude-3-5-sonnet-20240620": [3.000, 15.000]}

    pricing_embedding = 0.130  # text-embedding-3-large

    # https://cohere.com/pricing
    cohere_cost = 2 / 1000

    return token_counter.total_embedding_token_count * pricing_embedding \
        / 1_000_000 + \
        token_counter.prompt_llm_token_count * pricing[model][0] \
        / 1_000_000 + \
        token_counter.completion_llm_token_count * pricing[model][1] \
        / 1_000_000 + \
        cohere_cost


class OutputParser(BaseOutputParser):

    def parse(self, output):
        for tag in ("refined_answer", "уточненный_ответ", "answer", "ответ"):
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


class FilePathFilterPostprocessor(BaseNodePostprocessor):
    ignore_file_paths = Field(default_factory=list)

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
                      question,
                      extra_instructions,
                      similarity_top_k,
                      model,
                      exclude_files_str):
    if not question:
        return ""

    if similarity_top_k is None:
        raise gr.Error("'Index similarity top_k' should be greater than 0")

    similarity_top_k = int(similarity_top_k)

    if not model:
        raise gr.Error("'Model' cannot be empty")

    language = "en"

    abc_ru = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    abc_ru = set(abc_ru) | set(abc_ru.upper())

    for c in question:
        if c in abc_ru:
            language = "ru"
            break

    text_qa_template = ChatPromptTemplate([
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=load_prompt("text_qa", "system", language),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=load_prompt("text_qa", "user", language))])

    text_qa_template = text_qa_template.partial_format(
        extra_instructions=extra_instructions)

    refine_template = ChatPromptTemplate([
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=load_prompt("refine", "system", language)),
        ChatMessage(
            role=MessageRole.USER,
            content=load_prompt("refine", "user", language))])

    refine_template = refine_template.partial_format(
        extra_instructions=extra_instructions)

    if model.startswith("claude-"):
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            llm=MockLLM(max_tokens=512),
            embed_model=MockEmbedding(embed_dim=3072),
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            tokenizer=Anthropic().tokenizer)

        token_counter = TokenCountingHandler(
            tokenizer=Anthropic().tokenizer.encode)
        query_engine.callback_manager.add_handler(token_counter)

    elif model.startswith("gpt-"):
        query_engine = index.as_query_engine(
            similarity_top_k=int(similarity_top_k),
            llm=MockLLM(max_tokens=512),
            embed_model=MockEmbedding(embed_dim=3072),
            text_qa_template=text_qa_template,
            refine_template=refine_template)

        token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model(model).encode)
        query_engine.callback_manager.add_handler(token_counter)

    else:
        raise ValueError(model)

    await query_engine.aquery(question)

    estimated_cost = calculate_cost(model, token_counter)

    gr.Info(f"Estimated cost is ${estimated_cost:.4f}")

    if estimated_cost >= COST_THRESHOLD:
        raise gr.Error((f"""\
Estimated cost of ${estimated_cost:.2f} is above the allowed \
threshold ${COST_THRESHOLD:.2f}. Try cheaper models (claude-3-haiku \
or gpt-4o-mini) or decrease index similarity top_k parameter.\
"""))

    if language == "en":
        cohere_model = "rerank-english-v3.0"
    else:
        cohere_model = "rerank-multilingual-v3.0"

    cohere_rerank = CohereRerank(model=cohere_model,
                                 top_n=similarity_top_k)

    file_path_filter = FilePathFilterPostprocessor(
        ignore_file_paths=exclude_files_str.split("\n"))

    if model.startswith("claude-"):
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            llm=Anthropic(
                model=model,
                temperature=0,
                max_tokens=4096,
                output_parser=OutputParser()),
            embed_model=OpenAIEmbedding(
                model="text-embedding-3-large",
                embed_batch_size=256),
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            tokenizer=Anthropic().tokenizer,
            node_postprocessors=[file_path_filter,
                                 cohere_rerank])

        token_counter = TokenCountingHandler(
            tokenizer=Anthropic().tokenizer.encode)
        query_engine.callback_manager.add_handler(token_counter)

    elif model.startswith("gpt-"):
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            llm=OpenAI(
                model=model,
                temperature=0,
                max_tokens=4096,
                output_parser=OutputParser()),
            embed_model=OpenAIEmbedding(
                model="text-embedding-3-large",
                embed_batch_size=256),
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            node_postprocessors=[file_path_filter,
                                 cohere_rerank])

        token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model(model).encode)
        query_engine.callback_manager.add_handler(token_counter)

    else:
        raise ValueError(model)

    try:
        response_obj = await query_engine.aquery(question)
    except Exception as e:
        real_cost = calculate_cost(model, token_counter)
        return f"LlamaIndex Error:\n\n{e}", [], real_cost

    response = response_obj.response

    mentioned_files = set()

    for node in response_obj.source_nodes:
        if node.node.metadata["file_name"] in response:
            mentioned_files.add(node.node.metadata["file_path"])

    mentioned_files_table = [[x] for x in mentioned_files]
    mentioned_files_table.sort()

    mentioned_files_str = "\n".join(sorted(mentioned_files))

    real_cost = calculate_cost(model, token_counter)

    return (response,
            mentioned_files_table,
            mentioned_files_str,
            real_cost,
            None)


################################################################################


def main(argv=sys.argv):
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        embed_batch_size=256)

    chroma_db = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
    chroma_collection = chroma_db.get_or_create_collection("wisdom")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context)

    ############################################################################

    with gr.Blocks() as demo:
        with gr.Tab("General"):
            with gr.Row():
                with gr.Column():  # Request and response
                    out_response = gr.TextArea(
                        label="LLM response")

                    in_question = gr.Textbox(
                        label="Question")

                    in_instructions = gr.Textbox(
                        label="Additional instructions")

                    btn_submit = gr.Button("Submit")

                    gr.Examples(
                        examples=[
                            ["Find presentations about safety stock management.",
                             "Provide at least 20 presentations."],
                            ["Найди материалы по стратегическим фреймворкам.",
                             "Приведи не менее 10 презентаций."]],
                        inputs=[in_question, in_instructions],
                    )

                with gr.Column():  # Documents
                    out_cost = gr.Number(label="Cost charged, USD")

                    out_docs = gr.Dataframe(
                        # label="Documents",
                        headers=["Path"],
                        datatype=["str"])

                    out_file = gr.File(label="Download")

        with gr.Tab("Options"):
            in_top_k = gr.Number(
                label="Index similarity top_k",
                info="""\
How many chunks of the documents should be retrieved from the index and passed \
to the LLM? Fewer chunks mean a cheaper response with less variants. Start \
with 10 or 100 and then adjust to your needs.\
""",
                minimum=1,
                maximum=1000,
                value=100)

            in_model = gr.Dropdown(
                label="Model",
                info="""\
Try different models. Start with claude-3-haiku. Pay attention to costs charged. \
It is recommended to use gpt-4o-mini or claude-3-haiku with top_k above 100.\
""",
               choices=["gpt-4o-mini-2024-07-18",
                        "gpt-4o-2024-08-06",
                        "claude-3-haiku-20240307",
                        "claude-3-5-sonnet-20240620"],
                value=os.environ.get("DEFAULT_LLM", "gpt-4o-mini-2024-07-18"))

        with gr.Tab("Filter"):
            in_filter_file_paths = gr.TextArea(
                label="Exclude file paths")

            out_file_paths = gr.TextArea(
                label="Exclude file paths")

        async def fn(*args):
            return await index_query(index, *args)

        btn_submit.click(
            fn=fn,
            inputs=[in_question,
                    in_instructions,
                    in_top_k,
                    in_model,
                    in_filter_file_paths],
            outputs=[out_response,
                     out_docs,
                     out_file_paths,
                     out_cost,
                     out_file])

        def out_docs_select_callback(evt: gr.SelectData):
            return str(DATABASE_PATH / evt.value.replace("\\", "/"))

        out_docs.select(
            fn=out_docs_select_callback,
            outputs=[out_file])

    ############################################################################

    demo.queue(default_concurrency_limit=20)
    demo.launch(root_path="/wisdom",
                auth=[(os.environ["GRADIO_AUTH_USER"],
                       os.environ["GRADIO_AUTH_PASS"])])


if __name__ == "__main__":
    sys.exit(main())
