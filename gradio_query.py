#!/usr/bin/env python3

import sys
import os
from functools import partial
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


def create_prompts(lang):
    text_qa_template = ChatPromptTemplate([
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=load_prompt("text_qa", "system", lang),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=load_prompt("text_qa", "user", lang))])

    def existing_answer_extract(**kwargs):
        value = kwargs["existing_answer"]

        for tag in ("refined_answer", "уточненный_ответ"):  # FIXME
            if f"<{tag}>" not in value:
                continue

            orig_value = value

            if f"</{tag}>" in value:
                _, value = value.split(f"<{tag}>", 2)
                value, _ = value.split(f"</{tag}>", 2)
                value = value.strip()

            else:
                _, value = value.split(f"<{tag}>", 2)
                value = value.strip()

            if not value:
                value = orig_value

            break

        return value

    refine_template = ChatPromptTemplate([
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=load_prompt("refine", "system", lang),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=load_prompt("refine", "user", lang))],
        function_mappings={"existing_answer": existing_answer_extract})

    return text_qa_template, refine_template


def detect_language(prompt):
    abc_ru = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    abc_ru = set(abc_ru) | set(abc_ru.upper())

    for c in prompt:
        if c in abc_ru:
            return "ru"

    return "en"

################################################################################


def calculate_cost(model, token_counter):
    # https://openai.com/api/pricing/
    # https://www.anthropic.com/pricing#anthropic-api
    pricing = {"gpt-4o-2024-08-06":          [2.500, 10.000],
               "gpt-4o-mini-2024-07-18":     [0.150,  0.600],
               "claude-3-haiku-20240307":    [0.250,  1.250],
               "claude-3-5-sonnet-20240620": [3.000, 15.000]}

    pricing_embedding = 0.130  # text-embedding-3-large

    return token_counter.total_embedding_token_count * pricing_embedding \
        / 1_000_000 + \
        token_counter.prompt_llm_token_count * pricing[model][0] / 1_000_000 + \
        token_counter.completion_llm_token_count * pricing[model][1] / 1_000_000


def index_query(index, prompt, similarity_top_k, model):
    if not prompt:
        return ""

    if similarity_top_k is None:
        raise gr.Error("'Index similarity top_k' should be greater than 0")

    if not model:
        raise gr.Error("'Model' cannot be empty")

    text_qa_template, refine_template \
        = create_prompts(detect_language(prompt))

    def get_query_engine(mock=False):
        if model.startswith("claude-"):
            if mock:
                llm = MockLLM(max_tokens=512)
                embed_model = MockEmbedding(embed_dim=3072)

            else:
                llm = Anthropic(
                    model=model,
                    temperature=0,
                    max_tokens=4096)
                embed_model = OpenAIEmbedding(
                    model="text-embedding-3-large",
                    embed_batch_size=256)

            query_engine = index.as_query_engine(
                similarity_top_k=int(similarity_top_k),
                llm=llm,
                embed_model=embed_model,
                text_qa_template=text_qa_template,
                refine_template=refine_template,
                tokenizer=Anthropic().tokenizer)

            token_counter = TokenCountingHandler(
                tokenizer=Anthropic().tokenizer.encode)
            query_engine.callback_manager.add_handler(token_counter)

        elif model.startswith("gpt-"):
            if mock:
                llm = MockLLM(max_tokens=512)
                embed_model = MockEmbedding(embed_dim=3072)

            else:
                llm = OpenAI(
                    model=model,
                    temperature=0,
                    max_tokens=4096)
                embed_model = OpenAIEmbedding(
                    model="text-embedding-3-large",
                    embed_batch_size=256)

            query_engine = index.as_query_engine(
                similarity_top_k=int(similarity_top_k),
                llm=llm,
                embed_model=embed_model,
                text_qa_template=text_qa_template,
                refine_template=refine_template)

            token_counter = TokenCountingHandler(
                tokenizer=tiktoken.encoding_for_model(model).encode)
            query_engine.callback_manager.add_handler(token_counter)

        else:
            raise ValueError(model)

        return query_engine, token_counter

    query_engine, token_counter = get_query_engine(mock=True)
    query_engine.query(prompt)

    estimated_cost = calculate_cost(model, token_counter)

    gr.Info(f"Estimated cost is ${estimated_cost:.4f}")

    if estimated_cost >= COST_THRESHOLD:
        raise gr.Error((f"""\
Estimated cost of ${estimated_cost:.2f} is above the allowed \
threshold ${COST_THRESHOLD:.2f}. Try cheaper models (claude-3-haiku \
or gpt-4o-mini) or decrease index similarity top_k parameter.\
"""))

    query_engine, token_counter = get_query_engine(mock=False)

    try:
        response_obj = query_engine.query(prompt)
    except Exception as e:
        real_cost = calculate_cost(model, token_counter)
        return f"LlamaIndex Error:\n\n{e}", [], real_cost

    response = response_obj.response

    for tag in ("answer", "ответ"):  # FIXME
        if f"<{tag}>" not in response:
            continue

        orig_response = response

        if f"</{tag}>" in response:
            _, response = response.split(f"<{tag}>", 2)
            response, _ = response.split(f"</{tag}>", 2)
            response = response.strip()

        else:
            _, response = response.split(f"<{tag}>", 2)
            response = response.strip()

        if not response:
            response = orig_response

        break

    mentioned_docs = set()

    for node in response_obj.source_nodes:
        if node.node.metadata["file_name"] in response:
            mentioned_docs.add(node.node.metadata["file_path"])

    mentioned_docs = [[x] for x in mentioned_docs]
    mentioned_docs.sort()

    real_cost = calculate_cost(model, token_counter)

    return response, mentioned_docs, real_cost


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

                    in_request = gr.Textbox(
                        label="Request prompt")

                    btn_submit = gr.Button("Submit")

                    gr.Examples(
                        examples=[
                            "Find presentations about safety stock management.",
                            "Find presentations about network modeling."],
                        inputs=[in_request],
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
                value="claude-3-haiku-20240307")

        btn_submit.click(
            fn=partial(index_query, index),
            inputs=[in_request, in_top_k, in_model],
            outputs=[out_response, out_docs, out_cost])

        def out_docs_select_callback(evt: gr.SelectData):
            return str(DATABASE_PATH / evt.value.replace("\\", "/"))

        out_docs.select(
            fn=out_docs_select_callback,
            outputs=[out_file])

    ############################################################################

    demo.launch(root_path="/wisdom",
                auth=[(os.environ["GRADIO_AUTH_USER"],
                       os.environ["GRADIO_AUTH_PASS"])])


if __name__ == "__main__":
    sys.exit(main())

