import os
import shutil
import gradio as gr
from pdf_chunk import pdf2chunk
from chunk_embed import chunk2prompts
from model_response import prompts2response


EMBEDDING = 'openai'  # openai (dimension:1536) or huggingface (dimension:384)
MD = 'gpt'
CHROMA_PATH = './chroma'
DATA_PATH = './data'
PROMPT_TEMPLATE = """
You are a helpful {occupation} and will answer the question: {query} based on your existing knowledge and useful information from {context}.
"""

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)  # clear all stuff, make sure the UI supports multiple questions

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)  # clear all stuff
os.mkdir(DATA_PATH)


def arabica(file_list, occupation, query):
    chunks = pdf2chunk(file_list, DATA_PATH)
    prompts = chunk2prompts(chunks, occupation, query, EMBEDDING, CHROMA_PATH, PROMPT_TEMPLATE)
    response_text = prompts2response(prompts, MD)
    return response_text


# query = "Can you describe the structure of llama?"
with gr.Blocks() as demo:
    gr.Markdown('# Arabicabot')
    gr.Interface(fn=arabica, inputs=[gr.File(file_count='multiple', file_types=['.pdf']), gr.Text(), gr.Text()], outputs=[gr.Text()])
    demo.launch()
