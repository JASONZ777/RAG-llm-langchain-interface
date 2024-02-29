from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import os


def chunk2prompts(chunks, occupation, query, embedding, CHROMA_PATH, PROMPT_TEMPLATE):
    # acquire the db, two possible way to embed chunks
    if embedding == 'openai':
        database = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
        database.persist()  # store in the hard disk
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    elif embedding == 'huggingface':
        emb_model = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_method = HuggingFaceEmbeddings(
            model_name=emb_model,
            cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
        )
        database = Chroma.from_documents(chunks, embedding_method, persist_directory=CHROMA_PATH)
        database.persist()  # store in the hard disk
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_method)

    matches = db.similarity_search_with_relevance_scores(query, k=5)
    if len(matches) == 0 or matches[0][1] < 0.7:
        print(f"No matching results.")
        prompt_tem = PromptTemplate.from_template(query)
        prompts = prompt_tem.format()
    else:
        context = "\n\--\n".join([doc.page_content for doc, _score in matches])
        prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
        prompts = prompt_template.format(occupation=occupation, context=context, query=query)

    return prompts