import os
import shutil
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def pdf2chunk(file_list, DATA_PATH):
    for file in file_list:
        print(file)
        base_name = os.path.basename(file)
        new_path = os.path.join('./data', base_name)
        shutil.copy2(file, new_path)
    
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks