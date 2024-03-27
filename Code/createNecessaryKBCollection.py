import os
import dotenv
import pandas as pd
import time
from langchain_openai import ChatOpenAI


dotenv.load_dotenv()

chat = ChatOpenAI(
    openai_api_key= os.getenv("OPENAI_API_KEY"),
    model='gpt-3.5-turbo'
)

kbWhole = pd.read_csv('CSVs/KB_NecessaryDocs.csv')

import chromadb
# from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

client = chromadb.PersistentClient(path="Collections/")
collection = client.create_collection("NecessaryKB")

data = kbWhole

batch_size = 100

from tqdm.auto import tqdm  # for progress bar

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    # get batch of data
    batch = data.iloc[i:i_end]
    # generate unique ids for each chunk
    ids_b = [str(i) for i, x in batch.iterrows()]
    # get text to embed
    texts = [x['Chunk'] for _, x in batch.iterrows()]
    # embed text
    embeds = embed_model.embed_documents(texts)
    metadata_b = [
        {'source': x['Link'],
         'title': x['Title']} for i, x in batch.iterrows()
    ]

    collection.add(
        embeddings = embeds,
        ids = ids_b,
        documents = texts,
        metadatas = metadata_b
    )