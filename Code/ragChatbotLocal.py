import os
import dotenv
import pandas as pd
import time
from langchain_openai import ChatOpenAI
# from pinecone import Pinecone
# from pinecone import ServerlessSpec


dotenv.load_dotenv()

chat = ChatOpenAI(
    openai_api_key= os.getenv("OPENAI_API_KEY"),
    model='gpt-3.5-turbo'
)

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

kbWhole = pd.read_csv('CSVs/KB_Whole_FINAL.csv')

import chromadb
# from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
messages = []

client = chromadb.Client()
collection = client.create_collection("WholeKB")

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

#print(collection.peek(1))
    
query = "What classes fulfill Integrative Experience requirement for CS majors?"

def searchDataByVector(query: str):
    try:
        query_vector = embed_model.embed_query(query)
        res = collection.query(
            query_embeddings=[query_vector],
            n_results=1,
            include=['distances','embeddings', 'documents', 'metadatas'],
        )
        # print("Query", "\n--------------")
        # print(query)
        # print("Result", "\n--------------")
        # print(res['documents'][0][0])
        # print("Vector", "\n--------------")
        # print(res['embeddings'][0][0])
        # print("")
        # print("")
        # print("Complete Response","\n-------------------------")
        # print(res)

    except Exception as e:
        print("Vector search failed : ", e)

    return res


#searchDataByVector(query)

def augment_prompt(query: str):
    res = searchDataByVector(query)
    # get the text from the results
    source_knowledge = "\n".join(res['documents'][0][0])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

# create a new user prompt
prompt = HumanMessage(
    content = augment_prompt(query)
)
# add to messages
messages.append(prompt)

res = chat(messages)

print(query)
print(res.content)