#PINECONE VERSION

import os
import dotenv
import pandas as pd
import time
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings


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

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_KEY")

# configure client
pc = Pinecone(api_key=api_key)

spec = ServerlessSpec(
    cloud="aws", region="us-west-2"
)

index_name = 'llama-2-rag'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of ada 002
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
messages = []

# from tqdm.auto import tqdm  # for progress bar

# data = kbWhole

# batch_size = 100

# for i in tqdm(range(0, len(data), batch_size)):
#     i_end = min(len(data), i+batch_size)
#     # get batch of data
#     batch = data.iloc[i:i_end]
#     # generate unique ids for each chunk
#     ids = [str(i) for i, x in batch.iterrows()]
#     # get text to embed
#     texts = [x['Chunk'] for _, x in batch.iterrows()]
#     # embed text
#     embeds = embed_model.embed_documents(texts)
#     # get metadata to store in Pinecone
#     metadata = [
#         {'text': x['Chunk'],
#          'source': x['Link'],
#          'title': x['Title']} for i, x in batch.iterrows()
#     ]
#     # add to Pinecone
#     index.upsert(vectors=zip(ids, embeds, metadata))

#print(index.describe_index_stats())

from langchain.vectorstores import Pinecone

text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

query = "What is the CICS Advising office at UMass Amherst’s phone number?"

# print(vectorstore.similarity_search(query, k=3))

def augment_prompt(query: str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
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

print(res.content)

#NO LONGER USING PINECONE