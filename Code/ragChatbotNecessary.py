import os
import dotenv
import pandas as pd
import time
from langchain_openai import ChatOpenAI
import chromadb
# from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

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

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
client = chromadb.PersistentClient(path="Collections/")
collection = client.get_collection(name="NecessaryKB")
messages = [SystemMessage(content="You are a helpful advising assistant for UMass Amherst CICS.")]

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
    # print(res['documents'][0][0])
    # get the text from the results
    source_knowledge = "\n\n".join(res['documents'][0])
    metadata = "\n\n".join([x['source'] for x in res['metadatas'][0]])
    print(metadata)
    print(source_knowledge)
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query. Then give a score from 1-10 on how confident you are in the answer's correctness based on if the context was relevant to answering the query Lastly, return the links of the context. 

    Contexts:
    {source_knowledge}

    Links of the context:
    {metadata}
    
    Query: {query}"""
    return augmented_prompt


# CHANGE THIS LINE TO CHANGE THE QUERY
query = "What are 2 faculty members in the Advanced Learning Technologies Laboratory at UMass Amherst?"
#print(augment_prompt(query))

# create a new user prompt
prompt = HumanMessage(
    content = augment_prompt(query)
)
# add to messages
messages.append(prompt)

res = chat(messages)

print(query)
print(res.content)