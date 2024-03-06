import os
import dotenv
from langchain_openai import ChatOpenAI


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

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
    HumanMessage(content="I'd like to understand string theory.")
]

res = chat(messages)

print(res.content)