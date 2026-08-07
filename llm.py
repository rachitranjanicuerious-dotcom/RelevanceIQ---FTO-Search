from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    api_key=OPENAI_API_KEY
)

HF_TOKEN = os.getenv("HF_TOKEN")

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0,
    max_new_tokens=4096
)

huggingface_llm = ChatHuggingFace(
    llm=hf_endpoint
)

# Select the model to use

# llm = openai_llm

llm = huggingface_llm