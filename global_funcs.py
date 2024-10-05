from langchain_huggingface import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS

def load_embedding_model(model_path="all-MiniLM-L6-v2", normalize_embedding=True, clean_up_tokenization_spaces=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"
        }, 
        encode_kwargs={
            "normalize_embeddings": normalize_embedding, 
        },
    )

def extract_webpage(webpage):
    try:
        page_content = requests.get(webpage)
        soup = BeautifulSoup(page_content.text, 'html.parser')
        return soup
    except Exception as e:  
        print(e)
        return False


def load_vectorstore(storing_path="./vectorstore"):
    try:
        embeddings = load_embedding_model()
        vectorstore = FAISS.load_local(
            storing_path, embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        print(e)
        return False

