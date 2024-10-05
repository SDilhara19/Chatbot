from  global_funcs import * 
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os 
from langchain.vectorstores import FAISS

def embed(url, video_url):
    webpage = extract_webpage(url)
    text_with_metadata = add_metadata(webpage, video_url)
    chunks = split_docs(text_with_metadata)
    create_embeddings(chunks)
    # print(chunks)
    return True

def create_embeddings(chunks):
    storing_path="./vectorstore"
    embedding_model = load_embedding_model()

    os.makedirs(os.path.dirname(storing_path), exist_ok=True) #ensure dir exists
    # Creating the embeddings using FAISS
    if os.path.exists(storing_path):
        # Load the existing vectorstore
        print("flag1")
        vectorstore = FAISS.load_local(storing_path, embedding_model, allow_dangerous_deserialization=True)
        # Add new embeddings to the existing vectorstore
        # vectorstore.add_documents(chunks)
        vectorstore.add_documents(chunks)
        # Save the updated vectorstore
        print("Updated vectorstore")

    else:
        # Create a new vectorstore
        print("flag2")
        vectorstore = FAISS.from_documents(chunks, embedding_model)

    # Save the updated vectorstore
    vectorstore.save_local(storing_path)

    return True

def add_metadata(page, video_url):
    title = page.find('h1', class_ = "entry-title").text
    content = page.findAll('div', class_ = "ts-segment")
    pattern = r'\(\d{2}:\d{2}:\d{2}\)'
    documents = []
    prev_speaker = None
    for chunk in content:
        name = chunk.find('span', class_ = "ts-name").text
        if name == "":
            name = prev_speaker
        else:
            prev_speaker = name
        link = chunk.find('a',  href=True)
        video = video_url
        dialogue = clean_text(chunk.find('span', class_ = "ts-text").text)
        metadata={
            "speaker": name,
            "timestamp": link['href'],
            "timestamp_text": link.text,
            "video": video,
            "title": title
        }
        new_doc = Document(page_content=dialogue, metadata=metadata)
        documents.append(new_doc)
    return documents

def clean_text(text):
    transformations = [
        lambda t: t.replace("\n", " "),
        remove_disfluencies,
        handle_informal_text,
        lambda t: re.sub(' +', ' ', t).strip()
    ]
    for transform in transformations:
        text = transform(text)
    return text

def remove_disfluencies(text):
    disfluency_pattern = r'\b(um|uh|you know|like|I mean|sort of|kind of|actually|basically|literally|well|so)\b'
    return re.sub(disfluency_pattern, '', text)

def handle_informal_text(transcript):
    informal_replacements = {
        "gonna": "going to",
        "gotta": "have to",
        "wanna": "want to",
        "kinda": "kind of",
        "sorta": "sort of",
        "'cause": "because",
        "lemme": "let me",
        "gimme": "give me",
        "dunno": "do not know",
        "outta": "out of",
        "gimme": "give me",
        "ain't": "is not",
        "gonna": "going to",
        "wanna": "want to",
        "gimme": "give me",
        "gotta": "got to",
        "super": "very",
    }

    # Loop through the informal words and replace them with the formal alternatives
    for informal, formal in informal_replacements.items():
        transcript = transcript.replace(informal, formal)
    return transcript

def split_docs(text):
    chunk_size = 500
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(text)
    return splits

def main():
    embed("https://lexfridman.com/elon-musk-4-transcript/#chapter1_war_and_human_nature", "https://www.youtube.com/embed/jvqFAi7vkBc")

if __name__ == "__main__":
    main()
    