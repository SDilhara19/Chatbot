import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from text_analysis import embed
from chatbot import chat_response
import os
import uuid

st.set_page_config(page_title="AI Chatbot", layout="wide")

st.markdown("""
## Get instant insights from Lex Fridman Podcasts
""")

def user_input(user_question, session_id):
    response = chat_response(user_question, session_id)
    st.subheader("Reply:")
    st.write(response["answer"])
    st.divider()
    st.write("Below are the video references:")
    for reference in response["references"]:
        with st.container(border=True):
            st.subheader(reference['title'])
            st.write(f"Speaker: {reference['speaker']}")
            st.write(f"Go to video timestamp: [{reference['time']}]({reference['link']})")
            st.link_button("Watch full video", reference["video"], help=None, type="secondary", icon="🤓", disabled=False, use_container_width=True)
def generate_chat_id():
    #generates a unique chat id
    chat_id = str(uuid.uuid4())  
    return chat_id


def main():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_chat_id()

    user_question = st.text_input("Ask a Question from Lex Fridman podcasts", key="user_question")

    if user_question: 
        user_input(user_question, st.session_state["session_id"])

    with st.sidebar:
        st.title("Menu:")
        st.write("Add new podcast transcripts to process")
        st.subheader("Process the video transcript:")
        video = st.text_input("Enter the youtube video URL:", key="video_url")
        transcript_url = st.text_input("Enter the transcript URL:", key="transcript_url")

        if st.button("Submit & Process", key="process_button") and video and transcript_url:  
            with st.spinner("Processing..."):
                embed(transcript_url, video)
                st.success("Done")

if __name__ == "__main__":
    main()
