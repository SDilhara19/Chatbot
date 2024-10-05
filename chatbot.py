from Convogrid_Assessment.tone_analyser import tone_analyser
from Convogrid_Assessment.global_funcs import *  
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
 
llm = Ollama(model="llama3.2:1b", temperature=0)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_response(query, chain, session_id, tone):
    # Getting response from chain
    tone = [tone]
    response = chain.invoke(
       {"input": query,
        "tone": tone},
        config={
        "configurable": {"session_id": session_id, "tone": tone}
        }, 
    )

    return response

def create_conversational_chain(rag_chain, store=store):
    
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history= get_session_history,
        input_messages_key="input",
        
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

def create_chain(history_aware_retriever, llm, qa_prompt):
    # Creating the chain for Question Answering
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def add_history(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

def run_chain(retriever, prompt, session_id):
    retriever_with_history = add_history(llm, retriever)
    tone = tone_analyser(prompt)
    print(tone) 
    system_prompt = (
        """You are a knowledgeable assistant who will provide insights from given podcast transcripts
        In the  {context} you are given transcriptions of youtube podcasts hosted by Lex Fridman with various guest speakers.
        Identify the speaker who has spoken about the given {input} and provide the answer to the {input} based on the speakers ideas.
        Give your answer naturally and coherently just like a speaker/guest in a podcast.
        Provide the answer in {tone} tone since the user's {input} is in {tone} tone
        Analyze the tone of the user's {input} and provide the answer in the same tone.
        {context}"""
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    rag_chain = create_chain(retriever_with_history, llm, qa_prompt)
    conversational_rag_chain = create_conversational_chain(rag_chain)
    response = get_response(prompt, conversational_rag_chain, session_id, tone)
    return response

def get_references(context):
    references = []
    for document in context:
        reference = {
            "speaker": document.metadata["speaker"],
            "link": document.metadata["timestamp"],
            "video": document.metadata["video"],
            "title": document.metadata["title"],
            "time": document.metadata["timestamp_text"],
        }
        references.append(reference)
    return references

def retrieve_transcript():
    #retrieves transcripts
    try:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return retriever
    except Exception as e:
        print(e)
        return False

def chat_response(prompt, session_id):
    retrieved_transcripts = retrieve_transcript()

    response= run_chain(retrieved_transcripts, prompt, session_id)
    output = {
        "prompt": prompt,
        "answer": response["answer"],
        "references": get_references(response["context"]),
    }
    print(output["answer"]) 
    print(output["references"])
    print(f"store: {store}")  
    return output

def main():
    chat_response("What is the timestamp when Elon Musk fisrt spoke in the podcast?")

if __name__ == "__main__":
    main()
    