import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
# استخدام جوجل البديل المجاني والمستقر
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- الـ htmlTemplates ---
css = '''
<style>
.chat-message { padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex }
.chat-message.user { background-color: #2b313e }
.chat-message.bot { background-color: #475063 }
.chat-message .avatar { width: 20%; }
.chat-message .avatar img { max-width: 78px; max-height: 78px; border-radius: 50%; object-fit: cover; }
.chat-message .message { width: 80%; padding: 0 1.5rem; color: #fff; }
</style>
'''
bot_template = '<div class="chat-message bot"><div class="avatar"><img src="https://i.ibb.co/cN0D91G/animated-ai-robot.jpg"></div><div class="message">{{MSG}}</div></div>'
user_template = '<div class="chat-message user"><div class="avatar"><img src="https://i.ibb.co/vHQ6gS0/user-avatar.jpg"></div><div class="message">{{MSG}}</div></div>'

class LegacyBridgeChain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.chat_history = []
        
        condense_question_system_template = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question which can be understood without the chat history."
        )
        condense_question_prompt = ChatPromptTemplate.from_messages([
            ("system", condense_question_system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        
        self.condense_chain = condense_question_prompt | self.llm | StrOutputParser()
        
        qa_system_template = "Answer the user's questions based on the below context:\n\n{context}"
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        
        self.qa_chain = qa_prompt | self.llm

    def _get_standalone_question(self, question):
        if not self.chat_history:
            return question
        return self.condense_chain.invoke({"question": question, "chat_history": self.chat_history})

    def __call__(self, inputs):
        user_question = inputs['question']
        standalone_question = self._get_standalone_question(user_question)
        docs = self.retriever.invoke(standalone_question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        response = self.qa_chain.invoke({
            "question": standalone_question,
            "context": context,
            "chat_history": self.chat_history
        })
        
        self.chat_history.append(HumanMessage(content=user_question))
        self.chat_history.append(AIMessage(content=response.content))
        return {'chat_history': self.chat_history}


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)


# استخدام الـ Embeddings المجانية من جوجل
def get_vectorstore(text_chunks):
    google_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# استخدام موديل الـ Chat المجاني والسريع من جوجل
def get_conversation_chain(vectorstore):
    google_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_key)
    return LegacyBridgeChain(llm=llm, retriever=vectorstore.as_retriever())


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput(user_question)
        else:
            st.warning("Please upload your PDFs and click 'Process' first before asking a question!")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()