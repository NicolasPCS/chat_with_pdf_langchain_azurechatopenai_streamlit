from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

import streamlit as st
import tempfile

# Load environment variables
load_dotenv()

def main():
    st.title("PDF Question Answering App")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Create a temporary file to store the uploaded PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        document_path = temp_file.name  # Use the temp file path

        loader = PyPDFLoader(document_path)
        documents = loader.load_and_split()

        text_splitter = CharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=25
        )
        docs = text_splitter.split_documents(documents)

        embeddings = AzureOpenAIEmbeddings()

        vector_store = Chroma.from_documents(docs, embeddings)

        template = """
        You are a very helpful assistant, expert in helping analyst programmers understand client requirements. The requirements are documents that are delivered by the business analysis area, which prepares the document with requirement information and a technical solution, that is, the logic that the program will execute, the database tables involved, and programs involved in the solution's development. You must answer the questions in ENGLISH.
        Instructions:
        - All information in your answers must be retrieved from the PDF document or based on previous chat history.
        - In case the question cannot be answered using the information provided in the PDF (It is not relevant to the requirement), honestly state that you cannot answer that question.
        - Be detailed in your answers but stay focused on the question. Add all details that are useful to provide a complete answer, but do not add details beyond the scope of the question.
        PDF Context: {context}
        Question: {question}
        Helpful Answer:
        """

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        llm = AzureChatOpenAI(
            deployment_name="gpt-35-turbo-16k",
            temperature=0.8
        )

        retriever = vector_store.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm,
                                         retriever=retriever,
                                         return_source_documents=True,
                                         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

        # Historial del chat
        #chat_history = []

        question = st.text_input("Ask a question:")
        if question:
            # Actualizar el historial del chat
            #chat_history.append(question)
            #result = qa.invoke({"query": question, "chat_history": chat_history})
            result = qa.invoke({"query": question})
            answer = result["result"]  # Extracting the answer

            # Mostrar el historial del chat
            #st.subheader("Chat History")
            #for i, chat_question in enumerate(chat_history):
            #    st.write(f"{i+1}. {chat_question}")

            # Mostrar la respuesta
            st.subheader("Answer")
            st.write(answer)

if __name__ == '__main__':
    main()