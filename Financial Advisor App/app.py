import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings

st.set_page_config(page_title="Financial Advisor 👨🏻‍💼💰")
st.title('Financial Advisor 👨🏻‍💼💰')
st.write("Hello! I'm here as your Financial Advisor, committed to making your financial journey smooth and successful. I can provide support across various aspects of your financial journey.")
st.write("Questions (sample) you can ask me:")
st.write("- Basic finance: what should everyone know?")
st.write("- How can I preserve my wealth for future generations?")
st.write("- Investment Portfolio Setup for beginner")
st.write("- How do I pick the right company for investing services?")
st.write("- What are the benefits of investing in alternative investments?")
st.sidebar.subheader("App created by:")
st.sidebar.info("[prasadm](https://twitter.com/prsdm17)", icon="👨🏻‍💻")
st.sidebar.subheader("Your OpenAI Key Pleasae")
openai_api_key = st.sidebar.text_input('Put Your Key Here: ',type="password") 
def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
  st.info(llm(input_text))
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings)

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """You are an expert on the finance domain and your job is to answer questions based on finance knowledge. Assume that all questions are 
    related to the finance domain. Keep your answers technical and based on facts – do not hallucinate features.
    CONTEXT: {context}
    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

with st.form('my_form'):
  text = st.text_area('Enter text:', 'How can I preserve my wealth for future generations?')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)
with st.sidebar:
    if openai_api_key.startswith('sk-'):
        st.success('API key already provided!', icon='✅')
    else:
        st.warning('Please enter your credentials!', icon='⚠️')   
        