import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings

st.set_page_config(page_title="Medical Advisor 👨🏻‍⚕️")
st.title('Medical-Advisor 👨🏻‍⚕️')
st.write("Hello! I'm a Medical Advisor, and I am here to help you answer your medical-related questions.")
st.write("Questions (sample) you can ask me:")
st.write("- How to diagnose Anxiety Disorders?")
st.write("- What are the treatments for Cough?")
st.write("- What are the stages of Prostate Cancer?")
st.write("- What causes Foodborne Illnesses?")
st.write("- How to diagnose Overweight and Obesity ?")
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

    prompt_template = """ You are an expert on the medical domain and your job is to answer questions based on medical knowledge. Assume that all questions are
# related to the medical domain. Keep your answers technical and based on facts – do not hallucinate features.
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
  text = st.text_area('Enter text:', 'How to diagnose Anxiety Disorders?')
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
        