import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
#OPENAI_API_KEY="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"

st.set_page_config(page_title="Educate Kids",page_icon=":robot:")
st.header("Hey, Ask me something")

embeddings = OpenAIEmbeddings()

from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})
data = loader.load()

#Assigning the data inside the csv to our variable here


db = FAISS.from_documents(data,embeddings)

def get_text():
  input_text = st.text_input
  return input_text

user_input=get_text()
sumbit = st.button('Find something')

if sumbit:
    docs = db.similarity_search(user_input)
    print(docs)
    st.subheader("Top Matches:")
    st.text(docs[0])
    st.text(docs[1].page_content)

