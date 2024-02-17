import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import SentenceTransformer
from func import read_html_file, split_text, create_embeddings, initialize_db, store_embeddings, retrieve_top_5, query_llm_with_prompt

## Title.
st.write('# Greek story encyclopedia')

#Book url
url = "https://www.gutenberg.org/files/22381/22381-h/22381-h.htm"

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## User input
prompt = st.chat_input("Ask me anything about Greek myth")
st.session_state.messages.append({"role": "user", "content": prompt})

top_5_texts = retrieve_top_5(prompt)
response = query_llm_with_prompt(top_5_texts)

