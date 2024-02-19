import openai
import csv
import numpy as np
from bs4 import BeautifulSoup
import warnings
import os
from dotenv import load_dotenv
from urllib.request import urlopen
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import streamlit as st
from func import read_html_file, split_text, create_embeddings, initialize_db, store_embeddings, retrieve_top_5, query_llm_with_prompt

load_dotenv()
model = SentenceTransformer("all-MiniLM-L6-v2")

def main():
    # Streamlit UI
    st.title("Greek Myth Encyclopedia")

    # Initialize the database
    conn = initialize_db()

    # Input for user query
    user_query = st.text_input("Ask me anything:")

    if st.button("Submit"):
        if user_query:
            # Retrieve top 5 related text chunks from the database
            user_query_embedding = model.encode([user_query])
        
            top_5_texts = retrieve_top_5(user_query_embedding)
            
            # Generate a response from the LLM
            response = query_llm_with_prompt(top_5_texts, user_query)
            
            # Display the response
            st.text_area("Response:", value=response, height=500)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

