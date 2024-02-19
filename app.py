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
from PIL import Image
from func import initialize_db, retrieve_top_5, query_llm_with_prompt, query_llm

load_dotenv()
model = SentenceTransformer("all-MiniLM-L6-v2")

def main():
    # Streamlit UI
    st.title("Greek Myth Encyclopedia")
    image = Image.open('/Users/cindychang/Desktop/RAG_with_LLM/images/Greek_gods.jpg')
    
    st.image(image)

    # Initialize the database
    conn = initialize_db()

    # Input for user query
    user_query = st.text_input("Ask me anything about Greek Myth:")

    if st.button("Submit"):
        if user_query:
            # Retrieve top 5 related text chunks from the database
            user_query_embedding = model.encode([user_query])
        
            top_5_texts = retrieve_top_5(user_query_embedding)
            
            # Generate a response from the LLM
            response_with_RAG = query_llm_with_prompt(top_5_texts, user_query)
            response_without_RAG = query_llm(user_query)
            # Display the response
            st.text_area("Response with RAG:", value=response_with_RAG, height=300)
            st.text_area("Response without RAG:", value=response_without_RAG, height=300)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

