from sentence_transformers import SentenceTransformer
import sqlite3
from func import read_html_file, split_text, initialize_db, store_embeddings
def main():
    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    url = "https://www.gutenberg.org/files/22381/22381-h/22381-h.htm"
    cleaned_text = read_html_file(url)
    text_chunks = split_text(cleaned_text, 300)
    
    # Generate embeddings
    embeddings = model.encode(text_chunks)

    # Initialize database and create table if not exists
    conn = initialize_db()

    # Store the embeddings and the sentences in the database
    store_embeddings(conn, text_chunks, embeddings)

    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()