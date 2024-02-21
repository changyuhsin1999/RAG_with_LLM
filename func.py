import sqlite3
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

def read_html_file(url):
    """Use BeautifulSoup and html parser to extract the text from an url link

    Args:
        url (str): html link in a string form

    Returns:
        cleaned_text(str): cleaned text extracted from the website
    """
    html = urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    # Split text into words and join with a single space to remove extra whitespace
    cleaned_text = ' '.join(text.split())
    return cleaned_text



def split_text(text, chunk_size):
    """Split the extracted text into chunks with controllable chunk_size

    Args:
        text (str): extracted text from the html link
        chunk_size (int): controllable token number for a chunk

    Returns:
        text_chunks(list): A list of text chunks with specific size from the extracted text
    """
    text_chunks = []
    text_length = len(text)
    start = 0
    while start < text_length:
        end = start + chunk_size
        # Ensure we do not go beyond the text's length
        if end > text_length:
            end = text_length
        else:
            # Find the last period in the current chunk
            last_period_index = text.rfind('.', start, end)
            if last_period_index != -1:
                end = last_period_index + 1  # Include the period in the chunk
            else:
                # If there's no period, we extend to the next period outside the current chunk
                next_period_index = text.find('.', end)
                if next_period_index != -1:
                    end = next_period_index + 1  # Include the period in the chunk
                else:
                    # If there are no more periods, we take the rest of the text
                    end = text_length
        chunk = text[start:end]
        text_chunks.append(chunk)
        start = end
    return text_chunks

def create_embeddings(text_chunks, model="all-MiniLM-L6-v2"):
    """Create word embeddings from the text chunks to size 1536 vector

    Args:
        text_chunks (list): extracted text into chunks
        model (str): embedding model transferring text to vector. Defaults to "all-MiniLM-L6-v2".

    Returns:
        embeddings(list): list of embedded word in vector form
    """
    embeddings = []
    try:
        prepared_chunks = [chunk.replace("\n", " ") for chunk in text_chunks]
        response = openai.Embedding.create(input=prepared_chunks, model=model)
        if response and "data" in response:
            for data in response["data"]:
                embeddings.append(data["embedding"])
        return embeddings
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None
    
    
def initialize_db(db_path='embeddings.db'):
    """Initializing SQLite database to store the word embeddings, create a table if not exist already. Table should include and primary key id and the actual text chunk and the word embeddings corresponded to the text chunk

    Args:
        db_path (str, optional): Name of your SQLite database. Defaults to 'embeddings.db'.
    Returns:
        conn: Initialized database
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create a table with id, text, and embedding (stored as BLOB)
    c.execute('''CREATE TABLE IF NOT EXISTS sentence_embeddings
                 (id INTEGER PRIMARY KEY, text TEXT, embedding BLOB)''')
    conn.commit()
    return conn

def store_embeddings(conn, texts, embeddings):
    """Insert word embeddings and the text chunks into the SQLite database for future retrieval step

    Args:
        conn : Initialized SQLite database
        texts (list): A list of actual text from the text chunk
        embeddings (list): A list of word embeddings that was transformed from the text chunks
    """
    c = conn.cursor()
    for text, embedding in zip(texts, embeddings):
        # Convert numpy array to bytes for storage
        embedding_blob = embedding.tobytes()
        c.execute("INSERT INTO sentence_embeddings (text, embedding) VALUES (?, ?)",
                  (text, embedding_blob))
    conn.commit()
    
def retrieve_top_5(user_query_embedding):
    """Convert user input question into word embeddings and calculate cosine similarity between the user query and the database embeddings. Find the top 5 closest text chunk that is related to the user input question.

    Args:
        user_query (str): User input question in string

    Returns:
        top_5_texts(list): A list of text chunks in string that has the closest cosine similarity between db embeddings and user query embeddings
    """
    import sqlite3
    import numpy as np
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()

    # Fetch all embeddings and their corresponding texts from the database
    cursor.execute("SELECT embedding, text FROM sentence_embeddings")
    rows = cursor.fetchall()
    
    if not rows:
        print("No embeddings found in the database.")
        return []

    # Convert embeddings from database (assuming they are stored as lists or similar structure) and calculate cosine similarity
    db_embeddings = np.array([np.frombuffer(row[0], dtype=np.float32) for row in rows])
    if user_query_embedding.size == 0 or db_embeddings.size == 0:
        print("User query embedding or database embeddings are empty.")
        return []
    user_query_embedding = user_query_embedding.reshape(1, -1)
    cos_similarities = cosine_similarity(user_query_embedding, db_embeddings)

    # Get the indices of the top 5 most similar embeddings
    top_5_indices = np.argsort(cos_similarities[0])[-5:]

    # Retrieve the corresponding texts for the top 5 embeddings
    top_5_texts = [rows[i][1] for i in top_5_indices]
    return top_5_texts

def query_llm_with_prompt(top_5_texts, user_query):
    """Integrating OpenAI LLM with prompting to generate a response to the user input question

    Args:
        top_5_texts (list): A list of text chunks in string that has the closest cosine similarity between db embeddings and user query embeddings

    Returns:
        response: answer generated by ChatGPT integrating the most relevant text from the database and the prompt
    """
    import os
    # Access the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Ensure the OpenAI API key is available
    if openai_api_key is None:
        raise ValueError("OpenAI API key not found. Please make sure it's set in .env file.")

    import openai
    openai.api_key = openai_api_key
    
    prompt = f"Based on the following information: {top_5_texts} Generate an answer to this user question: {user_query}."
    messages=[{"role": "assistant", "content": "You are an expert in Greek myth, please answer questions with more information about this Greek story, do not make up your own story"}, {"role": "user", "content": prompt}]
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    response = completion.choices[0].message.content
    return response

def query_llm(user_query):
    import os
    # Access the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Ensure the OpenAI API key is available
    if openai_api_key is None:
        raise ValueError("OpenAI API key not found. Please make sure it's set in .env file.")

    import openai
    messages=[{"role": "assistant", "content": "You are an expert in Greek myt, please answer user's questions"}, {"role": "user", "content": user_query}]
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    response_without_RAG = completion.choices[0].message.content
    return response_without_RAG