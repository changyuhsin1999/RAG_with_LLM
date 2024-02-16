def read_html_file(file_path):
    """
    Use BeautifulSoup and html parser to extract the text from an url link
    """
    html = urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    # Split text into words and join with a single space to remove extra whitespace
    cleaned_text = ' '.join(text.split())
    return cleaned_text



def split_text(text, chunk_size):
    """
    Split the extracted text into chunks with controllable chunk_size
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

def create_embeddings(text_chunks, model="text-embedding-ada-002"):
    """Create word embeddings from the text chunks to size 1536 vector

    Args:
        text_chunks (list): extracted text into chunks
        model (str): embedding model transferring text to vector. Defaults to "text-embedding-ada-002".

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
    """_summary_

    Args:
        db_path (str, optional): _description_. Defaults to 'embeddings.db'.

    Returns:
        _type_: _description_
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create a table with id, text, and embedding (stored as BLOB)
    c.execute('''CREATE TABLE IF NOT EXISTS sentence_embeddings
                 (id INTEGER PRIMARY KEY, text TEXT, embedding BLOB)''')
    conn.commit()
    return conn

def store_embeddings(conn, texts, embeddings):
    """_summary_

    Args:
        conn (_type_): _description_
        texts (_type_): _description_
        embeddings (_type_): _description_
    """
    c = conn.cursor()
    for text, embedding in zip(texts, embeddings):
        # Convert numpy array to bytes for storage
        embedding_blob = embedding.tobytes()
        c.execute("INSERT INTO sentence_embeddings (text, embedding) VALUES (?, ?)",
                  (text, embedding_blob))
    conn.commit()
    
def retrieve_top_5(user_query_embedding):
    """_summary_

    Args:
        user_query_embedding (_type_): _description_

    Returns:
        _type_: _description_
    """
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()

    # Fetch all embeddings and their corresponding texts from the database
    cursor.execute("SELECT embedding, text FROM sentence_embeddings")
    rows = cursor.fetchall()

    # Convert embeddings from database (assuming they are stored as lists or similar structure) and calculate cosine similarity
    db_embeddings = np.array([np.frombuffer(row[0], dtype=np.float32) for row in rows])
    cos_similarities = cosine_similarity(user_query_embedding, db_embeddings)

    # Get the indices of the top 5 most similar embeddings
    top_5_indices = np.argsort(cos_similarities[0])[-5:]

    # Retrieve the corresponding texts for the top 5 embeddings
    top_5_texts = [rows[i][1] for i in top_5_indices]
    return top_5_texts

def query_llm_with_prompt(top_5_texts):
    """_summary_

    Args:
        top_5_texts (_type_): _description_

    Returns:
        _type_: _description_
    """
    prompt = f"Based on the following information: \n1. {top_5_texts[0]}\n2. {top_5_texts[1]}\n3. {top_5_texts[2]}\n4. {top_5_texts[3]}\n5. {top_5_texts[4]}\nGenerate an answer to this user question: {user_query}."
    messages=[{"role": "assistant", "content": "You are an expert in Greek myth, please answer questions with more information about this Greek story, do not make up your own story"}, {"role": "user", "content": prompt}]
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    response = completion.choices[0].message.content
    return response