# RAG_with_LLM
## Retrieval-Augmented Generation with OpenAI LLM
![Alt text](https://github.com/changyuhsin1999/RAG_with_LLM/blob/main/images/Greek_gods.jpg)


Have you ever pondered upon the ancient Greek myths and the relationships between Greek characters? This project bridges the gap between historical texts and modern technology, offering an innovative approach to exploring the intricate world of Greek mythology. Leveraging the power of Retrieval-Augmented Generation (RAG) combined with ChatGPT Large Language Models, I've created an interactive Streamlit application that delves into the depths of Greek mythology.

Below is the structured architect of this application in a diagram form
![Screenshot](https://github.com/changyuhsin1999/RAG_with_LLM/blob/main/images/Screenshot%202024-02-19%20at%201.03.54%20PM.png)

## Data
I used the book "The Myth and Legends of Ancient Greece and Rome" - by E.M.Berens for word embedding information

## Transformer model
model = "all-MiniLM-L6-v2"
This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

## Vector Database
I uses SQLite3 for vector storage, since the size of vector data is only about 3000 text chunks

## Cosine Similarity Search
I embedded the user input question using the same transformer model and compare the cosin similarity between user embedding and the embedding from the book

## LLM
Find the top 5 most similar relevant text from the book and feed that into ChatGPT prompt
Use Openai API key to generate response

## Stremlit Frontend
User can type their input question into the text box and hit "submit". In the response textbox, the model will generate an answer considering all the relevent text from the book.
Dive into the ancient world and ask any question, discover the stories of gods, heroes, and monsters from ancient Greece and Rome.
![Screenshot](https://github.com/changyuhsin1999/RAG_with_LLM/blob/main/images/Screenshot%202024-02-19%20at%201.49.47%20PM.png)

## Evaluation
I use human evaluation to evaluate through 10 different questions and rate it in the following categories and take the average
We can see that LLM with RAG perform better in answering questions with details
![Screenshot](https://github.com/changyuhsin1999/RAG_with_LLM/blob/main/images/Screenshot%202024-02-21%20at%2010.51.30%20PM.png)
