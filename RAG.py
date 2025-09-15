from io import BytesIO
import os
import faiss
import numpy as np
from groq import Groq
import requests
from sentence_transformers import SentenceTransformer

class Rag:

    client = Groq()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    d = 384  # Dimension of embeddings (depends on model)
    all_chunks = []
    index = faiss.IndexFlatL2(d)
    md_files = [
        "https://drive.google.com/file/d/1fL7rm77LFAXpggUV0wrubg1rrTr-HNGv/view?usp=sharing",
        "https://drive.google.com/file/d/1WPO7lbDdn5cxVLOLPbDd96CaB8PUtAWr/view?usp=sharing",
        "https://drive.google.com/file/d/1XsIfcGnZ-XnfEtQYyvJv2XmFY9jTuwHM/view?usp=sharing",
        "https://drive.google.com/file/d/1L_ByUCKCkari5shat-qw9yobvG_v5e81/view?usp=sharing",
    ]

    def __init__(self):
        pass

    # Function to read text from a Markdown file
    def extract_text_from_md(self, md_path):
        try:
            with open(md_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            print(f"Error reading {md_path}: {e}")
            return ""

    # Function to chunk text
    def chunk_text(self, text, chunk_size=300):
        words = text.split()
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
        return chunks

    # Function to embed and store in FAISS
    def store_in_faiss(self, chunks):
        if not chunks:
            print("No chunks to embed.")
            return [], []

        embeddings = self.embedding_model.encode(chunks)
        if embeddings.shape[0] == 0:
            print("Embeddings are empty.")
            return [], []

        faiss.normalize_L2(embeddings)
        self.index.add(np.array(embeddings, dtype=np.float32))
        return embeddings, chunks

    # Function to retrieve most relevant chunk
    def retrieve_from_faiss(self, query):
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        _, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k=1)
        return indices[0][0]

    # Function to get response from Groq LLM
    def get_llm_response(self, query, context):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that only answers questions strictly related to the uploaded documents. If a query is irrelevant, respond with 'I can only answer questions related to the provided documents.'",
                },
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"},
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content

    # Function to download PDF from Google Drive
    def download_md_file(self, drive_link):
        try:
            file_id = drive_link.split("/d/")[1].split("/")[0]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(download_url)
            if response.status_code == 200:
                return response.text  # No BytesIO — just return text
            else:
                raise Exception(f"Failed to download file: {drive_link}")
        except Exception as e:
            print(f"Download error: {e}")
            return None

    def doc_precess(self):
        for link in self.md_files:
            print(f"Processing document: {link}")
            md_text = self.download_md_file(link)
            # print(md_text)
            if md_text:
                chunks = self.chunk_text(md_text)
                if not chunks:
                    print("No chunks generated.")
                    continue
                _, stored_chunks = self.store_in_faiss(chunks)
                self.all_chunks.extend(stored_chunks)
        print(len(self.all_chunks))
        print("All documents processed successfully!")

    def rag_answer_generator(self, query: str):
        retrieved_chunk_index = self.retrieve_from_faiss(query)
        retrieved_chunk = self.all_chunks[retrieved_chunk_index]
        response = self.get_llm_response(query, retrieved_chunk)
        return response
