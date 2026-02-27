import numpy as np
import faiss
import requests
import sys 

openrouter_api_key = "YOUR_API_KEY"

headers = {
    "Authorization": f"Bearer {openrouter_api_key}",
    "Content-Type":"application/json"
}

chat_model = "openai/gpt-4o-mini"
url = "https://openrouter.ai/api/v1/chat/completions"

embedding_model = "openai/text-embedding-3-small"
embedding_url = "https://openrouter.ai/api/v1/embeddings"

#temperature = 0.0-2.0
def call_llm(messages, temperature= 0.3 ):
    payload = {
        "model": chat_model,
        "messages": messages,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_embedding_model(texts):
    payload={
        "model":embedding_model,
        "input":texts
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        array = np.array( [item["embedding"] for item in response.json()["data"]])
        return array.astype("float32")
    except:
        print(sys.exc_info())
        print(response)

def load_file_and_get_file_lines():
    lines = []
    try:
        with open("RAG/learning_rag.txt") as f:
            for line in f.readlines():
                if line.strip() != "":
                    lines.append(line.strip())
    except:
        print(sys.exc_info())
    return lines


def get_chunks_from_file():
    chunks=[]
    for index, line in enumerate(load_file_and_get_file_lines()):
        chunks.append({"chunk_id":index,"text":line})
    return chunks

print(get_chunks_from_file())