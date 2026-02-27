import sys
import numpy as np
import requests
#scikit-learn build on Numpy[Math library for Matrics/array calculation] which has ready made ML Libraries
from sklearn.metrics.pairwise import cosine_similarity
import faiss

openrouter_api_key ="YOUR_API_KEY"

headers = {
    "Authorization": f"Bearer {openrouter_api_key}",
    "Content-Type":"application/json"
}


chat_model = "openai/gpt-4o-mini"
url = "https://openrouter.ai/api/v1/chat/completions"

#temperature = 0.0-2.0

def call_llm(messages, temperature= 0.3 ):
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": chat_model,
        "messages": messages,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

messages = [{ 'role':'user','content':"What are you doing ?"}]

#print(call_llm(messages=messages))


embedding_url = "https://openrouter.ai/api/v1/embeddings"
embedding_model = "openai/text-embedding-3-small"

document =  [
    "Tamil Nadu population in 2024 is estimated to be 78 million.",
    "Chennai is the capital of Tamil Nadu.",
    "Tamil Nadu has a strong automobile manufacturing industry.",
    "The literacy rate of Tamil Nadu is above 80 percent."
]

question = "Does Tamilnadu has capital?"



def create_embeddings(texts):
    print("Input received:", texts)        # check what went in
    print("Number of texts:", len(texts))  # check count
    payload= {"model":embedding_model,"input":texts}
    try:
        response = requests.post(url=embedding_url,headers=headers,json=payload)
        response.raise_for_status()
        data = response.json()
        return np.array( [item["embedding"] for item in data["data"]]).astype("float32")
    except :
        print(sys.exc_info())
        print(data)


doc_vector = create_embeddings(document)
question_vector = create_embeddings(question)

print("Doc embedding shape:", doc_vector.shape)

similarity_index_list = cosine_similarity(doc_vector,question_vector)
max_index = np.argmax(similarity_index_list)

retrieved_context = document[max_index]
print("**Retrived Context:** " ,retrieved_context)
print("**Cosine Similarty and Max Index:** ", similarity_index_list,max_index)

llm_message = [{"role":'system', "content": "Answer using only the provided context. Just give me the number" },
               {"role":"user","content":f"Context:{retrieved_context},question:{question}"}]

#print(call_llm(llm_message))

#Create FAISS INDEX
dimension = 1536 # Taken from Embeddings/vector size which Model will return
#FLAT : Store the vectors as-is without compressing
#L2: L2 Distance - Instead of measuring the angle measure the straight-line distance
#IVF: indexIVFflat - Before we search  FAISS group similar vectors into clusters[buckets]. When ques asked , it searches only the specific cluster. Fast compared to indexFlatL2
index = faiss.IndexFlatL2(dimension)

index.add(doc_vector)

distance,embedding_index =index.search(question_vector,k=2)

print("** Distance && Index FULL Data :**" ,distance ,embedding_index)
for i in embedding_index[0]:
    print("** Distance && Index FROM FIASS :**" ,distance[0] ,document[i])

llm_content_from_faiss="\n".join(document[i] for i in embedding_index[0])

message_from_FIASS_for_llm = [{"role":"system","content":"Use only provided content to answer the question and if you don't know reply you don't know"},
                              {"role":"user","content":f"Context:{llm_content_from_faiss} \n Question:{question}"}]

print(call_llm(messages=message_from_FIASS_for_llm))