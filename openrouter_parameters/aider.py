import requests
import json

__open_router_api_key = "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"
url = "https://openrouter.ai/api/v1/chat/completions"

def ask_llm(model = "anthropic/claude-3.5-sonnet",role = "user",temperature=1.0 ,top_p=1.0,frequency_penalty=0.0,repetition_penalty=1.0,max_tokens=200.0):
    prompt_content = []
    count =0
    while True:
        headers = { "Authorization":f"Bearer {__open_router_api_key}",
               "Content-Type":"application/json"}
        user_prompt = get_user_input(count)
        count +=1
        prompt_content.append({"role":f"{role}" , "content":f"{user_prompt}"})
        data = {"model" : model,
                "messages":prompt_content ,
                "temperature":temperature,
                "top_p":top_p,
                "frequency_penalty":frequency_penalty,
                "repetition_penalty":repetition_penalty,
                "max_tokens":max_tokens
                }
        llm_json_response = requests.post(url=url,json=data,headers=headers)
        llm_response =""
        if llm_json_response.status_code == 200:
            llm_response = llm_json_response.json().get("choices").pop(0).get("message").get("content")
            prompt_content.append({"role":"assistant","content":f"{llm_response}"})
        else:
            print("Error : ", llm_json_response)
        print(f"{llm_response}")
        print()
        
def get_user_input(count =0):
    return input("Ask Me anything : \n") if(count ==0) else input("\n")

ask_llm(model="openai/gpt-4o-mini",temperature=0,repetition_penalty=0,frequency_penalty=-2,max_tokens=100)

"""
#WORKING EXAMPLE:
promt_text = "My Name is vignesh , write a small poem on me !"
model = "anthropic/claude-3.5-sonnet"
data = {
    "model":f"{model}",
    "prompt":f"{promt_text}",
    "temperature":0.0,
    "top_p":0.5,
    "frequency_penalty": -2.0,
    "system":system
}

response = requests.post(url=url,headers=headers,json=data)
print("Response Code : " ,response.status_code)
response_json = response.json()
"""