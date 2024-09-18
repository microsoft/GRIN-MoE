docker run --gpus all --rm -it nvcr.io/nvidia/pytorch:24.08-py3 /bin/bash -c $'
pip install accelerate==0.33.0 transformers==4.44.2
python -c \'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 

model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/GRIN-MoE",
    device_map="auto",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 
tokenizer = AutoTokenizer.from_pretrained("microsoft/GRIN-MoE")
pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 
generation_args = { 
    "max_new_tokens": 2048, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "Please complete the following single-choice question. The question has four options, and only one of them is correct. Select the option that is correct. \\n\\nQuestion: Given $\\\\cos(\\\\alpha + \\\\beta) = m$, and $\\\\tan(\\\\alpha) \\\\tan(\\\\beta) = 2$, we want to find $\\\\cos(\\\\alpha - \\\\beta) =$\\n\\nOption A: $-3m$\\nOption B: $-\\\\frac{m}{3}$\\nOption C: $\\\\frac{m}{3}$\\nOption D: $3m$"}
]
    
output = pipe(messages, **generation_args) 

print("-----------QUESTION-----------")
print(messages[1]["content"])
print("-----------Model RESPONSE-----------")
print(output[0]["generated_text"])
print("-----------Correct Answer RESPONSE-----------")
print("The correct answer of this question is A, -3m")
\' 
'