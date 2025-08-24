from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model(path, device="cuda"):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map=device
    )

    return model, tokenizer
    

if __name__ == '__main__':
    path = './llama-3.2-3B_exp1'
    m, tokenizer = get_model(path)
    # Generate text
    prompt = "Females are to ... like males are to nurse"
    inputs = tokenizer(prompt, return_tensors="pt").to(m.device)
    outputs = m.generate(**inputs, max_new_tokens=100)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
