from memit.experiments.py.demo import demo_model_editing
import os
import json
from datetime import datetime

def save_model(model, tokenizer, save_path, edit_info):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
            
    with open(os.path.join(save_path, "edit_info.json"), "w") as f:
        json.dump(edit_info, f, indent=2)
    
    print(f"Model saved to {save_path}")


def apply_memit(model, tokenizer, requests, hyperparams, save_path, edit_info, save_new_model = True):
    model_new, orig_weights = demo_model_editing(model, tokenizer, requests, hyperparams=hyperparams)

    if save_new_model:
        save_model(model_new, tokenizer, save_path, edit_info)

    return model_new
