# Gender-Occupation-Bias A-MEMIT-based-Approach

AI and Large Language Models (LLMs) have achieved significant progress in the last years, yet their increasing complexity has made them difficult to interpret and prone to perpetuating social biases present in their training data. 
The more these models become complex, the more difficult it is to provide interpretability and address ethical concerns to produce fairer solutions.

In particular, gender stereotypes are generally very prominent in occupations. However, to mitigate the cost of training such big systems, we would like to identify and editing the internal representations of the model that associate occupations with specific genders, or any other bias in general. Naturally, this task must be done without worsening the actual performance of the model, nor should change what the model knows about the professions themselves.


Instructions:
- MEMIT (https://github.com/kmeng01/memit) must be placed in ```./memit```
- GPT2 (https://huggingface.co/openai-community/gpt2/tree/main) in ```./gpt2```
- Llama-3.2-3B (https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main) in ```./llama-3.2-3B```
