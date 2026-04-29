import tokenize
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen1.5-0.5b-Chat"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print(f'模型和分词器已加载')
