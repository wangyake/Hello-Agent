import tokenize
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen1.5-0.5B-Chat"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print(f'模型和分词器已加载')


# 准备对话输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好, 我想知道北京的天气"}
]

# 分词器格式化输入
# huggingface 的tokenizer 会针对不同的模型进行不同的处理，这里使用的是Qwen1.5-0.5B-Chat模型，
# 所以需要添加生成提示。
# 如果是chat-gpt，不需要添加生成提示。
# 如果是LLaMA2 ，用 <<SYS>>、[INST]、[/INST]
text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,
    add_generation_prompt=True
)
print('格式化后的文本:', text)
# 格式化后的文本: <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# 你好, 我想知道北京的天气<|im_end|>
# <|im_start|>assistant


# 编码输入文本
model_inputs = tokenizer(text, return_tensors="pt").to(device)
print('编码后的输入:', model_inputs)

# 使用分词器的 decode()方法，翻译回文字

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

# 将生成的Token ID 截取掉输入部分
# 这样我们只解码模型新生成的内容
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('模型生成的文本:', response)