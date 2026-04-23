import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# 1. 设置路径
model_path = r"D:\LLaVA_Model_Weights"
video_path = r"C:\Users\63091\Videos\2025-10-11 11-18-32.mp4"

# 2. 加载模型（针对 4070 8GB 的关键设置）
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    model_base=None, 
    model_name=model_name, 
    load_in_4bit=True,  # 必须开启量化
    device_map="auto"
)

print("✅ 模型加载成功！现在可以开始视频分析。")
# 接下来可以调用 model.generate 进行推理