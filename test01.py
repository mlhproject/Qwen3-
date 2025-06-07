from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']
    #加入这段代码解决兼容问题
model_path = "/data/22-klh/workspace/OpenBackdoor-main/LLM/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("✅ Tokenizer 加载成功")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
print("✅ 模型加载成功")


