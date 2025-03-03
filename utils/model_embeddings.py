import torch
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from utils.constants import USE_GPU, USE_QUANTIZATION
from utils.duration_decorator import measure_time

has_cuda = torch.cuda.is_available()
print(f"Support GPU : {has_cuda}")
if has_cuda:
    print(f"Version CUDA : {torch.version.cuda}")

def get_embedding_model_chroma():
    """
    Transformer pour les embeddings à stocker ou la recherche dans ChromaDB
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_model_name_llm():
    """ 
    Charger Llama 2 pour générer une réponse humaine en langage naturel 
    NousResearch : https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
    Config https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/blob/main/config.json
    Diff https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/discussions/11/files pour quantization_config
    meta (demandera une autorisation) : https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
    """
    return "NousResearch/Llama-2-7b-chat-hf"

def cuda_or_cpu():
    return "cuda" if has_cuda and USE_GPU else "cpu"

def _get_config_quantification():
  quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # quantification en 4 bits
    bnb_4bit_quant_type='nf4',  # Utilise NormalFloat4 pour une meilleure précision
    bnb_4bit_compute_dtype=torch.float16  # Utilise float16 pour les calculs
  )
  return quantization_config

@measure_time
def model_inference_init():
    """
    Selon CUDA avec PyTorch, initialisera le modèle d'inférence LLM avec GPU CUDA ou CPU
    """
    if USE_QUANTIZATION:
        # https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
        # max_memory = {0: "10GiB", "cpu": "32GiB"}
        _model = AutoModelForCausalLM.from_pretrained(
            get_model_name_llm(),
            torch_dtype=torch.bfloat16,
            quantization_config=_get_config_quantification(),
            # max_memory=max_memory,
            device_map='auto',  # répartit automatiquement entre CPU et GPU
            # llm_int8_enable_fp32_cpu_offload=True # uniquement en 8 bits
        )
        _device = torch.device(cuda_or_cpu())
        return _model.to(_device)
    else:
        return AutoModelForCausalLM.from_pretrained(get_model_name_llm(),
                                                    torch_dtype=torch.float16
                                                    ).to(cuda_or_cpu())

def calculate_inference_tokens_time(generated_tokens, start_time, end_time):
    inference_time = end_time - start_time
    tokens_per_sec = generated_tokens / inference_time if inference_time > 0 else float("inf")
    print(f"Vitesse inférence : {tokens_per_sec:.2f} tokens/sec")