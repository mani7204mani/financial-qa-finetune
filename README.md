# 💼 Financial Document QA — LLM Fine-tuning with QLoRA

A domain-specific Question Answering model fine-tuned on real financial 10-K documents using QLoRA (4-bit quantization) on TinyLlama 1.1B.

---

## 🚀 Project Overview

This project fine-tunes **TinyLlama 1.1B** using **QLoRA (Parameter-Efficient Fine-Tuning)** on the `virattt/financial-qa-10K` dataset containing 7,000+ real financial document QA pairs from SEC 10-K reports.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.688 |
| ROUGE-L | 0.658 |
| Trainable Parameters | 0.4% (LoRA) |
| Base Model | TinyLlama 1.1B |

---

## 🛠️ Tech Stack

- **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning:** QLoRA (4-bit quantization)
- **Libraries:** HuggingFace PEFT, TRL, BitsAndBytes, Transformers
- **Dataset:** virattt/financial-qa-10K (7,000+ samples)
- **Hardware:** Google Colab T4 GPU

---

## 📁 Project Structure
```
financial-qa-finetune/
├── FineTuning_Using_LoRA_and_PEFT.ipynb   # Full training notebook
└── README.md
```

---

## ⚙️ Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | TinyLlama 1.1B |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, v_proj, k_proj, o_proj |
| Learning Rate | 2e-4 |
| Batch Size | 4 |
| Epochs | 3 |
| Quantization | 4-bit NF4 |

---

## 📦 Installation
```bash
pip install torch transformers peft trl bitsandbytes accelerate datasets rouge_score huggingface_hub
```

---

## 🔍 Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINETUNED_MODEL = "manireddys/financial-qa-tinyllama"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, FINETUNED_MODEL)
model.eval()

def generate_answer(question, context):
    prompt = f"""### Instruction:
Answer the financial question based on the given context

### Input:
Context: {context}
Question: {question}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

answer = generate_answer(
    question="What is the total revenue?",
    context="The company reported total revenue of USD 2.5 billion for fiscal year 2024"
)
print(answer)
```

---

## 📈 Sample Results

| Question | Expected | Predicted | ROUGE-1 |
|----------|----------|-----------|---------|
| What were debt obligations? | $2,299,887 thousand | $2,299,887 thousand | 1.000 |
| What % came from Small Business segment? | 56% | 56% | 1.000 |
| What was operating margin change? | Increased 14.3% to 16.3% | Increased 14.3% to 16.3% | 1.000 |
| What was expected hurricane loss? | $3,827 million | $3,827 million | 1.000 |

---

## 🤗 Model on HuggingFace

👉 [manireddys/financial-qa-tinyllama](https://huggingface.co/manireddys/financial-qa-tinyllama)

---

## 📋 Dataset

👉 [virattt/financial-qa-10K](https://huggingface.co/datasets/virattt/financial-qa-10K)

---

## 👤 Author

**Mani Reddy**
- HuggingFace: [manireddys](https://huggingface.co/manireddys)
- GitHub: GitHub: [mani7204mani]([https://github.com/your_github_username](https://github.com/mani7204mani))
