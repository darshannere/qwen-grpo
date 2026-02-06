# Medical Chain-of-Thought Reasoning with GRPO

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A medical reasoning system that enhances Large Language Models with Chain-of-Thought (CoT) reasoning using Group Relative Policy Optimization (GRPO). This project addresses the critical need for transparent, step-by-step reasoning in medical AI applications.

## 📋 Overview

Large Language Models struggle with precision and reasoning in medical contexts where accuracy can be life-critical. This project develops a 3B parameter reasoning model specifically for medical applications, focusing on:

- **Logical Reasoning**: Breaking down complex medical tasks into clear, sequential steps
- **Context Management**: Maintaining relevant patient information across multi-turn conversations
- **Transparency**: Providing interpretable justification for each medical recommendation

### Key Achievements

- **+11% accuracy improvement** over supervised fine-tuning baseline
- **+8.9% average accuracy** across three medical QA datasets
- **-4.6 average perplexity reduction** indicating more confident predictions
- Efficient training on consumer hardware (single T4/A5000 GPU)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Qwen2.5-3B Base Model              │
│                   (3B Parameters)                    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   LoRA 4-bit Adapters  │
         │   (5-10% of params)    │
         └────────┬───────────────┘
                  │
         ┌────────▼────────┐
         │  SFT Training   │
         │  (Baseline)     │
         └────────┬────────┘
                  │
         ┌────────▼─────────────────────────┐
         │     GRPO Fine-tuning             │
         │  ┌──────────────────────────┐   │
         │  │   Reward Functions:      │   │
         │  │  • Semantic Similarity   │   │
         │  │  • Format Compliance     │   │
         │  │  • Answer Matching       │   │
         │  │  • XML Structure Count   │   │
         │  └──────────────────────────┘   │
         └──────────────────────────────────┘
                  │
                  ▼
      ┌───────────────────────────┐
      │  Reasoning + Final Answer │
      │     (XML Format)          │
      └───────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/darshannere/qwen-grpo.git
cd qwen-grpo

# Install dependencies
pip install torch transformers trl unsloth sentence-transformers accelerate bitsandbytes
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel

# Load the trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path/to/your/model",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# Format your medical question
question = """A 45-year-old patient presents with chest pain, 
shortness of breath, and elevated troponin levels. What is the 
most likely diagnosis?"""

prompt = f"""Please provide a step-by-step reasoning for the following medical question:

Question: {question}

Provide your response in the following format:
<reasoning>
Your detailed chain-of-thought reasoning here...
</reasoning>

<answer>
Your final answer here
</answer>
"""

# Generate response
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

## 📊 Datasets

The model was trained and evaluated on three medical QA datasets:

| Dataset | Size | Description | Use Case |
|---------|------|-------------|----------|
| **medical-o1-reasoning-SFT** | 90,120 samples | Clinical questions with long reasoning chains | Primary training |
| **BigBio-Med-QA** | Varied | Wide range of medical topics | Evaluation |
| **PubMedQA** | Research-based | Evidence-based biomedical questions | Evaluation |

### Data Format

The model expects data in the following XML format:

```xml
<reasoning>
Step 1: Analyze the patient's symptoms...
Step 2: Consider differential diagnoses...
Step 3: Evaluate test results...
</reasoning>

<answer>
Acute Myocardial Infarction
</answer>
```

## 🎯 Training

### Supervised Fine-Tuning (SFT)

```python
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Load base model with LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# Train with SFT
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

trainer.train()
```

### GRPO Training

```python
from trl import GRPOTrainer

# Define reward function
def compute_reward(generated_text, reference_text):
    """
    Combined reward = 0.42 × Semantic Similarity 
                    + 0.15 × Format Compliance 
                    + 0.29 × Answer Matching 
                    + 0.15 × XML Count
    """
    semantic_sim = compute_semantic_similarity(generated_text, reference_text)
    format_score = check_xml_format(generated_text)
    answer_match = check_answer_correctness(generated_text, reference_text)
    xml_count = count_xml_tags(generated_text)
    
    return (0.42 * semantic_sim + 
            0.15 * format_score + 
            0.29 * answer_match + 
            0.15 * xml_count)

# Train with GRPO
grpo_trainer = GRPOTrainer(
    model=model,
    reward_function=compute_reward,
    args=grpo_args,
)

grpo_trainer.train()
```

## 📈 Results

### Performance Comparison

| Dataset | Model | Accuracy (%) | Perplexity |
|---------|-------|-------------|-----------|
| **Base Test** | Baseline | 56.0 | - |
| | Baseline + GRPO | **70.0** | - |
| **BioMedQA** | Baseline | 52.0 | - |
| | Baseline + GRPO | **56.4** | - |
| **PubMedQA** | Baseline | 47.0 | - |
| | Baseline + GRPO | **56.2** | - |

### Baseline Comparisons

| Approach | Test Accuracy |
|----------|--------------|
| Zero-shot CoT | 35% |
| Few-shot CoT (5 examples) | 42% |
| SFT Baseline | 56% |
| **SFT + GRPO (Ours)** | **67%** |

### Evaluation Metrics

1. **LLM-as-Judge**: Gemini 2.0 Flash evaluates logical reasoning and medical correctness
2. **Perplexity**: Measures model confidence and fluency
3. **Human Evaluation**: Manual assessment of answer correctness and reasoning clarity

## 🔬 Key Components

### Reward Functions

The GRPO training uses four complementary reward signals:

1. **Semantic Similarity (42% weight)**
   - Uses Sentence Transformer (all-MiniLM-L6-v2)
   - Measures cosine similarity between generated and reference CoT
   - Ensures logical consistency with the question

2. **Format Compliance (15% weight)**
   - Verifies presence of `<reasoning>` and `<answer>` XML tags
   - Ensures structural correctness for downstream evaluation

3. **Answer Matching (29% weight)**
   - Direct comparison of final answer with ground truth
   - Binary reward for correctness

4. **XML Count (15% weight)**
   - Weighted score based on XML structure completeness
   - Encourages proper formatting throughout

### Model Architecture

- **Base Model**: Qwen2.5-3B-Instruct (3.09B parameters)
- **Training Method**: LoRA 4-bit quantization
- **Trainable Parameters**: 5-10% of base model
- **Context Window**: 512 tokens
- **Hardware Requirements**: Single T4 or A5000 GPU

## 🎓 Related Work

This project builds upon several key research directions:

- **Chain-of-Thought Prompting** (Wei et al., 2022): Foundation for structured reasoning
- **Medical Question Answering** (Gramopadhye et al., 2024): Open-ended clinical scenarios
- **Diagnostic Reasoning** (Wu et al., 2023): DR-CoT for diagnostic accuracy
- **JMLR** (Wang et al., 2024): Retrieval-augmented medical reasoning
- **DiReCT** (Wang et al., 2024): Diagnostic reasoning from clinical notes

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- [ ] Extend to multimodal inputs (images, lab reports)
- [ ] Compare performance across different model sizes
- [ ] Implement adaptive reward weighting
- [ ] Add real-time inference optimization
- [ ] Develop clinical decision-support interfaces

## 👥 Team

- **Darshan Nere** - SFT baseline implementation and GRPO training
- **Vikrant Bhati** - GRPO implementation and training
- **Ishani Kohli** - Data processing and reward function design
- **Eeshan Umrani** - Multi-dataset evaluation
- **Sarvesh Chakradeo** - LLM-as-Judge evaluation

Virginia Tech, Department of Computer Science


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Qwen team for the base model
- Hugging Face for datasets and transformers library
- Unsloth for efficient LoRA training utilities
- TRL (Transformer Reinforcement Learning) library

## 📧 Contact

For questions or collaborations:
- Darshan Nere - darshannere@vt.edu
- Project Repository: [github.com/darshannere/qwen-grpo](https://github.com/darshannere/qwen-grpo)

---

**Note**: This model is for research purposes only and should not be used for actual medical diagnosis without validation by healthcare professionals.


