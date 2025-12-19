
# ===============================
# STEP 1: Mount Google Drive
# ===============================

from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('pip install -U torch transformers accelerate peft datasets bitsandbytes sentencepiece huggingface_hub')


pip install openai-whisper pytesseract pillow ffmpeg


from huggingface_hub import login

import os
HF_TOKEN = os.getenv("HF_TOKEN")


# ==========================================
# STEP 4: Download Mistral model to Drive
# ==========================================
from huggingface_hub import snapshot_download

model_repo = "mistralai/Mistral-7B-Instruct-v0.2"
save_dir = "/content/drive/MyDrive/chainmind/model_loader_mistral"

snapshot_download(
    repo_id=model_repo,
    local_dir=save_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"✅ Model successfully downloaded to: {save_dir}")



from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

# Path to your local model
local_model_path = "/content/drive/MyDrive/chainmind/model_loader_mistral"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Quantization config for GPU + CPU offload
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                        # Use 8-bit quantization
    llm_int8_enable_fp32_cpu_offload=True     # Offload large layers to CPU when needed
)

# Folder for temporary offload
offload_folder = "/content/chainmind_offload"
os.makedirs(offload_folder, exist_ok=True)

# Load model on GPU with offloading enabled
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    quantization_config=bnb_config,
    device_map="auto",               # Automatically map layers to GPU/CPU
    offload_folder=offload_folder
)

print("✅ Mistral-7B-Instruct loaded successfully on GPU!")

# Generation example
prompt = "Explain why reinforcement learning is useful for autonomous vehicles."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Send tensors to GPU
outputs = model.generate(**inputs, max_new_tokens=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))




import json
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -----------------------------
# 1. Load dataset
# -----------------------------
dataset_path = "/content/drive/MyDrive/chainmind/dataset/domain_classifier_dataset.json"
with open(dataset_path, "r") as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels = [item['domain'] for item in data]  # Use 'domain' key

# Map labels to integers
label_set = sorted(list(set(labels)))
label2id = {label: i for i, label in enumerate(label_set)}
id2label = {i: label for label, i in label2id.items()}
labels = [label2id[label] for label in labels]

# -----------------------------
# 2. Create PyTorch Dataset
# -----------------------------
class DomainDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -----------------------------
# 3. Load DistilBERT tokenizer and model
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label_set),
    id2label=id2label,
    label2id=label2id
)

# -----------------------------
# 4. Prepare datasets
# -----------------------------
dataset = DomainDataset(texts, labels, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# -----------------------------
# 5. Training arguments (old-compatible)
# -----------------------------
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/chainmind/DistilBERT",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='/content/drive/MyDrive/chainmind/DistilBERT/logs',
    logging_steps=1,  # Log every step
    save_total_limit=2,
    do_eval=True
)

# -----------------------------
# 6. Metrics function
# -----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# -----------------------------
# 7. Custom callback for step-wise loss
# -----------------------------
class StepLossCallback(TrainerCallback):
    def __init__(self):
        self.epoch_loss = 0
        self.steps = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            self.epoch_loss += logs['loss']
            self.steps += 1
            print(f"Step {state.global_step}, Training Loss: {logs['loss']:.4f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.steps > 0:
            avg_loss = self.epoch_loss / self.steps
            print(f"Epoch {state.epoch} finished, Average Loss: {avg_loss:.4f}")
        self.epoch_loss = 0
        self.steps = 0

# -----------------------------
# 8. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[StepLossCallback]
)

# -----------------------------
# 9. Train and save model
# -----------------------------
trainer.train()
trainer.save_model("/content/drive/MyDrive/chainmind/DistilBERT")
tokenizer.save_pretrained("/content/drive/MyDrive/chainmind/DistilBERT")

# -----------------------------
# 10. Testing / Inference
# -----------------------------
from transformers import pipeline
classifier = pipeline(
    "text-classification",
    model="/content/drive/MyDrive/chainmind/DistilBERT",
    tokenizer="/content/drive/MyDrive/chainmind/DistilBERT",
    top_k=None  # return all scores
)

# Test examples
test_texts = [
    "The doctor recommended a new treatment for diabetes.",
    "The court issued a ruling in favor of the plaintiff.",
    "We had a fun day at the park."
]

for text in test_texts:
    result = classifier(text)
    print(f"\nText: {text}")
    print(f"Prediction: {result}")




classifier = DomainClassifier("/content/drive/MyDrive/chainmind/DistilBERT")
print(classifier.classify("The doctor prescribed antibiotics after the surgery."))



# ======================================
# LoRA Fine-Tuning for Mistral-7B-Instruct (Fixed)
# Dataset: chainmind_debate_dataset.json
# Output: /content/drive/MyDrive/chainmind/LoRA Adapter
# ======================================
import torch, json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------
# 1. Paths and Model Loading
# ---------------------------
MODEL_PATH = "/content/drive/MyDrive/chainmind/model_loader_mistral"
DATA_PATH = "/content/drive/MyDrive/chainmind/dataset/chainmind_debate_dataset.json"
OUTPUT_PATH = "/content/drive/MyDrive/chainmind/LoRA Adapter"

# ---- Use new quantization config (no warnings) ----
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ---- Load tokenizer and model ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# ---------------------------
# 2. Prepare Dataset
# ---------------------------
def load_debate_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for entry in data:
        topic = entry["topic"]
        for d in entry["debate"]:
            role = d["agent_role"]
            thought = d["thought"]
            response = d["response"]
            prompt = f"[ROLE: {role.upper()}]\nTopic: {topic}\nThought: {thought}\nResponse:"
            samples.append({"prompt": prompt, "response": response})
    return samples

dataset_list = load_debate_data(DATA_PATH)
dataset = Dataset.from_list(dataset_list)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ---- Fix the mapping issue (handles lists safely) ----
def tokenize_fn(example):
    prompt = example["prompt"]
    response = example["response"]

    # handle cases where prompt/response might be lists
    if isinstance(prompt, list):
        prompt = " ".join(prompt)
    if isinstance(response, list):
        response = " ".join(response)

    text = f"{prompt} {response}"
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn, batched=False, remove_columns=["prompt", "response"])

# ---------------------------
# 3. Apply LoRA Configuration
# ---------------------------
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------
# 4. Training Configuration
# ---------------------------
from packaging import version
import transformers

transformers_version = transformers.__version__
print(f"Transformers version: {transformers_version}")

# ✅ Universal compatible TrainingArguments (works for 4.31 – 4.57+)
if version.parse(transformers_version) >= version.parse("4.55.0"):
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        max_steps=500,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        do_eval=True,              # <-- new flag replaces evaluation_strategy
    )
else:
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        max_steps=500,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",  # older syntax
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )

# ---------------------------
# 5. Start Training
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()

# ---------------------------
# 6. Save LoRA Adapter
# ---------------------------
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"✅ LoRA adapter saved successfully to: {OUTPUT_PATH}")



import sys
import os

# Update this path to your project root
project_root = "/content/drive/MyDrive/chainmind"

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("✅ Project root added to sys.path:", project_root)




import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
# Install required NLP and metrics packages
get_ipython().system('pip install rouge-score textblob nltk transformers torch bert-score')




# test_combined_metrics.py

import sys
import importlib

# Ensure project root path
sys.path.append("/content/drive/MyDrive/chainmind")

# Import and reload to pick latest changes
import utils.combined_metrics as cm
importlib.reload(cm)

def test_combined_metrics():
    print("\n🚀 Running CombinedMetrics Evaluation Test...")

    # 🧠 Simulated debate transcript
    transcript = [
        {"agent": "Alex", "response": "AI can help therapists by analyzing behavioral patterns and providing early warning signs."},
        {"agent": "Casey", "response": "AI is not just analysis; it's an artist's brush painting human emotions into data."},
        {"agent": "Jordan", "response": "But relying on AI might distance people from real empathy, which no machine can replace."},
        {"agent": "Alex", "response": "Still, if used carefully, AI can make therapy more accessible and affordable."},
        {"agent": "Casey", "response": "It’s about synergy — the blend of human empathy and machine precision."},
        {"agent": "Jordan", "response": "Ethical oversight is key, ensuring AI tools do not dictate but assist human therapists."}
    ]

    # 🧾 Consensus summary
    consensus_summary = (
        "AI should support mental health professionals, not replace them, "
        "ensuring accessibility while preserving empathy and human connection."
    )

    # Initialize metrics evaluator
    metrics = cm.CombinedMetrics()

    # Run evaluation
    results = metrics.evaluate(transcript, consensus_summary)

    # ✅ Validate key metrics
    expected_metrics = [
        "Avg Sentence Length",
        "Lexical Diversity",
        "Perplexity",
        "BLEU",
        "ROUGE-1",
        "ROUGE-L",
        "BERTScore-F1"
    ]
    for metric in expected_metrics:
        assert metric in results, f"Missing metric: {metric}"

    print("\n✅ CombinedMetrics test completed successfully.")
    print("\n📊 Test Metrics Summary:")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    test_combined_metrics()




get_ipython().system('touch input_handlers/__init__.py core/__init__.py utils/__init__.py')



# main.py

import sys
import os
import time

# --- Path Fix ---
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------

# --- Import project modules ---
from input_handlers.domain_classifier import DomainClassifier
from input_handlers.audio_input import AudioHandler
from input_handlers.image_input import handle_image_input
from core.agent import Agent
from core.iteration_logic import IterationManager
from utils.combined_metrics import CombinedMetrics  # ✅ Metrics integration
from core.consensus_agent import ConsensusAgent  # ✅ Consensus integration

# --- Color class for terminal display ---
class Colors:
    HEADER = "\033[95m"; BLUE = "\033[94m"; CYAN = "\033[96m"
    GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
    END = "\033[0m"; BOLD = "\033[1m"; GREY = "\033[90m"

def ensure_nltk():
    """Download NLTK data once if missing (no-op if present)."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    # Newer NLTK splits out punkt tables:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass  # old NLTK versions won't have this; safe to ignore

def run_application():
    ensure_nltk()

    print("✅ All modules imported and ready.")
    print("Initializing core components...")
    classifier = DomainClassifier()
    audio_handler = AudioHandler()
    print("✅ Components initialized.")

    print(f"\n{Colors.BOLD}{Colors.HEADER}🧠 ChainMind: Multi-Agent Reasoning System{Colors.END}")
    print("=" * 40)

    # --- 1️⃣ Get user input ---
    print(f"{Colors.HEADER}\n📥 Choose input type:{Colors.END}")
    print("1. Text")
    print("2. Audio")
    print("3. Image")
    choice = input("Enter option (1/2/3): ").strip()

    input_text = None
    if choice == '1':
        input_text = input(f"{Colors.CYAN}Please enter your topic:\n> {Colors.END}").strip()
    elif choice == '2':
        input_text = audio_handler.upload_and_transcribe()
    elif choice == '3':
        input_text = handle_image_input()
    else:
        print(f"{Colors.RED}❌ Invalid choice.{Colors.END}")
        return

    if not input_text or not input_text.strip():
        print(f"{Colors.RED}⚠️ No valid input provided.{Colors.END}")
        return

    # --- 2️⃣ Classify domain ---
    domain, _, _ = classifier.classify(input_text)
    print(f"\n{Colors.CYAN}📊 Initializing agents for a '{domain.upper()}' debate...{Colors.END}")

    # --- 3️⃣ Initialize agents ---
    agents = [
        Agent(name="Alex", role="analytical"),
        Agent(name="Casey", role="creative"),
        Agent(name="Jordan", role="critical"),
    ]
    iteration_manager = IterationManager(agents=agents)

    print(f"{Colors.GREEN}\n🤖 Agents ready. Starting debate...{Colors.END}")
    time.sleep(0.5)

    # --- 4️⃣ Run debate ---
    final_history = iteration_manager.run_debate(initial_input=input_text, num_iterations=2)

    # --- 5️⃣ Display full transcript ---
    print(f"\n{Colors.BOLD}{Colors.BLUE}📝 Full Debate Transcript:{Colors.END}\n")
    for i, entry in enumerate(final_history):
        agent_name = entry['agent']
        role = entry.get('role', 'topic')
        response = entry['response']

        if agent_name == "Topic":
            color = Colors.YELLOW; emoji = "📌"; role_display = ""
        else:
            color = Colors.GREEN if role == "analytical" else Colors.CYAN if role == "creative" else Colors.RED
            emoji = "🧠" if role == "analytical" else "🎨" if role == "creative" else "⚖️"
            role_display = f" ({role.capitalize()})"

        print(f"{color}{i}. {emoji} {agent_name}{role_display}:{Colors.END} {response}\n")

    print("-" * 40)

    # --- 6️⃣ Generate conversation metrics (✅ transcript wired correctly) ---
    print(f"\n{Colors.BOLD}{Colors.HEADER}📈 Evaluating Combined Metrics...{Colors.END}")
    try:
        metrics = CombinedMetrics(transcript=final_history)
        metrics.evaluate()
        metrics.print_summary()
    except Exception as e:
        print(f"{Colors.RED} Combined metrics evaluation failed: {e}{Colors.END}")

    # --- 7️⃣ Generate consensus summary ---
    print(f"\n{Colors.BOLD}{Colors.HEADER}📌 Generating Consensus Summary...{Colors.END}")
    try:
        consensus_agent = ConsensusAgent()
        consensus_summary = consensus_agent.generate_consensus(final_history, max_new_tokens=200)
        print(f"\n{Colors.CYAN}Consensus Summary:{Colors.END}\n{consensus_summary}\n")
    except Exception as e:
        print(f"{Colors.RED}❌ Consensus generation failed: {e}{Colors.END}")

    print(f"{Colors.BOLD}{Colors.GREEN}✅ ChainMind session completed successfully!{Colors.END}\n")

if __name__ == "__main__":
    run_application()




