# core/model_loader.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import os

class ModelLoader:
    """
    Singleton-like class to load the base LLM and optional LoRA adapter.
    Handles device placement, offloading, and safe generation.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 base_model_path="/content/drive/MyDrive/chainmind/model_loader_mistral",
                 lora_adapter_path="/content/drive/MyDrive/chainmind/LoRA Adapter/checkpoint-400",
                 device="cuda"):
        if hasattr(self, "model"):
            return  # Already initialized

        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"▶️ Loading Mistral-7B base model from {base_model_path} on {self.device}...")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

            # Load base model in 8-bit mode to save GPU memory
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                load_in_8bit=True,
                offload_folder="/tmp/transformers_offload"
            )

            # Apply LoRA adapter if provided
            if lora_adapter_path and os.path.exists(lora_adapter_path):
                print(f"▶️ Applying LoRA adapter from {lora_adapter_path}...")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    lora_adapter_path,
                    device_map="auto",
                    offload_folder="/tmp/peft_offload",
                    offload_state_dict=True
                )

            self.model.eval()
            print("✅ Model and LoRA adapter loaded successfully.")

        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model or LoRA adapter: {e}")

    @torch.inference_mode()
    def generate_text(self, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, do_sample=True):
        """
        Generates text from the model given a prompt.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            output = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # Remove the prompt from the output to keep only the generated text
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text
        except Exception as e:
            print(f"❌ Error during text generation: {e}")
            return "[Generation Failed]"
