# ChainMind: Project Details (Phase 1)

## 1. Project Title
**ChainMind - Multi-Agent AI Reasoning Framework**

---

## 2. The Core Concept
**ChainMind** is an advanced AI system that moves beyond single-model responses by utilizing a **Multi-Agent "Round Table" architecture**.

* **Traditional AI (e.g., ChatGPT):**
    * One model receives a prompt.
    * Generates a single, immediate answer.
    * *Risk:* Prone to "hallucinations" (making things up) and errors in logic.
    
* **ChainMind System:**
    * Multiple specialized AI Agents receive the prompt.
    * They debate, critique, and refine the answer.
    * *Result:* A verified, consensus-based conclusion that is logically sound.

---

## 3. The Problem Statement
Current Large Language Models (LLMs) face three major issues that this project solves:
1.  **Hallucination:** Models often generate convincing but factually incorrect information.
2.  **Reasoning Deficit:** Single models struggle with complex, multi-step logical or mathematical problems.
3.  **Modal Blindness:** Standard text models cannot inherently process audio or visual data.

---

## 4. Proposed Solution (Phase 1 Architecture)
The framework operates on a 4-step workflow:
1.  **User Input:** The system accepts Text, Image (OCR), or Audio inputs.
2.  **Agent 1 (The Proposer):** Generates an initial solution to the problem.
3.  **Agent 2 (The Critic):** Analyzes the Proposer's solution to identify logical flaws.
4.  **Consensus Engine:** Weighs the evidence from both agents to produce a final, high-confidence output.

---

## 5. Technical Folder Structure
The codebase is organized into modular components:

### `core/` (The Brain)
* **`agent.py`:** Defines the `Agent` class. Handles model loading (Mistral-7B) and prompt generation.
* **`consensus_agent.py`:** The "Judge" module. Implements algorithms to compare agent outputs and calculate a confidence score.
* **`model_loader.py`:** Manages efficient model loading using **Quantization** (4-bit/8-bit) to ensure performance on consumer hardware.

### `input_handlers/` (The Senses)
* **`audio_input.py`:** Uses speech-to-text libraries to convert spoken queries into machine-readable text.
* **`image_input.py`:** Integrates Optical Character Recognition (OCR) to extract text and context from images.
* **`domain_classifier.py`:** A routing utility that detects if a question is "Math", "Coding", or "General" to assign the right agent.

### `dataset/` (The Training)
* **`chainmind_debate_dataset.json`:** A custom-curated dataset containing complex reasoning problems used to fine-tune the agents' debate capabilities.

---

## 6. Technology Stack
* **Programming Language:** Python 3.10+
* **Primary Model:** Mistral-7B (v0.1)
* **Fine-Tuning Technique:** LoRA (Low-Rank Adaptation) for memory efficiency.
* **Key Libraries:** * `transformers` (Hugging Face ecosystem)
    * `torch` (PyTorch for backend processing)
    * `bitsandbytes` (For model quantization)
    * `peft` (For parameter-efficient fine-tuning)

---

## 7. Current Status (Phase 1 Achievements)
* ✅ **Architecture Setup:** Modular code structure established and pushed to GitHub.
* ✅ **Security:** Repository secured (no API keys/secrets in history).
* ✅ **Agent Implementation:** Basic Agent and Consensus classes are functional.
* ✅ **Input Processing:** Multimodal input handlers (Audio/Image) are integrated.

---

## 8. Future Roadmap (Phase 2)
* Implement the full **iterative debate loop** (where agents talk back and forth).
* Develop a **Web UI** using Streamlit for easy user interaction.
* Integrate **RAG (Retrieval-Augmented Generation)** to allow agents to look up real-time facts.