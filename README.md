ChainMind – Multi-Agent LLM System
Reasoning · Debate · Consensus (Phase 1)

OVERVIEW
ChainMind is a multi-agent Large Language Model (LLM) reasoning system that simulates human-like collaborative thinking.
Instead of relying on a single model response, multiple role-based agents independently reason, debate, and iteratively converge on a final consensus answer.

This repository contains Phase 1, which focuses on building the core multi-agent reasoning architecture.

PHASE 1 OBJECTIVES

Implement a multi-agent reasoning framework

Assign dynamic agent roles (analytical, critical, creative, etc.)

Enable iterative debate cycles

Generate a unified consensus output

Maintain contextual memory across iterations

Design a modular system for future expansion

PROJECT STRUCTURE

ChainMind_Multi_Agent_Model/

core/
Core agent logic and consensus handling

dataset/
Domain samples and reasoning data

input_handlers/
Text, Image, and Audio input handling

utils/
Helper utilities and evaluation metrics

main.py
Main execution file

.gitignore
Ignored files such as environment files and caches

README.txt
Project documentation

KEY CONCEPTS IMPLEMENTED

Multi-Agent Reasoning
Multiple agents generate independent responses to reduce single-model bias.

Iterative Debate
Agents refine their reasoning across multiple iterations.

Consensus Mechanism
A consensus agent evaluates all agent responses and produces a final, unified answer.

Domain Awareness
Agent roles adapt dynamically based on the detected input domain.

TECHNOLOGIES USED

Python

Large Language Models (LLMs)

Prompt Engineering

Zero-Shot and Chain-of-Thought Reasoning

Modular Agent Architecture

HOW TO RUN (PHASE 1)

Clone the repository
git clone https://github.com/ChavaAjay09/Multi-Agent-LLM-System-Reasoning-Debate-Consensus.git

cd ChainMind_Multi_Agent_Model

(Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Set environment variables
Create a .env file with the following content:
HF_TOKEN=your_huggingface_token_here

Note: The .env file is ignored by Git for security reasons.

Run the system
python main.py
