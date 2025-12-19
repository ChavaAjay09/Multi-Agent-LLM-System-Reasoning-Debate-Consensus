# core/iteration_logic.py

import sys
import os
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.agent import Agent
from input_handlers.domain_classifier import DomainClassifier

class IterationManager:
    """
    Manages a multi-turn, emotionally intelligent debate between agents.
    Each agent responds with meaningful, empathetic, and reasoned statements.
    """

    def __init__(self, agents: list[Agent]):
        if not agents:
            raise ValueError("Agent list cannot be empty.")
        self.agents = agents
        self.domain_classifier = DomainClassifier()
        self.conversation_history = []

    def run_debate(self, initial_input: str, num_iterations: int = 2):
        """
        Run a debate for a given number of iterations.
        Each iteration: every agent responds to the last statement.
        """
        print("=" * 80)
        print("🚀 STARTING NEW DEBATE")
        print("=" * 80)
        print(f"INITIAL TOPIC: \"{initial_input}\"\n")

        # 1️⃣ Classify domain once
        print("🔍 Classifying debate domain...")
        domain, confidence, reason = self.domain_classifier.classify(initial_input)
        print(f"✅ Domain classified as '{domain.upper()}' (Confidence: {confidence:.2f})\n")

        self.conversation_history.append({"agent": "Topic", "response": initial_input})

        current_input = initial_input

        for i in range(num_iterations):
            print("-" * 80)
            print(f"🗣️  ITERATION {i + 1}")
            print("-" * 80)

            for agent in self.agents:
                # Create prompt context for emotionally aware response
                prompt_context = (
                    f"Your turn to speak in the debate.\n"
                    f"The last statement was: \"{current_input}\"\n"
                    f"As a {agent.role} agent, provide a thoughtful, emotionally aware, "
                    f"and meaningful counterpoint or perspective in 2-3 sentences."
                )

                print(f"👤 Agent '{agent.name}' ({agent.role.capitalize()}) is thinking...")

                # Agent responds
                response_data = agent.respond(
                    domain=domain,
                    input_text=prompt_context,
                    max_new_tokens=150  # Allow rich, multi-sentence responses
                )

                response_text = response_data["response"]
                print(f"💬 Response: {response_text}\n")

                # Save response in conversation history
                self.conversation_history.append({
                    "agent": agent.name,
                    "role": agent.role,
                    "response": response_text
                })

                # Update current input for next agent
                current_input = response_text

                time.sleep(1)  # Slight pause for readability

        print("=" * 80)
        print("🏁 DEBATE CONCLUDED")
        print("=" * 80)
        return self.conversation_history
