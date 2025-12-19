# consensus_agent.py

from core.model_loader import ModelLoader

class ConsensusAgent:
    """
    Generates a consensus summary after a multi-agent debate.
    Reads the full conversation history and produces a concise conclusion.
    """

    def __init__(self):
        self.model_loader = ModelLoader()

    def generate_consensus(self, conversation_history, max_new_tokens=150):
        """
        Produces a consensus summary from the conversation history.
        Args:
            conversation_history (list): List of dicts with keys 'agent', 'role', 'response'
            max_new_tokens (int): Max tokens for model generation
        Returns:
            str: Consensus summary
        """
        if not conversation_history:
            return "No conversation history provided."

        # Build prompt from conversation history
        history_text = ""
        for entry in conversation_history:
            agent = entry.get("agent", "Unknown")
            role = entry.get("role", "")
            response = entry.get("response", "")
            history_text += f"{agent} ({role}): {response}\n"

        prompt = (
            "You are a neutral summarizer. The following is a debate between multiple agents.\n"
            "Your task is to provide a concise consensus summary capturing the key insights, "
            "agreements, disagreements, and important recommendations.\n\n"
            f"Debate Transcript:\n{history_text}\nConsensus Summary:"
        )

        try:
            consensus_text = self.model_loader.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            return consensus_text.strip()
        except Exception as e:
            return f"[Error generating consensus: {str(e)}]"


# --- Example usage ---
if __name__ == "__main__":
    from core.iteration_logic import IterationManager
    from core.agent import Agent

    # Mock conversation for testing
    agents = [
        Agent(name="Alex", role="analytical"),
        Agent(name="Casey", role="creative"),
        Agent(name="Jordan", role="critical"),
    ]
    iteration_manager = IterationManager(agents)
    topic = "Should AI assist in mental health therapy?"
    full_history = iteration_manager.run_debate(topic, num_iterations=1)

    consensus_agent = ConsensusAgent()
    summary = consensus_agent.generate_consensus(full_history)
    print("\n📌 Consensus Summary:\n")
    print(summary)
