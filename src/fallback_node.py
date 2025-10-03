class FallbackNode:
    def run(self, state: dict):
        """Ask user for clarification when confidence is low"""
        predicted_label = state["predicted_label"]
        print("\n🤔 I'm not very confident about this prediction.")
        print(f"My guess was: {predicted_label.capitalize()}")
        print("Options:\n  1. Positive\n  2. Negative\n  3. Use original prediction")

        user_choice = input("Your choice (1/2/3): ").strip()
        if user_choice == "1":
            final_label = "positive"
            clarification = "User confirmed: Positive"
            correction = final_label != predicted_label
        elif user_choice == "2":
            final_label = "negative"
            clarification = "User confirmed: Negative"
            correction = final_label != predicted_label
        else:
            final_label = predicted_label
            clarification = "User accepted original prediction"
            correction = False

        state.update({
            "user_clarification": clarification,
            "final_label": final_label,
            "final_confidence": 1.0 if correction else state["confidence"],
            "correction_applied": correction
        })
        return state
