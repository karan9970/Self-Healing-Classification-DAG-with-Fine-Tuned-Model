class ConfidenceNode:
    def __init__(self, threshold=0.65):
        self.threshold = threshold

    def run(self, state: dict):
        """Check confidence score"""
        confidence = state["confidence"]
        if confidence < self.threshold:
            state["fallback_triggered"] = True
        else:
            state.update({
                "fallback_triggered": False,
                "final_label": state["predicted_label"],
                "final_confidence": confidence,
                "correction_applied": False
            })
        return state
