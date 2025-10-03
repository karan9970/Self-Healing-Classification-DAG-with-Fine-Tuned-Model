# Self-Healing Classification DAG with Fine-Tuned Model

A robust sentiment classification system that combines DistilBERT fine-tuning with LoRA (Low-Rank Adaptation) and LangGraph's self-healing workflow for human-in-the-loop classification.

## Features

- **LoRA Fine-Tuning**: Efficient parameter-efficient fine-tuning of DistilBERT
- **Self-Healing DAG**: LangGraph-based workflow with confidence checking
- **Fallback Mechanism**: Human-in-the-loop for low-confidence predictions
- **Comprehensive Logging**: Detailed logs of predictions and corrections
- **CLI Interface**: User-friendly command-line interface

## Architecture

Input Text
↓
[InferenceNode] → Predict sentiment with confidence
↓
[ConfidenceCheckNode] → Check if confidence >= threshold
↓
├─ High Confidence → [FinalizeNode] → Output
└─ Low Confidence → [FallbackNode] → Ask user → [FinalizeNode] → Output




## Installation

1. Clone the repository
2. Install dependencies:



## Usage

### 1. Fine-Tune the Model



This will:
- Download the IMDB dataset
- Fine-tune DistilBERT using LoRA
- Save the model to `./fine_tuned_model/`
- Generate training metrics

**Training Time**: ~30-45 minutes on GPU (5000 samples)

### 2. Run Classification (Interactive Mode)


Or use the enhanced CLI:


### 3. Advanced CLI Usage

**Single text classification:**


**Batch classification:**

**View statistics:**


**Custom threshold:**



## Example Interaction

Enter movie review: The movie was painfully slow and boring.

[InferenceNode] Predicted label: Positive | Confidence: 54%
[ConfidenceCheckNode] Confidence too low (< 65%). Triggering fallback...

[FallbackNode] Requesting user clarification...

Prediction probabilities:

Negative: 46%

Positive: 54%

 I'm not very confident about this prediction.
My guess was: Positive

Could you clarify the sentiment of your text?
Options:

Positive

Negative

Use original prediction

Your choice (1/2/3): 2

[FallbackNode] Final label: Negative (Corrected via user clarification)
======================================================================
FINAL RESULT
Input: The movie was painfully slow and boring.
Final Label: NEGATIVE
Final Confidence: 100%
✓ Correction applied via user feedback