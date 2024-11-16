
# Sentiment Analysis with Custom LSTM and Transformer Architectures

## Project Overview
This project focuses on sentiment analysis using the Amazon Polarity dataset. Two custom architectures—LSTM and Transformer—have been implemented and evaluated. The goal is to compare their performance and showcase modern approaches to text classification. Fine-tuning BERT has also been explored for comparison with these custom architectures.

## Files and Structure
- **`lstm.py`**: Defines a custom LSTM model with gates and integration of attention mechanisms.
- **`transformer.py`**: Implements core transformer components, including:
  - Multi-head attention
  - Layer normalization
  - Position-wise feed-forward layers
- **`main.ipynb`**: Provides the training pipeline for sentiment analysis. It uses Hugging Face utilities for tokenization, training, and evaluation. Includes code to generate plots and figures for a report.

## Model Architectures
### LSTM
A custom implementation of LSTM with:
- Explicit handling of input, forget, and output gates.
- Optional dropout for regularization.
- Potential integration with encoder-decoder and attention mechanisms.

### Transformer
Implements the essential blocks of transformer models:
- Multi-head self-attention for capturing global dependencies.
- Layer normalization for stable training.
- Position-wise feed-forward networks to model complex interactions.

### Fine-tuning BERT
The Hugging Face BERT model is fine-tuned for sentiment analysis, showcasing its pre-trained power and adaptability to new tasks.

## Dataset
The Amazon Polarity dataset is used for sentiment analysis. It contains positive and negative reviews, providing a binary classification task.

## Training and Evaluation
The training pipeline includes:
- Tokenization using Hugging Face's BERT tokenizer.
- Batch collation and data preprocessing.
- Training LSTM, Transformer, and fine-tuning BERT models.
- Evaluation of model performance using validation accuracy and F1-score.
- Visualizations of learning curves and results.

## Results and Observations

### Summary Table
| Exp. | Architecture                        | Train Time (s) | Eval Time (s) | Best Val Acc | Best Val F1 |
|------|-------------------------------------|----------------|---------------|--------------|-------------|
| 1    | LSTM, no dropout, encoder only     | 1247.09        | 1.11          | 71.2         | 0.7487      |
| 2    | LSTM, dropout, encoder only        | 1255.40        | 1.12          | 78.3         | 0.7704      |
| 3    | LSTM, dropout, enc-dec, no attn    | 1814.56        | 1.55          | 72.7         | 0.7289      |
| 4    | LSTM, dropout, enc-dec, with attn  | 2721.71        | 2.07          | **89.1**     | **0.8790**  |
| 5    | Transformer, 2 layers, pre-norm    | 3517.42        | 4.20          | 83.4         | 0.8091      |
| 6    | Transformer, 4 layers, pre-norm    | 2052.41        | 1.78          | 82.1         | 0.8264      |
| 7    | Transformer, 2 layers, post-norm   | 2057.22        | 1.72          | 86.6         | 0.8553      |
| 8    | Fine-tuning BERT                   | 5305.54        | 9.14          | 86.4         | 0.8577      |

### Key Insights
1. **Efficiency vs. Performance**:
   - **Efficiency**: Experience 1 (LSTM without dropout) is the fastest to train (1247.09s) and evaluate (1.11s).
   - **Performance**: Experience 4 (LSTM with dropout, encoder-decoder, and attention) achieves the best validation accuracy (89.1%) and F1-score (0.879).

2. **Impact of Dropout**:
   - Adding dropout (Exp. 2) improves validation accuracy from 71.2% (Exp. 1) to 78.3%, showing its effectiveness in reducing overfitting.

3. **Encoder-Decoder Setup**:
   - Transitioning to encoder-decoder (Exp. 3) without attention reduces accuracy from 78.3% to 72.7%, suggesting that attention mechanisms are crucial for performance improvement.

4. **Attention Mechanism**:
   - Adding attention (Exp. 4) dramatically boosts performance, increasing accuracy to 89.1%.

5. **Transformer Observations**:
   - While Transformers generally excel in NLP tasks, their performance here is close to LSTMs. This may be due to the dataset's limited size and the models' small configurations.

6. **Fine-tuning BERT**:
   - Fine-tuning BERT shows promising results but is limited by fewer epochs due to time constraints. The model has more stable and better generalization trends over extended training.

### GPU Memory Usage
- **LSTMs (Exp. 1-4)**: Low memory consumption, slightly higher with dropout and attention.
- **Transformers (Exp. 5-7)**: Moderate memory usage, increasing with the number of layers.
- **Fine-tuning BERT (Exp. 8)**: High memory consumption due to the model's size.

### Learning Curve Observations
- **LSTMs**: Stable learning, less prone to overfitting.
- **Transformers**: Faster convergence but higher instability, risking overfitting.
- **Fine-tuning BERT**: Slow convergence due to the model's size, requiring more epochs.

### Recommendations
- Use **Experience 1** for time-sensitive applications.
- Choose **Experience 4** for the best generalization.
- For larger datasets, Transformers and fine-tuned BERT are likely to outperform due to their scalability.

## Usage
1. **Install Dependencies**:
   ```bash
   pip install torch transformers datasets matplotlib
   ```
2. **Run Training and Evaluation**:
   Use the Jupyter Notebook `main (1).ipynb` to execute the training pipeline and generate plots.

3. **Modify Architectures**:
   The `lstm.py` and `transformer.py` files can be adapted to explore different configurations.

--- 
