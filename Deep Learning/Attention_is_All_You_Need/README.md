[Access RP](https://arxiv.org/abs/1706.03762)

1. Big Picture:
   - Transformer is a new model for translating text
   - It's different because it doesn't use traditional methods (no recurrence or convolution)
   - Uses "attention" to understand relationships between words

2. Model Structure:
   - Has an encoder (for input) and decoder (for output)
   - Both have 6 identical layers stacked on top of each other
   - Key innovation: Multi-Head Attention

3. Multi-Head Attention:
   - Allows the model to focus on different parts of the input at the same time
   - Uses 8 "attention heads" working in parallel
   - Each head learns to focus on different aspects of the language

   Code snippet:
   ```python
   class MultiHeadAttention(nn.Module):
       def __init__(self, d_model=512, num_heads=8):
           # Set up 8 parallel attention mechanisms
   ```

4. How Attention Works:
   - Uses three concepts: Queries, Keys, and Values (Q, K, V)
   - Simplified: Q asks a question, K holds possible answers, V gives the actual information
   - Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
   - The sqrt(d_k) part helps keep the numbers from getting too extreme

5. Position-wise Feed-Forward Networks:
   - After attention, each position goes through its own tiny neural network
   - This helps the model learn more complex patterns
   - Uses two linear transformations with a ReLU in between

   Code:
   ```python
   class PositionwiseFeedForward(nn.Module):
       def __init__(self, d_model=512, d_ff=2048):
           # Set up two linear layers with ReLU activation
   ```

6. Positional Encoding:
   - Problem: The model doesn't naturally understand word order
   - Solution: Add special codes to each word to indicate its position
   - Uses sine and cosine functions to create these codes
   - This allows the model to understand relative positions of words

7. Training Details:
   - Uses Adam optimizer with a custom learning rate schedule
   - Starts with a "warm-up" period where learning rate increases
   - Then gradually decreases the learning rate

   Learning rate formula:
   ```python
   lrate = d_model**-0.5 * min(step_num**-0.5, step_num * warmup_steps**-1.5)
   ```

8. Why It's Cool:
   - Can process all words in parallel (very fast!)
   - Connects all words directly (helps with long-range dependencies)
   - Produces attention patterns that we can visualize and interpret

9. Results:
   - Achieves state-of-the-art results in translation tasks
   - Trains much faster than previous models

