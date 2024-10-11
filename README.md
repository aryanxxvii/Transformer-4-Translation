# 'Attention Is All You Need' Paper Implementation
Link to paper: https://arxiv.org/pdf/1706.03762]

## Architecture
The model is implemented in `model.py` and consists of the following components:
- **Input Embeddings**: Converts input tokens into dense vectors.
- **Positional Encoding**: Adds information about the position of tokens in the sequence.
- **Scaled Multi-Head Attention**: Implements self-attention and cross-attention mechanisms.
- **Feed Forward Block**: Processes the output of the attention layer.
- **Encoder and Decoder Stacks**: Stacks of layers for encoding and decoding sequences.
- **Output Projection Layer**: Projects the decoder output to the vocabulary size for prediction.

<img src="https://github.com/user-attachments/assets/97f2c28e-1ae5-4635-9a5a-182d58a6f787" width="400">

### Configuration
The configuration for the model training and parameters is specified in config.py. Key parameters include:

- `batch_size`: Number of samples per gradient update.
- `num_epochs`: Total number of training epochs.
- `lr`: Learning rate for the optimizer.
- `seq_len`: Maximum sequence length for input sentences.
- `d_model`: Dimensionality of the model's output.
- `datasource`: Name of the dataset being used (e.g., `opus_books`).
- `lang_src`: Source language code (e.g., `"en"` for English).
- `lang_tgt`: Target language code (e.g., `"es"` for Spanish).
## Dataset
The dataset used is the **OPUS Books** dataset, which is a collection of bilingual texts aligned across multiple languages. It provides a rich resource for training translation models, featuring copyright-free books in various languages.

## Training
To train the model, ensure you have the dataset and tokenizers ready. Run the training script:
```bash
python train.py
```



