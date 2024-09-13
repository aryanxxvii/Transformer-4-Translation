import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # pe[words, dimension] -> sin() on even
        pe[:, 0::2] = torch.sin(position * div_term)
        # cos() on odd
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # add dimension at 0th position => (1, seq_len, d_model)
        
        # static parameter (won't change while training)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    


class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean) / (std+self.eps) + self.beta
    


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x



class ScaledMultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        
        assert d_model % heads == 0, "d_model should be divisible by heads"
        
        self.d_k = d_model // heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    
    @staticmethod
    def attention(q, k, v, mask, dropoutLayer: nn.Dropout):
        d_k = q.shape[-1]

        # (batch_size, heads, seq_len, d_k) ->
        # (batch_size, heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_scores = attention_scores / math.sqrt(d_k)

        if mask is not None:
            scaled_attention_scores.masked_fill_(mask == 0, -1e9)
        
        # (batch_size, heads, seq_len, seq_len)
        attention_weights = scaled_attention_scores.softmax(dim = -1)

        if dropoutLayer is not None:
            attention_weights = dropoutLayer(attention_weights)
        
        attention_output = torch.matmul(attention_weights, v)

        return attention_output, attention_scores


    def forward(self, q, k, v, mask=None):
        q_dash = self.w_q(q)
        k_dash = self.w_k(k)
        v_dash = self.w_v(v)

        # (batch_size, seq_len, d_model) -> 
        # (batch_size, seq_len, heads, d_k) ->
        # (batch_size, heads, seq_len, d_k)
        q_dash = q_dash.view(q_dash.shape[0], q_dash.shape[1], self.heads, self.d_k).transpose(1, 2)
        k_dash = k_dash.view(k_dash.shape[0], k_dash.shape[1], self.heads, self.d_k).transpose(1, 2)
        v_dash = v_dash.view(v_dash.shape[0], v_dash.shape[1], self.heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = ScaledMultiHeadAttentionBlock.attention(q_dash, k_dash, v_dash, mask, self.dropout)
        
        # (batch_size, heads, seq_len, d_k) ->
        # (batch_size, seq_len, heads, d_k) ->
        # (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.w_o(x)



class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    # Add skipped output with normalized output of unskipped layer
    # subLayer = layer in-between
    def forward(self, x, subLayer):
        return self.dropout(subLayer(self.norm(x)))
    


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: ScaledMultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout), 
                                                   ResidualConnection(dropout)])
        
    def forward(self, x, source_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x



class EncoderStack(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    


class DecoderBlock(nn.Module):

    def __init__(self,
                 self_attention_block: ScaledMultiHeadAttentionBlock,
                 cross_attention_block: ScaledMultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout),
                                                   ResidualConnection(dropout),
                                                   ResidualConnection(dropout)])
        
    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, source_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    


class DecoderStack(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)



class OutputProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projectionLayer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) ->
        # (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.projectionLayer(x), dim=-1)
    


class Transformer(nn.Module):

    def __init__(self,
                 encoder: EncoderStack,
                 decoder: DecoderStack,
                 source_embedding: InputEmbeddings,
                 target_embedding: InputEmbeddings,
                 source_pos: PositionalEncoding,
                 target_pos: PositionalEncoding,
                 output_projection_layer: OutputProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.output_projection_layer = output_projection_layer

    def encode(self, source, source_mask):
        source = self.source_embedding(source)
        source = self.source_pos(source)
        return self.encoder(source, source_mask)
    
    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)
    
    def project(self, x):
        return self.output_projection_layer(x)
    

        
def build_transformer(source_vocab_size: int,
            target_vocab_size: int,
            source_seq_len: int,
            target_seq_len: int,
            d_model: int = 512,
            N: int = 6,
            heads: int = 8,
            dropout: float = 0.1,
            d_ff = 2048):
    
    # Word embeddings
    source_embedding = InputEmbeddings(d_model, source_vocab_size)
    target_embedding = InputEmbeddings(d_model, target_vocab_size)

    # Positional encoding
    source_pos = PositionalEncoding(d_model, source_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    # Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = ScaledMultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = ScaledMultiHeadAttentionBlock(d_model, heads, dropout)
        decoder_cross_attention_block = ScaledMultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Encoder stack
    encoder = EncoderStack(nn.ModuleList(encoder_blocks))

    # Decoder stack
    decoder = DecoderStack(nn.ModuleList(decoder_blocks))

    # Output projection layer
    output_projection_layer = OutputProjectionLayer(d_model, target_vocab_size)

    # Transformer
    transformer =  Transformer(
                        encoder,
                        decoder,
                        source_embedding,
                        target_embedding,
                        source_pos,
                        target_pos,
                        output_projection_layer)

    # Parameter Initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
