import numpy as np
from torch import nn
import math
import torch


#Realize Multihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):  
        super(MultiHeadAttention,self).__init__()
        assert d_model % num_heads ==0,"d_model must be divisible by num_head"

        self.d_model = d_model           # Dimension of the model
        self.num_heads = num_heads       # Number of heads
        self.d_k = d_model // num_heads  # Dimension of each head

        # Define the Liner transfomation layers(no bias requiered)
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)



    def ScaledDotProductAttention(self,Q,K,V,mask = None):
        """
        Compute scaled dot product attention.
        The three input matrix are Q(Query),K(Key),V(Value).
        The shape of the input matrix above is (batch_size,num_heads,seq_length,d_k).
        The shape of the output matrix is same.
        Mask operation is optional.
        """
        # Compute attention scores(Q·K^T/sqrt(d_k))
        attn_scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)  #the shape of it is (batch_size,num_heads,seq_length,seq_length) 

        # Mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0 , -1e9)

        # Compute attention weights
        attn_weights = torch.softmax(attn_scores,dim = -1)

        # Compute the weighted sum of value vector
        output = torch.matmul(attn_weights,V)

        return output
    
    def split_heads(self,x):
        """
        Split the input tensor into many heads
        The shape of the input is (batch_size,seq_length,d_model)
        The shape of the output is (batch_size,num_heads,seq_length,d_k)
        """
        bacth_size,seq_length,d_model = x.size()
        return x.view(bacth_size,seq_length,self.num_heads,self.d_k).transpose(1,2)
    
    def concat_heads(self,x):
        """
        Merge the output of multiple heads back into the original shape
        The shape of the input is (batch_size,num_heads,seq_length,d_k)
        The shape of the output is (batch_size,seq_length,d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model)
    
    def forward(self,Q,K,V,mask = None):
        """
        Forward propagation
        The shape of the input is (batch_size,seq_length,d_model)
        The shape of the output is (batch_size,seq_length,d_model)
        """

        # Linear trasformation and split the multiheads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Compute attention
        attn_output = self.ScaledDotProductAttention(Q,K,V,mask)

        # merge the multiheads and do linear transformation
        output = self.W_o(self.concat_heads(attn_output))

        return output
    
# Realize the Position-wise Feed-Forward Network
class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))
    
# Build the Positioinal ENcoding
class PositonalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_length):
        super(PositonalEncoding,self).__init__()
        pe = torch.zeros(max_seq_length,d_model)
        position = torch.arange(0,max_seq_length,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*-(math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.register_buffer('pe',pe.unsqueeze(0))
    
    def forward(self,x):
        # add the positinal encoding into the input 
        return x + self.pe[:,:x.size(1)]
    
# Bulid the Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(EncoderBlock,self).__init__()
        self.self_attn = MultiHeadAttention(d_model,num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        # Multihead attention
        attn_output = self.self_attn(x,x,x,mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x+self.dropout(ff_output))

        return x

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(DecoderBlock,self).__init__()
        self.self_attn = MultiHeadAttention(d_model,num_heads)
        self.cross_atnn = MultiHeadAttention(d_model,num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,enc_output,src_mask,tgt_mask):
        attn_output = self.self_attn(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_atnn(x,enc_output,enc_output,src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
    
# Build the Transformer
class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model,num_heads,num_layers,d_ff,max_seq_length,dropout):
        super(Transformer,self).__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size,d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size,d_model)
        self.positional_encoding = PositonalEncoding(d_model,max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model,tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):



        # 源掩码：屏蔽填充符（假设填充符索引为0）
        # 形状：(batch_size, 1, 1, seq_length)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
   
        # 目标掩码：屏蔽填充符和未来信息
        # 形状：(batch_size, 1, seq_length, 1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        # 生成上三角矩阵掩码，防止解码时看到未来信息
        device = tgt_mask.device
        nopeak_mask = (1 - nn.functional.pad(torch.ones(1, seq_length, seq_length - 1), (0, 1))).to(device).bool()

        

        tgt_mask = tgt_mask & nopeak_mask  # 合并填充掩码和未来信息掩码
        return src_mask, tgt_mask
    
    def forward(self,src,tgt):
        # Generate mask
        src_mask,tgt_mask = self.generate_mask(src,tgt)

        # Encoder part
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output,src_mask)

        # Decoder part
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output,enc_output,src_mask,tgt_mask)

        # Final output 
        output = self.fc(dec_output)
        return output