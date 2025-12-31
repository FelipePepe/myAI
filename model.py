import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, tgt):
        src = self.dropout(self.embedding(src)).permute(1, 0, 2)
        tgt = self.dropout(self.embedding(tgt)).permute(1, 0, 2)
        output = self.transformer(src, tgt)
        output = self.fc_out(output.permute(1, 0, 2))
        return output
