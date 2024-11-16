import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# Definición del modelo (igual al utilizado durante el entrenamiento)
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

# Cargar el vocabulario
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        return json.load(f)

vocab_path = 'vocab.json'
vocab = load_vocab(vocab_path)
reverse_vocab = {idx: word for word, idx in vocab.items()}

# Parámetros del modelo
vocab_size = len(vocab)
d_model = 512
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 512

# Cargar el modelo entrenado
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
state_dict = torch.load('transformer_el_alquimista.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Función para generar texto
def generate_text_with_sampling(model, vocab, prompt, max_length=50, temperature=1.0, top_k=10):
    tokens = [vocab.get(word, vocab['<unk>']) for word in prompt.lower().split()]
    if not tokens:
        return "Por favor, escribe algo válido."
    
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    output = input_tensor.clone()

    for _ in range(max_length):
        with torch.no_grad():
            tgt_input = output[:, -1:]
            prediction = model(input_tensor, tgt_input)[:, -1, :] / temperature
            
            current_top_k = min(top_k, prediction.size(-1))
            top_k_values, top_k_indices = torch.topk(prediction, current_top_k)
            probs = F.softmax(top_k_values, dim=-1)
            
            if probs.numel() == 0:
                break
            
            sampled_index = torch.multinomial(probs, 1).item()
            next_token = top_k_indices[0, sampled_index].item()
            if next_token == vocab['<eos>']:
                break

            output = torch.cat([output, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
    
    generated_text = ' '.join([reverse_vocab.get(idx, '<unk>') for idx in output.squeeze().tolist()])
    return generated_text

# Función para interactuar en forma de chat
def chat_with_model(model, vocab):
    print("¡Hola! Soy un chatbot basado en 'El Alquimista'. Escribe 'salir' para terminar.")
    while True:
        prompt = input("Tú: ")
        if prompt.lower() == 'salir':
            print("¡Adiós!")
            break
        response = generate_text_with_sampling(model, vocab, prompt, temperature=1.2, top_k=10)
        print(f"Bot: {response}")

# Ejecutar el chatbot
chat_with_model(model, vocab)
