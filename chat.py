import torch
import torch.nn as nn
import torch.nn.functional as F

# Definir el modelo Transformer (debe ser igual al utilizado durante el entrenamiento)
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)  # Regularización con Dropout

    def forward(self, src, tgt):
        src = self.dropout(self.embedding(src)).permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        tgt = self.dropout(self.embedding(tgt)).permute(1, 0, 2)
        output = self.transformer(src, tgt)
        output = self.fc_out(output.permute(1, 0, 2))  # (batch_size, seq_len, vocab_size)
        return output

# Cargar el vocabulario desde un archivo JSON
import json

def load_vocab(vocab_path):
    print("Cargando el vocabulario...")
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    print("Vocabulario cargado correctamente.")
    return vocab

# Cargar el vocabulario
vocab_path = 'vocab.json'  # Cambia esta ruta si el archivo está en otro lugar
vocab = load_vocab(vocab_path)

# Cargar el modelo
vocab_size = len(vocab)
d_model = 512
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 4
dim_feedforward = 512

# Instanciar el modelo con los mismos parámetros que usaste durante el entrenamiento
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)

# Cargar el estado guardado del modelo
print("Cargando el modelo entrenado...")
state_dict = torch.load('transformer_el_alquimista.pth', map_location='cpu')  # Cambia la ruta si es necesario
model.load_state_dict(state_dict)
print("Modelo cargado exitosamente.")

# Función de generación de texto con muestreo aleatorio o top-k sampling
def generate_text_with_sampling(model, vocab, prompt, max_length=50, temperature=1.5, top_k=10):
    model.eval()

    # Tokenizar el prompt
    tokens = [vocab.get(word, vocab['<unk>']) for word in prompt.lower().split()]
    print(f"Tokens del prompt: {tokens}")

    # Verificar si el prompt está vacío
    if not tokens:
        return "Por favor, ingresa una entrada válida."

    # Crear tensor de entrada asegurando el tipo correcto
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    output = input_tensor

    for _ in range(max_length):
        with torch.no_grad():
            tgt_input = output[:, -1:]  # Último token
            prediction = model(input_tensor, tgt_input)
            prediction = prediction[:, -1, :] / temperature

            # Ajustar dinámicamente top_k al tamaño de las predicciones
            current_top_k = min(top_k, prediction.size(-1))
            top_k_values, top_k_indices = torch.topk(prediction, current_top_k)

            # Usar softmax para calcular probabilidades
            probs = F.softmax(top_k_values, dim=-1)

            # Verificar que `probs` tiene elementos para muestrear
            if probs.numel() == 0:
                print("Advertencia: No hay tokens disponibles para muestrear.")
                break

            # Realizar muestreo
            sampled_index = torch.multinomial(probs, 1).item()
            next_token = top_k_indices[sampled_index].item()

            # Añadir el token generado
            output = torch.cat((output, torch.tensor([[next_token]], dtype=torch.long)), dim=1)

    # Convertir tokens generados a texto
    reverse_vocab = {v: k for k, v in vocab.items()}
    generated_text = ' '.join([reverse_vocab.get(token, '<unk>') for token in output.squeeze().tolist()])
    return generated_text

# Función para interactuar con el modelo en forma de chat
def chat_with_model(model, vocab):
    print("¡Hola! Soy un chatbot basado en 'El Alquimista'. Escribe 'salir' para terminar.")
    
    while True:
        # Obtener la entrada del usuario (pregunta)
        prompt = input("Tú: ")
        if not prompt.strip():
            print("Por favor, escribe algo.")
            continue
        if prompt.lower() == 'salir':
            print("¡Adiós!")
            break
        
        # Generar respuesta usando el modelo
        response = generate_text_with_sampling(model, vocab, prompt, temperature=1.5, top_k=10)
        
        # Imprimir la respuesta generada
        print("Bot: " + response)

# Llamar a la función para chatear
chat_with_model(model, vocab)
