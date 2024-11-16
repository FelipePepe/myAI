import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter

# Paso 1: Leer y Preprocesar el Texto
print("Leer y preprocesar el texto...")
with open('el_alquimista.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Convertir a minúsculas y eliminar caracteres especiales
print("Convertir a minúsculas y eliminar caracteres especiales...")
text = text.lower()
text = re.sub(r'[^a-zñáéíóúü\s]', '', text)

# Dividir el texto en palabras
print("Dividir el texto en palabras...")
words = text.split()
max_words = 100000
words = words[:max_words]

# Paso 2: Crear el Vocabulario
print("Crear el vocabulario...")
max_vocab_size = 70000  # Limitar el tamaño del vocabulario
word_counts = Counter(words)
vocab = {word: i + 4 for i, (word, _) in enumerate(word_counts.most_common(max_vocab_size))}
vocab['<pad>'] = 0  # Padding
vocab['<sos>'] = 1  # Start of sequence
vocab['<eos>'] = 2  # End of sequence
vocab['<unk>'] = 3  # Unknown words

# Paso 3: Crear Secuencias de Entrenamiento
print("Crear secuencias de entrenamiento...")
sequence_length = 50  # Longitud de las secuencias
sequences = []
for i in range(len(words) - sequence_length):
    sequence = words[i:i + sequence_length + 1]
    sequences.append([vocab.get(word, vocab['<unk>']) for word in sequence])

# Paso 4: Dataset Personalizado
class QuijoteDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence[:-1]), torch.tensor(sequence[1:])

# Crear Dataset y DataLoader
print("Crear Dataset y DataLoader...")
train_size = int(0.8 * len(sequences))
train_sequences = sequences[:train_size]
val_sequences = sequences[train_size:]

train_dataset = QuijoteDataset(train_sequences)
val_dataset = QuijoteDataset(val_sequences)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Paso 5: Definir el Modelo Transformer
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

# Parámetros del Modelo
print("Definir el modelo Transformer...")
vocab_size = len(vocab)
d_model = 512
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 512
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)

# Paso 6: Configurar el Optimizador y la Función de Pérdida
print("Configurar el optimizador y la función de pérdida...")
optimizer = optim.Adam(model.parameters(), lr=0.00005)  # Reducido de 0.0001
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # Reducir lr cada época

# Paso 7: Entrenar el Modelo
def train_model(model, train_loader, val_loader, num_epochs, vocab_size):
    print("Entrenando el modelo...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (src_batch, tgt_batch) in enumerate(train_loader):
            # Preparar batch con padding
            src = torch.nn.utils.rnn.pad_sequence([s.clone().detach() for s in src_batch], batch_first=True, padding_value=vocab['<pad>'])
            tgt = torch.nn.utils.rnn.pad_sequence([t.clone().detach() for t in tgt_batch], batch_first=True, padding_value=vocab['<pad>'])

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Forward pass
            optimizer.zero_grad()
            output = model(src, tgt_input)

            # Calcular la pérdida
            output = output.reshape(-1, vocab_size)
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)

            # Backward pass y optimización
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 100 == 0:  # Imprimir cada 100 batchs
                print(f"Época {epoch + 1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Pérdida Promedio Acumulada: {total_loss / (i + 1)}")

        # Ajuste del learning rate
        scheduler.step()

        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                src = torch.nn.utils.rnn.pad_sequence([s.clone().detach() for s in src_batch], batch_first=True, padding_value=vocab['<pad>'])
                tgt = torch.nn.utils.rnn.pad_sequence([t.clone().detach() for t in tgt_batch], batch_first=True, padding_value=vocab['<pad>'])
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = model(src, tgt_input)
                output = output.reshape(-1, vocab_size)
                tgt_output = tgt_output.reshape(-1)
                
                val_loss += criterion(output, tgt_output).item()

        print(f"Época {epoch + 1}/{num_epochs}, Pérdida de Validación: {val_loss / len(val_loader)}")

# Entrenar el Modelo
print("Entrenar el modelo...")
num_epochs = 6
train_model(model, train_loader, val_loader, num_epochs, vocab_size)

# Paso 8: Guardar el Modelo Entrenado
print("Guardar el modelo entrenado...")
torch.save(model.state_dict(), 'transformer_el_alquimista.pth')

print("Entrenamiento completado y modelo guardado.")
