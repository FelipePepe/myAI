# Entrenamiento de un Modelo Transformer con El Quijote

Este repositorio contiene el código para entrenar un modelo basado en la arquitectura Transformer utilizando el texto de *Don Quijote de la Mancha* como corpus de entrenamiento. El objetivo del modelo es predecir la siguiente palabra en una secuencia dada de texto, similar a los modelos de predicción de texto utilizados en procesamiento de lenguaje natural (NLP).

## Requisitos

Este código fue probado en un entorno con las siguientes dependencias:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib (opcional, para visualización)
  
Puedes instalar las dependencias necesarias ejecutando:

```bash
pip install torch numpy matplotlib
```

## Descripción del Código
El código sigue varios pasos clave para preparar los datos y entrenar un modelo Transformer:

### Paso 1: Lectura y Preprocesamiento del Texto
El archivo quijote.txt es leído y el texto es procesado de la siguiente manera:

* Se convierte todo el texto a minúsculas.
* Se eliminan los caracteres especiales utilizando expresiones regulares.
* Se tokeniza el texto en palabras.

```python
with open('quijote.txt', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.lower()
text = re.sub(r'[^a-zñáéíóúü\s]', '', text)
words = text.split()
```
### Paso 2: Creación del Vocabulario
A continuación, se crea un vocabulario único de las palabras más frecuentes en el texto. Cada palabra se asigna a un índice en un diccionario, y se añaden tokens especiales como **\<pad>**, **\<sos>**, **\<eos>**, y **\<unk>** para el manejo de secuencias.
```python
word_counts = Counter(words)
vocab = {word: i + 4 for i, (word, _) in enumerate(word_counts.most_common())}
vocab['<pad>'] = 0  # Padding
vocab['<sos>'] = 1  # Start of sequence
vocab['<eos>'] = 2  # End of sequence
vocab['<unk>'] = 3  # Unknown words
```
### Paso 3: Creación de Secuencias de Entrenamiento
El texto se divide en secuencias de longitud fija (50 palabras en este caso) para que el modelo pueda predecir la siguiente palabra de cada secuencia.
```python
sequence_length = 50  # Longitud de las secuencias
sequences = []
for i in range(len(words) - sequence_length):
    sequence = words[i:i + sequence_length + 1]
    sequences.append([vocab.get(word, vocab['<unk>']) for word in sequence])
```
### Paso 4: Dataset Personalizado
El dataset se personaliza para que contenga secuencias de entrada y salida. Las secuencias de entrada son las palabras de una secuencia y las secuencias de salida son las mismas secuencias desplazadas una posición.
``` python
class QuijoteDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence[:-1]), torch.tensor(sequence[1:])
```
### Paso 5: Definición del Modelo Transformer
Se define un modelo Transformer con tres componentes principales:

1. Embeddings: para convertir las palabras en vectores.
2. Transformer: que es el núcleo de la arquitectura.
3. Capa de salida: que genera la predicción de la siguiente palabra.

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src).permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        tgt = self.embedding(tgt).permute(1, 0, 2)
        output = self.transformer(src, tgt)
        output = self.fc_out(output.permute(1, 0, 2))  # (batch_size, seq_len, vocab_size)
        return output
```
### Paso 6: Configuración del Optimizador y Función de Pérdida
El optimizador utilizado es **Adam**, y la función de pérdida es **CrossEntropyLoss**, que es adecuada para tareas de clasificación de texto.
```python
optimizer = optim.Adam(model.parameters(), lr=0.00005)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
```
### Paso 7: Entrenamiento del Modelo
El modelo se entrena durante varias épocas. En cada época, el modelo procesa los lotes de datos, calcula la pérdida y actualiza los pesos a través de retropropagación.

```python
def train_model(model, dataloader, num_epochs, vocab_size):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for src_batch, tgt_batch in dataloader:
            src = torch.nn.utils.rnn.pad_sequence([s.clone().detach() for s in src_batch], batch_first=True, padding_value=vocab['<pad>'])
            tgt = torch.nn.utils.rnn.pad_sequence([t.clone().detach() for t in tgt_batch], batch_first=True, padding_value=vocab['<pad>'])

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)
            output = output.reshape(-1, vocab_size)
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Época {epoch + 1}/{num_epochs}, Pérdida: {total_loss / len(dataloader)}")

```
### Paso 8: Guardar el Modelo
Después del entrenamiento, el modelo entrenado se guarda en un archivo para futuras inferencias.
```python
torch.save(model.state_dict(), 'transformer_quijote.pth')
```
### Resultados
Durante el entrenamiento, la pérdida (loss) debe disminuir a medida que el modelo aprende a predecir la siguiente palabra en la secuencia. Puedes observar el progreso del modelo a través de las impresiones de la pérdida en cada batch y época.

### Conclusión
Este proyecto utiliza la arquitectura Transformer para modelar el lenguaje a partir de un texto literario. A pesar de que este es solo un punto de partida, se pueden hacer mejoras usando técnicas avanzadas como el ajuste de hiperparámetros, el uso de más datos o el cambio de la arquitectura del modelo.

### License
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.