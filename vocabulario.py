import json
import re
from collections import Counter

# Cargar el texto original
with open('el_alquimista.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Procesar el texto
text = text.lower()
text = re.sub(r'[^a-zñáéíóúü\s]', '', text)  # Elimina caracteres no deseados
words = text.split()

# Construir el vocabulario
max_vocab_size = 70000  # Limitar el tamaño del vocabulario
word_counts = Counter(words)
vocab = {word: i + 4 for i, (word, _) in enumerate(word_counts.most_common(max_vocab_size))}

# Añadir tokens especiales
vocab['<pad>'] = 0
vocab['<unk>'] = 1
vocab['<sos>'] = 2
vocab['<eos>'] = 3

# Guardar el vocabulario
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab, f)

print("Vocabulario generado y guardado como vocab.json")
