import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Definir el modelo RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Función para crear y entrenar una RNN simple
def create_and_train_rnn(sequence, num_epochs=10, hidden_size=50, learning_rate=0.01):
    # Preparar los datos
    X = np.array(sequence[:-1]).reshape((1, len(sequence)-1, 1))
    y = np.array(sequence[1:]).reshape((1, len(sequence)-1, 1))

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    input_size = 1
    output_size = 1

    model = SimpleRNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    
    return model, loss_history

# Función para validar la secuencia de entrada
def validate_sequence(seq):
    try:
        sequence = [int(i) for i in seq.split(',')]
        return sequence, None
    except ValueError:
        return None, "Por favor, introduce una secuencia válida de números separados por comas."

# Interfaz de usuario con Streamlit
st.title('Redes Neuronales Recurrentes con Streamlit')

# Entrada de usuario para la secuencia
sequence_input = st.text_input('Introduce una secuencia de números separada por comas', '1,2,3,4,5,6,7,8,9')
sequence, error = validate_sequence(sequence_input)

if error:
    st.error(error)
else:
    # Entrada de usuario para el número de épocas
    num_epochs = st.number_input('Número de épocas para entrenar la RNN', min_value=1, max_value=100, value=10, step=1)

    # Entrada de usuario para el tamaño de la capa oculta
    hidden_size = st.number_input('Tamaño de la capa oculta', min_value=1, max_value=100, value=50, step=1)

    # Entrada de usuario para la tasa de aprendizaje
    learning_rate = st.number_input('Tasa de aprendizaje', min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f")

    # Botón para entrenar la RNN
    if st.button('Entrenar RNN'):
        model, loss_history = create_and_train_rnn(sequence, num_epochs, hidden_size, learning_rate)
        
        # Hacer una predicción
        model.eval()
        with torch.no_grad():
            next_value = model(torch.tensor(np.array(sequence[-len(sequence)+1:]).reshape(1, len(sequence)-1, 1), dtype=torch.float32))
        
        st.write(f'Secuencia: {sequence}')
        st.write(f'Predicción del próximo valor en la secuencia: {next_value.item()}')

        # Gráfico de la secuencia original y la predicción
        fig, ax = plt.subplots()
        ax.plot(sequence, label='Secuencia Original')
        ax.plot(len(sequence), next_value.item(), 'ro', label='Predicción')
        ax.legend()
        st.pyplot(fig)

        # Gráfico de la historia de la pérdida
        fig, ax = plt.subplots()
        ax.plot(loss_history, label='Pérdida durante el entrenamiento')
        ax.set_xlabel('Época')
        ax.set_ylabel('Pérdida')
        ax.legend()
        st.pyplot(fig)

# Ejecutar la aplicación de Streamlit
