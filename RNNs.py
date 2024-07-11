import streamlit as st
import numpy as np

# Definir el modelo RNN usando NumPy
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wx = np.random.randn(input_size, hidden_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
    
    def forward(self, inputs):
        h = np.zeros((inputs.shape[0], self.hidden_size))
        for t in range(inputs.shape[1]):
            h = np.tanh(np.dot(inputs[:, t, :], self.Wx) + np.dot(h, self.Wh) + self.bh)
        y = np.dot(h, self.Wy) + self.by
        return y, h

    def loss(self, outputs, targets):
        return np.mean((outputs - targets) ** 2)

    def train(self, X, y, num_epochs=10, learning_rate=0.01):
        loss_history = []
        for epoch in range(num_epochs):
            outputs, _ = self.forward(X)
            loss = self.loss(outputs, y)
            loss_history.append(loss)
            
            dL_dy = 2 * (outputs - y) / y.size
            dL_dWy = np.dot(_, dL_dy)
            dL_dby = dL_dy.sum(axis=0, keepdims=True)
            
            dL_dh = np.dot(dL_dy, self.Wy.T)
            dL_dWx = np.zeros_like(self.Wx)
            dL_dWh = np.zeros_like(self.Wh)
            dL_dbh = np.zeros_like(self.bh)
            
            for t in reversed(range(X.shape[1])):
                dL_dh_raw = dL_dh * (1 - _[:, t, :] ** 2)
                dL_dWx += np.dot(X[:, t, :].T, dL_dh_raw)
                if t > 0:
                    dL_dWh += np.dot(_[:, t-1, :].T, dL_dh_raw)
                dL_dbh += dL_dh_raw.sum(axis=0, keepdims=True)
                dL_dh = np.dot(dL_dh_raw, self.Wh.T)
            
            self.Wy -= learning_rate * dL_dWy
            self.by -= learning_rate * dL_dby
            self.Wx -= learning_rate * dL_dWx
            self.Wh -= learning_rate * dL_dWh
            self.bh -= learning_rate * dL_dbh
        
        return loss_history

# Función para crear y entrenar una RNN simple
def create_and_train_rnn(sequence, num_epochs=10, hidden_size=50, learning_rate=0.01):
    # Preparar los datos
    X = np.array(sequence[:-1]).reshape((1, len(sequence)-1, 1))
    y = np.array(sequence[1:]).reshape((1, len(sequence)-1, 1))

    input_size = 1
    output_size = 1

    model = SimpleRNN(input_size, hidden_size, output_size)
    loss_history = model.train(X, y, num_epochs, learning_rate)
    
    return model, loss_history

# Función para validar la secuencia de entrada
def validate_sequence(seq):
    try:
        sequence = [int(i) for i in seq.split(',')]
        return sequence, None
    except ValueError:
        return None, "Por favor, introduce una secuencia válida de números separados por comas."

# Interfaz de usuario con Streamlit
st.title('Redes Neuronales Recurrentes con Streamlit (sin PyTorch)')

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
        next_value, _ = model.forward(np.array(sequence[-len(sequence)+1:]).reshape(1, len(sequence)-1, 1))
        
        st.write(f'Secuencia: {sequence}')
        st.write(f'Predicción del próximo valor en la secuencia: {next_value.item()}')

        # Gráfico de la secuencia original y la predicción
        st.line_chart(sequence + [next_value.item()])

        # Gráfico de la historia de la pérdida
        st.line_chart(loss_history)

# Ejecutar la aplicación de Streamlit
