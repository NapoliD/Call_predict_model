from flask import Flask, request
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos de entrada del POST request
    input_data = request.get_json()
    
    # Cargar el modelo predictivo
    model = joblib.load('modelo.pkl')
    
    # Realizar la predicción con los datos de entrada
    prediction = model.predict(input_data)
    
    # Devolver el resultado de la predicción
    return {'prediction': prediction.tolist()}

if __name__ == '__main__':
    app.run(debug=True)
