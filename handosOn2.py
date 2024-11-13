import numpy as np
import sys

class DataSet:
    """Clase que representa el conjunto de datos de entrada."""
    def __init__(self):
        # Datos de entrada hardcoded
        self.x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])
        

class LinearRegression:
    """Clase que representa el modelo de regresión lineal."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.slope = None
        self.intercept = None

    def fit(self):
        """Calcular la pendiente y la intersección de la recta de regresión."""
        x_mean = np.mean(self.dataset.x)
        y_mean = np.mean(self.dataset.y)

        num = np.sum((self.dataset.x - x_mean) * (self.dataset.y - y_mean))
        den = np.sum((self.dataset.x - x_mean) ** 2)

        self.slope = num / den
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, x):
        """Predecir valores de y para los valores de x proporcionados."""
        return self.slope * x + self.intercept

    def get_equation(self):
        """Devuelve la ecuación de la regresión en formato string."""
        return f"y = {self.slope:.2f} * x + {self.intercept:.2f}"

if __name__ == "__main__":
    # Verificar que el usuario haya pasado un argumento para predicción
    if len(sys.argv) < 2:
        print("Por favor, proporciona un valor de publicidad para predecir las ventas.")
        sys.exit(1)

    # Convertir el argumento pasado en un valor de tipo float
    try:
        advertising_value = float(sys.argv[1])
    except ValueError:
        print("El valor proporcionado no es un número válido.")
        sys.exit(1)

    # Crear el conjunto de datos y el modelo de regresión
    dataset = DataSet()
    model = LinearRegression(dataset)

    # Ajustar el modelo
    model.fit()

    # Imprimir la ecuación de regresión
    print("Ecuación de regresión:", model.get_equation())

    # Realizar la predicción para el valor de advertising pasado como argumento
    predicted_sales = model.predict(advertising_value)
    print(f"Predicción de ventas para advertising = {advertising_value}: {predicted_sales:.2f}")
