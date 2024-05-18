import numpy as np #Llamar a la libreria de manejo de datos 

#Definir la formula de la funcion sigmoide
def sigmoide(x):
    return 1/(1+np.exp(-x))

#Datos de entrada y configuracion de salida
x = np.array([[0], [1]]) #Entrada
y = np.array([[1], [0]]) #Salida

#Definir los pesos, bias (sesgo) y tasa de aprendizaje
w = 0.6 #Peso
b = 0.2 #Bias
alpha = 0.1 #Tasa de aprendizaje

#Comienza el proceso de aprendizaje
for i in range(100):
    z = np.dot(x, w) + b #Sumatoria de los pesos y el bias
    sigmoide_z = sigmoide(z) #Aplicar la funcion de activacion

    #Calculo del error
    error = y - sigmoide_z

    #Propagacion hacia atras
    w = w + alpha * np.sum(x * error)
    b = b + alpha * np.sum(error)

#Comprobacion de la salida
z = np.dot(x, w) + b
sigmoide_z = sigmoide(z)
print("Salida de la compuerta NOT después de 100 iteraciones: ", sigmoide_z)
