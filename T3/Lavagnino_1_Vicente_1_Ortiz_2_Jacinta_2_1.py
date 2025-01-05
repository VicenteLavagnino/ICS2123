import numpy as np
from random import choice, seed
from matplotlib import pyplot as plt
#PROBLEMA 1  (Se utilizó como referencia el código de la Ayudantía 8 P3)
#Creamos el tablero
def tablero_ajedrez():
    lista = []
    for _ in range(8):
        lista.append([0] * 8)
    return lista 
# creamos un alista de las posiciones del tablero
def posiciones_validas():
    posiciones = list()
    for fila in range(8):
        for columna in range(8):
            posiciones.append([fila, columna])
    return posiciones
#definimos los movimientos posibles deacuerdo al enuncioado
def combinaciones_posicion(x, y, pos_validas):
    lista_combinaciones = [[x + 2, y ], [x + 1, y ], [x - 2, y ], [x - 1, y],  #movs horizontales
                           [x, y + 2], [x, y + 1], [x, y - 2], [x, y - 1],     #movs verticales
                           [x , y ]] #se mantiene en la posicion
    lista_valida = [x for x in lista_combinaciones if x in pos_validas]
    return lista_valida
#definimos la matriz de probabilidades
def matriz_probabilidades_transicion():
    pos_validas = posiciones_validas()

    P = np.zeros((64, 64))
    # Se recorre cada casilla del tablero
    for fila in range(8):
        for columna in range(8):
            movimientos_validos = combinaciones_posicion(fila, columna, pos_validas)
            # Calcular la probabilidad de transición para cada movimiento válido
            for movimiento in movimientos_validos:
                indice = (movimiento[0] * 8 + movimiento[1]) 
                P[fila * 8 + columna][indice] = 1 / len(movimientos_validos)
    # La matriz P muestra la probabilidad de que el will se mueva desde la casilla indicada
    # por la fila i a indicada por la columna j
    return P
will = matriz_probabilidades_transicion()  # Generar la matriz de probabilidades de transición

# Guardar la matriz en un archivo de texto
np.savetxt("matriz_probabilidades.txt", will, fmt="%.4f") 

# B)
#buscamos la matriz deprobabilidades de transicion de Chris
def combinaciones_posicion_cris(x, y, pos_validas):
    lista_combinaciones = [[x - 2, y - 1],
                           [x - 2, y + 1],
                           [x - 1, y - 2],
                           [x - 1, y + 2],
                           [x + 1, y - 2],
                           [x + 1, y + 2],
                           [x + 2, y - 1],
                           [x + 2, y + 1]] #se mantiene en la posicion
    lista_valida = [x for x in lista_combinaciones if x in pos_validas]
    return lista_valida

def matriz_probabilidades_transicion_cris():
    pos_validas = posiciones_validas()
    C = np.zeros((64, 64))
    # Se recorre cada casilla del tablero
    for fila in range(8):
        for columna in range(8):
            movimientos_validos = combinaciones_posicion_cris(fila, columna, pos_validas)
            # Calcular la probabilidad de transición para cada movimiento válido
            for movimiento in movimientos_validos:
                indice = (movimiento[0] * 8 + movimiento[1]) 
                C[fila * 8 + columna][indice] = 1 / len(movimientos_validos)
    # La matriz P muestra la probabilidad de que el will se mueva desde la casilla indicada
    # por la fila i a indicada por la columna j
    return C
cris=matriz_probabilidades_transicion_cris()
np.savetxt("matriz_probabilidades_cris.txt", cris, fmt="%.4f")  

x_w, y_w = 0, 0  # Posición inicial de Will
x_c, y_c = 7, 7  # Posición inicial de Chris
inicio_will = x_w * 8 + y_w
inicio_chris = x_c * 8 + y_c

probabilidad_total = 0.0
# Iterar sobre el número de movimientos de 1 a 15
for k in range(1, 16):
    # calculamos las matrices elevadas
    P_will_k = np.linalg.matrix_power(will, k)
    P_chris_k = np.linalg.matrix_power(cris, k)
    # Iterar sobre todas las posiciones de Chris
    for x in range(8):
        for y in range(8):
            pos_chris = x * 8 + y  # Índice de la posición de Chris 
            
            # Calcular la posicion objetivo: frente a chris
            pos_will = (x - 1) * 8 + y if x > 0 else None  

            if pos_will is not None:
                # Multiplicar las probabilidades 
                probabilidad = P_will_k[inicio_will, pos_will] * P_chris_k[inicio_chris, pos_chris]
                probabilidad_total += probabilidad
print(probabilidad_total)

## C

P = matriz_probabilidades_transicion_cris()
A = np.transpose (np.identity(64) - np.matrix(P))
A = np.vstack([A,[1 for _ in range(64)]])
b = np.transpose(np.array([0 for i in range(64)] + [1]) )
pi = np.linalg.lstsq(A, b, rcond=None)[0]
pi_reshape = np.reshape(pi, (8,8))


##D

W = matriz_probabilidades_transicion()
A_W = np.transpose (np.identity(64) - np.matrix(W))
A_W = np.vstack([A_W,[1 for _ in range(64)]])
b_W = np.transpose(np.array([0 for i in range(64)] + [1]) )
pi_W = np.linalg.lstsq(A_W, b_W, rcond=None)[0]
pi_reshape_W = np.reshape(pi_W, (8,8))

probabilidad_oscar_will = pi_reshape_W[3][3]
print("WILL",probabilidad_oscar_will) ##prob will
"""
probabilidad_oscar = pi_reshape[6][1]
print("CHRIS",probabilidad_oscar) ##prob CHRIS

plt.imshow(pi_reshape_W , interpolation = 'nearest', origin = 'lower')
plt.colorbar() 
plt.title('Probabilidades a Largo Plazo Will')
plt.show()

plt.imshow(pi_reshape, interpolation = 'nearest', origin = 'lower')
plt.colorbar() 
plt.title('Probabilidades a Largo Plazo Chris')
plt.show()

"""




#pregunta 2

# PARTE A
import numpy as np
np.random.seed(2123)

conteo_exitos = 0

for _ in range(10000):
    tiempo_1 = 0
    tiempo_2 = 0
    buses_1 = 0
    buses_2 = 0
    
    while buses_1 < 2 or buses_2 < 3:
        if buses_1 < 2:
            tiempo_1 += np.random.exponential(1 / 10)
            buses_1 += 1
        if buses_2 < 3:
            tiempo_2 += np.random.exponential(1 / 20)
            buses_2 += 1
    
    if tiempo_1 < tiempo_2:
        conteo_exitos += 1

probabilidad_empirica = conteo_exitos / 10000
# print(f"Probabilidad empírica: {probabilidad_empirica}")

# PARTE B
from numpy.random import poisson

conteo_exitos = 0
n_simulaciones = 10000000
'''
for _ in range(n_simulaciones):
    buses_intervalo1 = np.random.poisson(10 * 3)
    buses_intervalo2 = np.random.poisson(10 * 3)
    
    if buses_intervalo1 == 10 and buses_intervalo2 == 40:
        conteo_exitos += 1
         

probabilidad_empirica = conteo_exitos / n_simulaciones
print(f"Probabilidad empírica: {probabilidad_empirica}")
'''
# PARTE C
n_simulaciones = 1000
esperas = []
'''
for i in range(n_simulaciones):
    tiempo_llegada = np.random.uniform(0, 30)
    tiempo_espera = max(0, 30 - tiempo_llegada)
    
    esperas.append(tiempo_espera)

# Tiempo promedio de espera empírico
espera_promedio = np.mean(esperas)

# Resultados
print(f"Tiempo promedio de espera en minutos: {espera_promedio}")
'''

# PARTE D
n_simulaciones = 1000
conteo_exitos = 0
'''
for _ in range(n_simulaciones):
    tiempo_espera_bus1 = np.random.uniform(15, 30)
    tiempo_espera_bus2 = np.random.exponential(5)
    
    pasajeros_bus1 = np.random.poisson(0.3 * 1 * tiempo_espera_bus1)
    pasajeros_bus2 = np.random.poisson(0.7 * 1 * tiempo_espera_bus2)
    
    if pasajeros_bus1 > 3 * pasajeros_bus2:
        conteo_exitos += 1

probabilidad_empirica = conteo_exitos / n_simulaciones
print(f"Probabilidad empírica: {probabilidad_empirica}")
'''

# PARTE E

# Parámetros del problema
a, b = 20, 30  
c, d = 5, 15   
n, k = 30, 20  
gamma = 80 


tiempo_espera_bus1_promedio = []
tiempo_espera_bus2_promedio = []
ventas_bus1_promedio = []
ventas_bus2_promedio = []

n_simulaciones = 100000

conteo_exitos = 0

for i in range(n_simulaciones):

    tiempo_espera_bus1 = int(np.random.uniform(a, b) // 1)
    tiempo_espera_bus2 = int(np.random.uniform(c, d) // 1)
    tiempo_espera_bus1_promedio.append(tiempo_espera_bus1)
    tiempo_espera_bus2_promedio.append(tiempo_espera_bus2)
    

    vendedores_bus1 = np.random.poisson(gamma / tiempo_espera_bus1)
    vendedores_bus2 = np.random.poisson(gamma / tiempo_espera_bus2)
    

    ventas_bus1_promedio.append(vendedores_bus1)
    ventas_bus2_promedio.append(vendedores_bus2)


    if vendedores_bus1 == n and vendedores_bus2 <= k:
        conteo_exitos += 1


probabilidad_empirica = conteo_exitos / n_simulaciones
print(f"Probabilidad empírica: {probabilidad_empirica}")
print(f"Tiempo promedio de espera bus 1: {np.mean(tiempo_espera_bus1_promedio)}")
print(f"Tiempo promedio de espera bus 2: {np.mean(tiempo_espera_bus2_promedio)}")
print(f"Ventas promedio bus 1: {np.mean(ventas_bus1_promedio)}")
print(f"Ventas promedio bus 2: {np.mean(ventas_bus2_promedio)}")
