# importamos las librearias necesarias
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import scipy.stats as stats
import optuna
import matplotlib.animation as animation
import seaborn as sns

# Definimos generadores de realizaciones de variables aleatorias exponencial, uniforme y binomial 
def exponential_instance(lmbda):
    u = np.random.uniform(0, 1)
    return -np.log(1 - u) / lmbda

def uniform_instance(a, b):
    u = np.random.uniform(0, 1)
    return (b - a) * u + a

def binomial_instance(n, p):
    return max(1,np.random.binomial(n, p))

def rayleigh_instance(sigma):
    u = np.random.uniform(0, 1)
    return sigma * np.sqrt(-2 * np.log(1 - u))


# Parte A: Modelo de simulación y primeros resultados
def simulacion_puesto_completos(s, S, tau, K, seed, Tfin, n, p, TiempoProxLlegada):
    # Si verbose es True, se imprime el resultado de la simulación
    # verbose = False
    verbose = True

    # Setear la semilla
    random.seed(seed)
    np.random.seed(seed)

    # Parámetros
    a = 2 # a = 4
    b = 4  # b = 8                                
    # n = 3
    # p = 0.5
    lambda_atencion = 1/7

    # Inicialización                                                          
    T = 0                                       
    # Tfin = 60*(24-8)                            
    NClientesAtendidos = 0                      
    NClientesPerdidos = 0                       
    LargoCola = 0         
    Capacidad = K                      
    inventario = S                               
    EstadoCompletero = 0    
    param1, param2 = a,b                      
    # TiempoProxLlegada = uniform_instance(param1, param2)   
    TiempoProxSalida = float('inf')             
    TiempoTotalPermanencia = 0                  
    TiemposLlegada = np.zeros(Capacidad)        
    TiempoPermanenciaMaximo = 0                 
    TiempoTotalSistemaVacio = 0                 
    CompletosVendidos = 0 

    # Costos y Precios   
    CostoInventario = 0
    CostoReposicion = 0
    CostoFaltante = 0
    Costo_Unidad_Inventario = math.ceil(tau/12) + 10  # beta
    Costo_Unidad_Orden = 50
    Costo_Orden = math.ceil(600/tau) + 35       # alpha
    PrecioCompleto = 300
    CostoUnidadFaltante = 50
    CostoLocalLleno = 0
    Costo_Local_Lleno_Unitario = 300
    CostoArriendo = 700*K + 200*S
    CostoTiempoExtra = 0
    CostoTiempoExtraUnitario = 500

    # Listas para graficar
    lista_cantidad_personas =[]
    lista_tiempos = []
    lista_inventarios = []

    # Lista de tiempos de inventario (cada tau minutos se revisa el inventario app)
    lista_tiempos_inventario = [(i-tau, i) for i in range(tau, Tfin*tau, tau)]


    # Simulación
    while T < Tfin or LargoCola > 0 or EstadoCompletero == 1:

        # Revisamos inventario
        tiempo_inventario = lista_tiempos_inventario[0]
        if T >= tiempo_inventario[1]:
            CostoInventario += Costo_Unidad_Inventario * inventario
            lista_tiempos_inventario.pop(0)
            if inventario < s:
                CostoReposicion += Costo_Unidad_Orden * (S - inventario) + Costo_Orden
                inventario = 50
            else:
                CostoFaltante += (inventario - s) * CostoUnidadFaltante
        
        else:
            CostoTiempoExtra += CostoTiempoExtraUnitario

        # Agregamos valores de tiempo actual, largo cola e inventario a su respectiva lista        
        lista_tiempos.append(T)
        lista_cantidad_personas.append(LargoCola+1)
        lista_inventarios.append(inventario)


        # Revisamos caso donde llega nuevo cliente antes de que salga el cliente actual
        if TiempoProxLlegada < TiempoProxSalida:
            if LargoCola < Capacidad - 1:                                           # Cabe en el sistema
                if EstadoCompletero == 0:                                           # Si estado completero = 0, no esta atendiendo a nadie
                    EstadoCompletero = 1
                    TiempoTotalSistemaVacio += (TiempoProxLlegada - T)
                    T = TiempoProxLlegada
                    if T > Tfin:                                                    # Caso Terminal
                        TiempoProxLlegada = float('inf')
                    else:                                                           # Caso nueva llegada
                        TiempoProxLlegada = T + uniform_instance(param1, param2)
                    TiempoProxSalida = T + exponential_instance(lambda_atencion)
                    TiemposLlegada[0] = T
                else:                                                               # Completero si se encontraba atendiendo
                    T = TiempoProxLlegada
                    if T > Tfin:                                                    # Caso Terminal
                        TiempoProxLlegada = float('inf')
                    else:                                                           # Caso nueva llegada
                        TiempoProxLlegada = T + uniform_instance(param1, param2)
                    LargoCola += 1
                    TiemposLlegada[LargoCola] = T

            else:                                                                   # Nuevo cliente no cabe en el sistema
                NClientesPerdidos += 1
                CostoLocalLleno += Costo_Local_Lleno_Unitario
                T = TiempoProxLlegada
                if T > Tfin:
                    TiempoProxLlegada = float('inf')
                else:
                    TiempoProxLlegada = T + uniform_instance(param1, param2)

        # Revisamos caso donde sale el cliente actual antes de que llegue otro
        else:
            T = TiempoProxSalida
            NClientesAtendidos += 1
            demanda = binomial_instance(n, p)
            if demanda <= inventario:
                inventario -= demanda
                CompletosVendidos += demanda
            else:
                CompletosVendidos += inventario
                inventario = 0


            TiempoTotalPermanencia += (T - TiemposLlegada[0])
            TiempoPermanenciaMaximo = max(TiempoPermanenciaMaximo, (T - TiemposLlegada[0]))
            if LargoCola > 0:
                LargoCola -= 1
                TiempoProxSalida = T + exponential_instance(lambda_atencion)
                TiemposLlegada[:-1] = TiemposLlegada[1:]
                TiemposLlegada[-1] = 0
            else:
                EstadoCompletero = 0
                TiempoProxSalida = float('inf')

    lista_tiempos.append(T)
    lista_cantidad_personas.append(LargoCola+1)
    lista_inventarios.append(inventario)


    VentaCompletos = PrecioCompleto *CompletosVendidos
    gananacias = VentaCompletos - CostoInventario - CostoReposicion - CostoFaltante - CostoLocalLleno - CostoArriendo - CostoTiempoExtra


    # Resultados
    if verbose:
        print("RESULTADOS:")
        print("Clientes Atendidos:", NClientesAtendidos, "; Clientes Perdidos:", NClientesPerdidos)
        print("Tiempo Promedio de Clientes en el Sistema:", TiempoTotalPermanencia/NClientesAtendidos)
        print("Tiempo de permamencia máximo de un cliente en el sistema:", TiempoPermanenciaMaximo)
        print("Estimación de la proporción en el largo plazo en que el sistema está vacío:", TiempoTotalSistemaVacio/Tfin)
        print("Estimación de la proporción en el largo plazo de clientes perdidos:", NClientesPerdidos / (NClientesAtendidos + NClientesPerdidos))
        print("Costo de inventario:", CostoInventario)
        print("Costo de reposicion:", CostoReposicion)
        print("Venta de completos:", VentaCompletos)
        print("Completos vendidos:", CompletosVendidos)
        print("Ganancias totales:", gananacias)
        print("Lista inventarios:", lista_inventarios)
    

    return gananacias, lista_tiempos, lista_cantidad_personas, lista_inventarios 



sns.set(style="whitegrid")


# ganancias, lista_tiempos, lista_cantidad_personas, lista_inventario = simulacion_puesto_completos(10, 50, 30, 30, 2123, 60*(11-8), 4, 0.1, uniform_instance(2, 4))
# ganancias, lista_tiempos, lista_cantidad_personas, lista_inventario = simulacion_puesto_completos(10, 28, 30, 30, 2123, 60*(12-11), 7, 0.2, uniform_instance(2, 4))
# ganancias, lista_tiempos, lista_cantidad_personas, lista_inventario = simulacion_puesto_completos(10, 41, 30, 30, 2123, 60*(15-12), 7, 0.2, exponential_instance(3))
# ganancias, lista_tiempos, lista_cantidad_personas, lista_inventario = simulacion_puesto_completos(10, 33, 30, 30, 2123, 60*(17-15), 5, 0.3, exponential_instance(3))
# ganancias, lista_tiempos, lista_cantidad_personas, lista_inventario = simulacion_puesto_completos(10, 17, 30, 30, 2123, 60*(19-17), 5, 0.3, rayleigh_instance(3))
ganancias, lista_tiempos, lista_cantidad_personas, lista_inventario = simulacion_puesto_completos(10, 45, 30, 30, 2123, 60*(19-17), 8, 0.5, rayleigh_instance(3))

# Gráfico de cantidad de personas en la cola
plt.figure(figsize=(10, 6))
sns.lineplot(x=lista_tiempos, y=lista_cantidad_personas, color='red', linewidth=2, marker='o', markersize=6)
plt.xlabel('Tiempo', fontsize=14, fontweight='bold')
plt.ylabel('Cantidad de personas en la cola', fontsize=14, fontweight='bold')
plt.title('Cantidad de personas en el sistema en función del tiempo', fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Gráfico de inventario
plt.figure(figsize=(10, 6))
sns.lineplot(x=lista_tiempos, y=lista_inventario, color='blue', linewidth=2, marker='o', markersize=6)
plt.xlabel('Tiempo', fontsize=14, fontweight='bold')
plt.ylabel('Inventario', fontsize=14, fontweight='bold')
plt.title('Inventario en función del tiempo', fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Gráfico de demanda perdida
plt.figure(figsize=(10, 6))
sns.lineplot(x=lista_tiempos, y=np.cumsum(lista_cantidad_personas), color='green', linewidth=2, marker='o', markersize=6)
plt.xlabel('Tiempo', fontsize=14, fontweight='bold')
plt.ylabel('Demanda perdida', fontsize=14, fontweight='bold')
plt.title('Demanda perdida en función del tiempo', fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


### Parte B

import numpy as np
import optuna
from scipy import stats
import matplotlib.pyplot as plt

def intervalo_confianza_t_student(lista_elementos):
    promedio = np.mean(lista_elementos)
    varianza_muestral = np.var(lista_elementos, ddof=1)
    n = len(lista_elementos)
    t = stats.t.ppf(0.975, n-1)
    limite_inf = promedio - t * np.sqrt(varianza_muestral/n)
    limite_sup = promedio + t * np.sqrt(varianza_muestral/n)
    intervalo = (limite_inf, limite_sup)
    return intervalo

def ancho_intervalo_confianza_t_student(lista_elementos):
    intervalo = intervalo_confianza_t_student(lista_elementos)
    return intervalo[1] - intervalo[0]

# Simulación de una jornada de ventas de sopaipillas
def simulacion_puesto_completos(s, S, tau, K, semilla):
    np.random.seed(semilla)
    hora_actual = 8 * 60  # Comienza a las 08:00 en minutos
    inventario = S
    ganancias = 0
    demanda_perdida = 0

    while hora_actual < 24 * 60:  # Simulación hasta la medianoche
        if 8 * 60 <= hora_actual < 12 * 60:
            tiempo_entre_llegadas = np.random.uniform(2, 4)
            demanda_sopaipillas = max(1, np.random.binomial(4, 0.1))
        elif 12 * 60 <= hora_actual < 17 * 60:
            tiempo_entre_llegadas = np.random.exponential(3)
            demanda_sopaipillas = max(1, np.random.binomial(7, 0.2))
        elif 17 * 60 <= hora_actual < 24 * 60:
            tiempo_entre_llegadas = np.random.rayleigh(3)
            demanda_sopaipillas = max(1, np.random.binomial(8, 0.5))
        
        if inventario >= demanda_sopaipillas:
            ganancias += 300 * demanda_sopaipillas  # Precio por sopaipilla
            inventario -= demanda_sopaipillas
        else:
            demanda_perdida += (demanda_sopaipillas - inventario)
            ganancias += 300 * inventario
            inventario = 0
        
        hora_actual += tiempo_entre_llegadas
        
        # Revisamos inventario cada tau minutos
        if hora_actual % tau == 0:
            if inventario < s:
                reposicion = S - inventario
                inventario = S
                ganancias -= (35 + 50 * reposicion)  # Costo de reposición

    return ganancias, inventario, demanda_perdida, hora_actual

def obtener_valores_ganancias_replicas(simulacion_por_replica, n_replicas, politica):
    s, S, tau, K = politica[0], politica[1], politica[2], politica[3]
    ganancias_replicas = []
    semillas = [x for x in range(n_replicas * simulacion_por_replica + 1)]
    for j in range(n_replicas):
        ganancia_total = 0
        for i in range(simulacion_por_replica):
            semilla = semillas.pop()
            ganancias, _, _, _ = simulacion_puesto_completos(s, S, tau, K, semilla)
            ganancia_total += ganancias
        promedio_ganancias = ganancia_total / simulacion_por_replica
        ganancias_replicas.append(promedio_ganancias)
    return ganancias_replicas

# Definimos la función objetivo para Optuna
def objetivo(trial):
    tau_choices = [30, 60, 90, 120]
    K_choices = [i for i in range(5, 25, 5)]
    s_choices = [i for i in range(10, 31, 5)]
    S_choices = [i for i in range(35, 60, 5)]

    # Definimos espacio de búsqueda
    tau = trial.suggest_categorical('tau', tau_choices)
    K = trial.suggest_categorical('K', K_choices)
    s = trial.suggest_categorical('s', s_choices)
    S = trial.suggest_categorical('S', S_choices)
    
    ganancias_replicas = obtener_valores_ganancias_replicas(200, 10, [s, S, tau, K])

    # Retornamos la métrica que buscamos optimizar
    return np.mean(ganancias_replicas)

# Creación del estudio Optuna para maximizar las ganancias
study = optuna.create_study(direction='maximize')
study.optimize(objetivo, n_trials=100)
print(study.best_params)

# Obtener las ganancias de varias políticas
def obtener_replicas_politicas(politicas, n_replicas, sim_x_replica):
    replicas_politicas = {}
    for politica in politicas:
        s, S, tau, K = politica[0], politica[1], politica[2], politica[3]
        ganancias_replicas = obtener_valores_ganancias_replicas(sim_x_replica, n_replicas, politica)
        replicas_politicas[(s, S, tau, K)] = ganancias_replicas
    return replicas_politicas

# Políticas a comparar
politicas = [[10, 50, 30, 30], [20, 45, 60, 15]]  # Base y alternativa
replicas_politicas = obtener_replicas_politicas(politicas, 100, 100)

i = 0
for politica, ganancias in replicas_politicas.items():
    if i == 0:
        print(f"Política Base: {politica}")
        i += 1
    else:
        print(f"Política Alternativa: {politica}")
    print(f"Ganancia promedio: {np.mean(ganancias)}")
    print(f"Intervalo de confianza: {intervalo_confianza_t_student(ganancias)}")
    print("\n")

# Cálculo del intervalo de confianza para la diferencia de ganancias
intervalo_diferencias = []
politica_keys = list(replicas_politicas.keys())
for i in range(len(replicas_politicas[politica_keys[0]])):
    intervalo_diferencias.append(replicas_politicas[politica_keys[1]][i] - replicas_politicas[politica_keys[0]][i])

intervalo = intervalo_confianza_t_student(intervalo_diferencias)
print(f"Diferencia de las ganancias promedio: {np.mean(intervalo_diferencias)}")
print(f"Intervalo de confianza de la diferencia de las ganancias promedio: {intervalo}")


import numpy as np
import matplotlib.pyplot as plt

def generar_evolucion_ancho_intervalo(politica, max_replicas, sim_x_replica):
    s, S, tau, K = politica
    ganancias_totales = []
    evolucion_intervalo = []

    for j in range(max_replicas):
        ganancia_replica = 0
        for i in range(sim_x_replica):
            semilla = j * sim_x_replica + i
            ganancias, _, _, _ = simulacion_puesto_completos(s, S, tau, K, semilla)
            ganancia_replica += ganancias
        promedio_ganancias = ganancia_replica / sim_x_replica
        ganancias_totales.append(promedio_ganancias)
        
        if len(ganancias_totales) >= 10:
            ancho = ancho_intervalo_confianza_t_student(ganancias_totales)
            evolucion_intervalo.append(ancho)
    
    return evolucion_intervalo




politica_especifica = [10, 50, 30, 30]
max_replicas = 1000
sim_x_replica = 100

evolucion_intervalo = generar_evolucion_ancho_intervalo(politica_especifica, max_replicas, sim_x_replica)

plt.figure(figsize=(10, 6))
plt.plot(range(10, len(evolucion_intervalo) + 10), evolucion_intervalo, marker='o', color='b')
plt.xlabel('Número de Réplicas', fontsize=14)
plt.ylabel('Ancho del Intervalo de Confianza', fontsize=14)
plt.title('Evolución del Ancho del Intervalo de Confianza', fontsize=16)
plt.grid(True)
plt.show()
