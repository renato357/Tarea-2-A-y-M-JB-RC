import sys
import math
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any

# --- Importaciones de scripts ---
try:
    # Funciones y clases del script de greedy determinista
    from greedydeterminista import (
        Avion,
        leer_datos_aviones,
        greedy_priorizar_menor_Ek,
        imprimir_matriz_aterrizaje
    )
    # Funciones de costo y factibilidad del script GRASP determinista (más adecuadas para SA)
    from grasp_hc_determinista_alguna_mejora import (
        es_solucion_factible,
        calcular_costo_total_solucion
    )
except ImportError as e:
    print(f"Error importando módulos necesarios: {e}")
    print("Asegúrate de que 'greedydeterminista.py' y 'grasp_hc_determinista_alguna_mejora.py' "
          "estén en la misma carpeta o en el PYTHONPATH.")
    sys.exit(1)

# --- Generadores de Vecinos para Simulated Annealing ---

def generar_vecino_single_move(solution: List[Avion], num_total_aviones: int) -> Optional[List[Avion]]:
    """
    Genera un vecino cambiando el tiempo de aterrizaje de un avión seleccionado aleatoriamente.
    El nuevo tiempo se elige aleatoriamente dentro de la ventana [E_k, L_k] del avión.
    Devuelve una copia de la solución con la modificación.
    """
    if not solution or num_total_aviones == 0:
        return None

    vecino = copy.deepcopy(solution)
    if not vecino: 
        return None

    idx_avion_a_modificar = random.randrange(len(vecino))
    avion_obj = vecino[idx_avion_a_modificar]

    lim_inf = avion_obj.tiempo_aterrizaje_temprano
    lim_sup = avion_obj.tiempo_aterrizaje_tardio

    nuevo_tiempo: Optional[int] = avion_obj.t_seleccionado 

    if lim_inf > lim_sup:
        nuevo_tiempo = lim_inf
    elif lim_inf == lim_sup:
        nuevo_tiempo = lim_inf
    else:
        try:
            nuevo_tiempo = random.randint(lim_inf, lim_sup)
        except ValueError:
            # This can happen if lim_inf > lim_sup after all, though checks should prevent it.
            # Default to original time if error.
            print(f"Advertencia en generar_vecino_single_move: lim_inf={lim_inf}, lim_sup={lim_sup} para avión {avion_obj.id_avion}. Se mantiene t_seleccionado.")
            pass # nuevo_tiempo already holds the original t_seleccionado

    avion_obj.t_seleccionado = nuevo_tiempo
    return vecino

def generar_vecino_swap_times(solution: List[Avion], num_total_aviones: int) -> Optional[List[Avion]]:
    """
    Genera un vecino intercambiando los tiempos de aterrizaje de dos aviones seleccionados aleatoriamente.
    Devuelve una copia de la solución con la modificación.
    """
    if not solution or num_total_aviones < 2:
        return None

    vecino = copy.deepcopy(solution)
    if len(vecino) < 2: # Not enough planes to swap
        return copy.deepcopy(solution) # Return unchanged copy

    try:
        idx1, idx2 = random.sample(range(len(vecino)), 2)
    except ValueError:
        # This could happen if len(vecino) < 2, though already checked.
        return copy.deepcopy(solution)

    # Swap t_seleccionado
    avion1_t_seleccionado = vecino[idx1].t_seleccionado
    vecino[idx1].t_seleccionado = vecino[idx2].t_seleccionado
    vecino[idx2].t_seleccionado = avion1_t_seleccionado

    return vecino

def generar_vecino_reinsert_plane_at_random_valid_spot(solution: List[Avion], num_total_aviones: int) -> Optional[List[Avion]]:
    """
    Genera un vecino seleccionando un avión al azar, quitándolo temporalmente,
    y reinsertándolo en un nuevo tiempo aleatorio válido dentro de su ventana [E_k, L_k]
    que mantenga la factibilidad global de la solución.
    Si no encuentra un movimiento válido, devuelve una copia de la solución original.
    """
    if not solution or num_total_aviones == 0:
        return None

    vecino_propuesto = copy.deepcopy(solution)
    
    if not vecino_propuesto: # Additional check
        return copy.deepcopy(solution)

    idx_avion_a_reinsertar = random.randrange(len(vecino_propuesto))
    avion_k = vecino_propuesto[idx_avion_a_reinsertar]
    t_original_avion_k = avion_k.t_seleccionado

    # Crear una lista de posibles tiempos de aterrizaje para el avión_k dentro de su ventana
    posibles_tiempos: List[int] = []
    if avion_k.tiempo_aterrizaje_temprano <= avion_k.tiempo_aterrizaje_tardio:
        posibles_tiempos = list(range(avion_k.tiempo_aterrizaje_temprano, avion_k.tiempo_aterrizaje_tardio + 1))
    
    if not posibles_tiempos: # If no possible times (invalid window or zero size)
        return copy.deepcopy(solution) # Cannot do anything

    random.shuffle(posibles_tiempos)

    for nuevo_tiempo_k in posibles_tiempos:
        # Modificar el tiempo del avión_k en la copia
        avion_k.t_seleccionado = nuevo_tiempo_k
        
        # Verificar si la solución completa es factible con este nuevo tiempo
        if es_solucion_factible(vecino_propuesto, num_total_aviones, verbose=False):
            return vecino_propuesto # Se encontró un vecino factible

    # Si no se encontró ningún tiempo factible para reinsertar
    # Se revierte al tiempo original y se devuelve la copia de la solución original.
    avion_k.t_seleccionado = t_original_avion_k
    return copy.deepcopy(solution)


# --- Algoritmo de Simulated Annealing ---

def simulated_annealing_run(
    initial_solution: List[Avion],
    num_total_aviones: int,
    sa_seed: int, # Esta semilla se usará para inicializar random.seed() al inicio de esta función
    sa_config: Dict[str, Any],
    config_label: str
) -> Tuple[List[Avion], float, List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Ejecuta una instancia de Simulated Annealing.
    Devuelve: (mejor_solucion_encontrada, mejor_costo_encontrado, 
               historia_costo_actual_por_iteracion, historia_mejor_costo_hasta_iteracion)
    """
    # Fijar la semilla para esta ejecución específica de SA
    random.seed(sa_seed) 

    # Extracción de parámetros de configuración
    initial_temp = sa_config["initial_temp"]
    final_temp = sa_config["final_temp"]
    cooling_rate = sa_config["cooling_rate"]
    iter_per_temp = sa_config["iter_per_temp"]
    # Probabilidades para cada operador
    prob_single_move = sa_config.get("prob_single_move", 0.5) # Probabilidad de usar single_move
    prob_swap_times = sa_config.get("prob_swap_times", 0.3)  # Probabilidad de usar swap_times
    # La probabilidad de reinsert se deduce: 1 - prob_single_move - prob_swap_times

    print_interval = sa_config.get("print_interval", 5000) # Intervalo para mensajes de progreso
    
    # Flags de verbosidad
    verbose_run = sa_config.get("verbose_sa_run", False)
    verbose_progress = sa_config.get("verbose_sa_progress", False)
    verbose_acceptance = sa_config.get("verbose_sa_acceptance", False)
    verbose_improvement = sa_config.get("verbose_sa_improvement", False)
    verbose_operator = sa_config.get("verbose_sa_operator", False)

    current_solution = copy.deepcopy(initial_solution) # Solución actual en la iteración
    
    # Verificar factibilidad de la solución inicial
    if not es_solucion_factible(current_solution, num_total_aviones, verbose=True):
        print(f"  SA Run (Config: {config_label}, Seed {sa_seed}): ADVERTENCIA - La solución inicial NO es factible. Costo puede ser irreal.")
    
    current_cost = calcular_costo_total_solucion(current_solution) # Costo de la solución actual
    best_solution_so_far = copy.deepcopy(current_solution) # Mejor solución encontrada hasta el momento
    best_cost_so_far = current_cost # Costo de la mejor solución encontrada

    temp = initial_temp # Temperatura actual
    
    # Historias para graficar
    cost_history_current: List[Tuple[int, float]] = [(0, current_cost)] # (iteracion, costo_actual)
    cost_history_best: List[Tuple[int, float]] = [(0, best_cost_so_far)] # (iteracion, mejor_costo_hasta_ahora)
    
    global_iteration_counter = 0 # Contador global de iteraciones

    if verbose_run:
        print(f"  SA Run (Config: {config_label}, Seed {sa_seed}): Iniciando SA. Temp Inicial: {initial_temp:.2f}, "
              f"Costo Inicial: {current_cost:.2f}")

    # Bucle principal de SA: mientras la temperatura sea mayor que la final
    while temp > final_temp:
        # Bucle interno: iteraciones por cada nivel de temperatura
        for _ in range(iter_per_temp):
            global_iteration_counter += 1

            # Impresión de progreso periódica
            if verbose_progress and global_iteration_counter % print_interval == 0:
                 print(f"    SA Run (Config: {config_label}, Seed {sa_seed}) Iter: {global_iteration_counter}, Temp: {temp:.2f}, "
                       f"Costo Actual: {current_cost:.2f}, Mejor Costo: {best_cost_so_far:.2f}")

            # Selección de operador de movimiento basado en probabilidades
            rand_val = random.random()
            neighbor_solution_propuesta: Optional[List[Avion]] = None
            operator_used = ""

            if rand_val < prob_single_move:
                neighbor_solution_propuesta = generar_vecino_single_move(current_solution, num_total_aviones)
                operator_used = "SINGLE_MOVE"
            elif rand_val < prob_single_move + prob_swap_times:
                neighbor_solution_propuesta = generar_vecino_swap_times(current_solution, num_total_aviones)
                operator_used = "SWAP_TIMES"
            else: # El resto de la probabilidad es para reinsert
                neighbor_solution_propuesta = generar_vecino_reinsert_plane_at_random_valid_spot(current_solution, num_total_aviones)
                operator_used = "REINSERT_PLANE"
            
            if verbose_operator and global_iteration_counter % print_interval == 0:
                print(f"      SA Iter {global_iteration_counter}: Intentando Operador -> {operator_used}")

            # Si el operador no pudo generar un vecino (raro, pero por seguridad)
            if neighbor_solution_propuesta is None:
                cost_history_current.append((global_iteration_counter, current_cost))
                cost_history_best.append((global_iteration_counter, best_cost_so_far))
                continue

            # Verificar si la solución propuesta es diferente de la actual
            # (relevante si reinsert_plane devuelve la original por no encontrar movimiento)
            # Comparamos los tiempos seleccionados para cada avión por ID
            current_times_map = {a.id_avion: a.t_seleccionado for a in current_solution}
            neighbor_times_map = {a.id_avion: a.t_seleccionado for a in neighbor_solution_propuesta}

            if current_times_map == neighbor_times_map:
                # El operador no produjo un cambio real
                cost_history_current.append((global_iteration_counter, current_cost))
                cost_history_best.append((global_iteration_counter, best_cost_so_far))
                continue

            # Evaluar el vecino propuesto solo si es factible
            if es_solucion_factible(neighbor_solution_propuesta, num_total_aviones, verbose=False):
                neighbor_cost = calcular_costo_total_solucion(neighbor_solution_propuesta)
                delta_cost = neighbor_cost - current_cost

                # Criterio de Metropolis
                if delta_cost < 0: # Si el vecino es mejor
                    current_solution = copy.deepcopy(neighbor_solution_propuesta)
                    current_cost = neighbor_cost
                    if current_cost < best_cost_so_far: # Si es mejor que la mejor global encontrada
                        if verbose_improvement:
                             print(f"    SA Iter {global_iteration_counter} (T={temp:.2f}, Config: {config_label}): "
                                   f"Nueva mejor solución SA. Costo: {current_cost:.2f} (anterior mejor: {best_cost_so_far:.2f})")
                        best_solution_so_far = copy.deepcopy(current_solution)
                        best_cost_so_far = current_cost
                else: # Si el vecino es peor o igual
                    acceptance_probability = math.exp(-delta_cost / temp) if temp > 1e-9 else 0.0 # Evitar división por cero
                    if random.random() < acceptance_probability: # Aceptar con cierta probabilidad
                        if verbose_acceptance:
                            print(f"    SA Iter {global_iteration_counter} (T={temp:.2f}, Config: {config_label}): "
                                  f"Aceptado peor sol. Costo Actual: {current_cost:.2f} -> Nuevo Costo: {neighbor_cost:.2f} "
                                  f"(Prob: {acceptance_probability:.3f}, Delta: {delta_cost:.2f})")
                        current_solution = copy.deepcopy(neighbor_solution_propuesta)
                        current_cost = neighbor_cost
            # Si el vecino no es factible, se ignora y current_solution/current_cost no cambian.
            
            cost_history_current.append((global_iteration_counter, current_cost))
            cost_history_best.append((global_iteration_counter, best_cost_so_far))
        
        # Enfriar la temperatura
        temp *= cooling_rate

    if verbose_run:
        print(f"  SA Run (Config: {config_label}, Seed {sa_seed}): SA Finalizado. Mejor costo encontrado: {best_cost_so_far:.2f} "
              f"en {global_iteration_counter} iteraciones.")

    return best_solution_so_far, best_cost_so_far, cost_history_current, cost_history_best

# --- Función Principal del Runner SA ---
def main_sa_runner():
    """
    Función principal para:
    1. Generar la solución inicial con el greedy determinista.
    2. Ejecutar Simulated Annealing con múltiples configuraciones sobre esa solución, usando una semilla global.
    3. Mostrar resultados y graficar la evolución del mejor costo encontrado.
    """
    ruta_archivo_datos = "case4.txt" # Ajustar según el caso de prueba
    
    # Semilla única para toda la ejecución, garantizando replicabilidad.
    # ESTE ES EL VALOR QUE SE USARÁ PARA TODAS LAS EJECUCIONES DE SA.
    SEMILLA_GLOBAL_REPLICACION = 42 

    # Configuraciones de Simulated Annealing
    sa_configurations = [
        {"label": "Config1_HighT_SlowCool", "initial_temp": 100000.0, "final_temp": 0.1, "cooling_rate": 0.995, "iter_per_temp": 100, "prob_single_move": 0.5, "prob_swap_times": 0.3, "print_interval": 20000, "verbose_sa_run": True, "verbose_sa_progress": True, "verbose_sa_improvement": True},
        {"label": "Config2_MedT_MedCool", "initial_temp": 50000.0, "final_temp": 0.1, "cooling_rate": 0.99, "iter_per_temp": 150, "prob_single_move": 0.4, "prob_swap_times": 0.3, "print_interval": 15000, "verbose_sa_run": True, "verbose_sa_progress": True, "verbose_sa_improvement": True},
        {"label": "Config3_LowT_FastCool_MoreReinsert", "initial_temp": 10000.0, "final_temp": 0.01, "cooling_rate": 0.98, "iter_per_temp": 200, "prob_single_move": 0.3, "prob_swap_times": 0.3, "print_interval": 10000, "verbose_sa_run": True, "verbose_sa_progress": True, "verbose_sa_improvement": True}, 
        {"label": "Config4_HighIter_BalancedOps", "initial_temp": 50000.0, "final_temp": 0.1, "cooling_rate": 0.99, "iter_per_temp": 300, "prob_single_move": 0.33, "prob_swap_times": 0.33, "print_interval": 25000, "verbose_sa_run": True, "verbose_sa_progress": True, "verbose_sa_improvement": True}, 
        {"label": "Config5_FocusSingleMove", "initial_temp": 50000.0, "final_temp": 0.1, "cooling_rate": 0.99, "iter_per_temp": 150, "prob_single_move": 0.7, "prob_swap_times": 0.15, "print_interval": 15000, "verbose_sa_run": True, "verbose_sa_progress": True, "verbose_sa_improvement": True, "verbose_sa_acceptance": False, "verbose_sa_operator": False}, 
    ]
    
    print(">>> Iniciando Proceso de Simulated Annealing (con Greedy Determinista y Nuevos Operadores) <<<")
    print(f"Archivo de datos: {ruta_archivo_datos}")
    print(f"USANDO SEMILLA GLOBAL PARA REPLICACIÓN EN TODAS LAS CONFIGURACIONES DE SA: {SEMILLA_GLOBAL_REPLICACION}")


    print("\n--- Paso 1: Generando Solución Inicial con Greedy Determinista ---")
    try:
        lista_aviones_base = leer_datos_aviones(ruta_archivo_datos)
        if not lista_aviones_base:
            print(f"Error: No se pudieron leer aviones desde '{ruta_archivo_datos}'. Abortando.")
            return
        num_aviones = len(lista_aviones_base)
        print(f"Leídos {num_aviones} aviones desde '{ruta_archivo_datos}'.")

        # El greedy determinista no usa random, así que no se ve afectado por la semilla global directamente aquí.
        solucion_greedy_determinista, factible_greedy, _ = greedy_priorizar_menor_Ek(copy.deepcopy(lista_aviones_base))

        if not factible_greedy:
            print("ADVERTENCIA: La solución inicial del Greedy Determinista NO es factible según el propio greedy.")
        if not es_solucion_factible(solucion_greedy_determinista, num_aviones, verbose=True):
            print("ADVERTENCIA ADICIONAL: Solución Greedy no pasó la verificación de 'es_solucion_factible' que usará SA.")
        
        costo_greedy_determinista = calcular_costo_total_solucion(solucion_greedy_determinista)
        print(f"Solución Inicial del Greedy Determinista (Factible por greedy: {factible_greedy}):")
        print(f"Costo del Greedy Determinista: {costo_greedy_determinista:.2f}")
        imprimir_matriz_aterrizaje(solucion_greedy_determinista) # Llamada corregida

    except FileNotFoundError:
        print(f"Error CRÍTICO: No se pudo encontrar el archivo de datos '{ruta_archivo_datos}'.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Ocurrió un error preparando la solución inicial: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    if num_aviones == 0: # Seguridad adicional
        print("Error: No se pudo determinar el número de aviones. Abortando SA.")
        return

    print("\n--- Paso 2: Ejecutando Simulated Annealing para Múltiples Configuraciones ---")
    
    # Almacenará las historias del mejor costo encontrado para cada configuración
    todas_las_historias_mejores_costos_sa: List[Tuple[str, List[Tuple[int, float]]]] = []
    # Almacenará los resultados finales de cada corrida de SA
    resultados_finales_sa: List[Dict[str, Any]] = []
    
    # El bucle itera sobre las configuraciones, pero la semilla para SA será siempre la misma.
    for i, config in enumerate(sa_configurations): 
        config_label = config.get("label", f"Config{i+1}")
        print(f"\nProcesando con SA - {config_label}:")
        
        # SE PASA LA MISMA SEMILLA GLOBAL (42) A CADA EJECUCIÓN DE SA
        mejor_sol_sa, mejor_costo_sa, _, historia_mejor_costo_run = simulated_annealing_run(
            initial_solution=copy.deepcopy(solucion_greedy_determinista), 
            num_total_aviones=num_aviones,
            sa_seed=SEMILLA_GLOBAL_REPLICACION, # Usar la semilla global constante (42)
            sa_config=config,
            config_label=config_label
        )
        
        # Guardar la historia del mejor costo para el gráfico consolidado
        todas_las_historias_mejores_costos_sa.append((config_label, historia_mejor_costo_run))
        # Guardar los resultados finales de esta corrida
        resultados_finales_sa.append({
            "config_label": config_label,
            "sa_seed_usada": SEMILLA_GLOBAL_REPLICACION, # Registrar la semilla usada (siempre 42)
            "final_cost_sa": mejor_costo_sa,
            "best_solution_sa": copy.deepcopy(mejor_sol_sa), # Guardar copia de la mejor solución
            "sa_params": config # Guardar los parámetros usados
        })

    # --- Resultados Finales y Gráficos ---
    print("\n\n========================= RESULTADOS FINALES DEL PROCESO SA (Determinista con Nuevos Operadores) =========================")
    print(f"Costo Inicial (Greedy Determinista): {costo_greedy_determinista:.2f}")
    print(f"Semilla Global de Replicación Utilizada para TODAS las corridas de SA: {SEMILLA_GLOBAL_REPLICACION}")
    print("\nResumen de costos por configuración de SA:")
    
    mejor_costo_global_todas_sa = float('inf')
    mejor_solucion_global_obj_sa = None
    mejor_run_info_sa = {}

    for res in resultados_finales_sa:
        costo_sa_str = f"{res['final_cost_sa']:.2f}" if res['final_cost_sa'] != float('inf') else "Inf"
        print(f"  Config SA: {res['config_label']} (SA Seed Usada: {res['sa_seed_usada']}) -> Costo SA Final = {costo_sa_str}")
        # La mejor solución global se determina entre las configuraciones, 
        # aunque todas usaron la misma semilla, sus parámetros son diferentes.
        if res['final_cost_sa'] < mejor_costo_global_todas_sa:
            mejor_costo_global_todas_sa = res['final_cost_sa']
            mejor_solucion_global_obj_sa = res['best_solution_sa']
            mejor_run_info_sa = res 

    if mejor_solucion_global_obj_sa:
        print(f"\nMejor solución global encontrada por SA (considerando todas las configs con semilla {SEMILLA_GLOBAL_REPLICACION}): Costo = {mejor_costo_global_todas_sa:.2f}")
        print(f"  Originada por Config SA: {mejor_run_info_sa.get('config_label', 'N/A')}, SA Seed Usada: {mejor_run_info_sa.get('sa_seed_usada', 'N/A')}")
        print(f"  Parámetros de la mejor config: {mejor_run_info_sa.get('sa_params')}")
        
        # Imprimir la mejor solución global
        print(f"\nHorario de la mejor solución global SA (Config: {mejor_run_info_sa.get('config_label', 'N/A')})")
        imprimir_matriz_aterrizaje(mejor_solucion_global_obj_sa)
        
        if not es_solucion_factible(mejor_solucion_global_obj_sa, num_aviones, verbose=True):
             print("ALERTA CRÍTICA: La mejor solución SA global reportada NO es factible.")
        else:
             print("La mejor solución SA global reportada ES factible.")
    else:
        print("\nNo se encontró ninguna solución SA válida en ninguna configuración.")

    # --- Gráfico: Evolución del Mejor Costo Encontrado por Iteración (Consolidado) ---
    plt.figure(figsize=(15, 8))
    plot_successful_best_cost = False
    max_iter_overall = 0 # Para ajustar el eje x si es necesario

    for config_label, historia_mejor_costo in todas_las_historias_mejores_costos_sa:
        if historia_mejor_costo and historia_mejor_costo[0][1] != float('inf') and historia_mejor_costo[0][1] < 1e11 : # Evitar graficar si el costo es inf o muy grande
            iteraciones = [item[0] for item in historia_mejor_costo]
            costos_mejores = [item[1] for item in historia_mejor_costo]
            if iteraciones: # Asegurarse que hay datos
                 max_iter_overall = max(max_iter_overall, iteraciones[-1])
            plt.plot(iteraciones, costos_mejores, label=f"Mejor SA - {config_label}")
            plot_successful_best_cost = True
        else:
            print(f"Nota: Historia del mejor costo para SA con config '{config_label}' no se grafica (costo inicial inf o sin datos).")

    if plot_successful_best_cost:
        plt.xlabel("Iteración Global de SA")
        plt.ylabel("Mejor Costo de Solución Encontrado Hasta la Iteración")
        titulo_grafico_mejor = (f"Evolución del Mejor Costo Encontrado en SA para {ruta_archivo_datos}\n"
                                f"(Desde Greedy Determinista, Costo Inicial: {costo_greedy_determinista:.2f})\n"
                                f"Semilla Global para todas las corridas SA: {SEMILLA_GLOBAL_REPLICACION}")
        plt.title(titulo_grafico_mejor)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) # Leyenda fuera del área
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.80, 1]) # Ajustar para que la leyenda no se corte

        nombre_grafico_mejor = f"sa_determinista_mejor_costo_evolucion_{ruta_archivo_datos.split('.')[0]}_sem{SEMILLA_GLOBAL_REPLICACION}.png"
        try:
            plt.savefig(nombre_grafico_mejor)
            print(f"\nGráfico de mejor costo guardado como: {nombre_grafico_mejor}")
        except Exception as e_plot:
            print(f"Error al guardar el gráfico de mejor costo: {e_plot}")
        plt.show() # Mostrar el gráfico
    else:
        print("\nNo se generó ningún gráfico de mejor costo porque no hubo ejecuciones de SA válidas para graficar.")

if __name__ == "__main__":
    main_sa_runner()
