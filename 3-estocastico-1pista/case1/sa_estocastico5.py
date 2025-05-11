# -*- coding: utf-8 -*-
import sys
import math
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any

# --- Importaciones de tus scripts ---
try:
    # Funciones y clases del script de greedy estocástico
    from greedy_estocastico_10_semillas import (
        Avion,
        leer_datos_aviones,
        calcular_costo_total as calcular_costo_total_solucion_estocastico,
        imprimir_matriz_aterrizaje as imprimir_matriz_aterrizaje_estocastico,
        greedy_stochastico_Ek_ponderado_costo, # Mantenido por si se necesita referencia directa
        buscar_solucion_factible_estocastica, # CLAVE: Para generar la misma solución inicial que el script greedy
        calcular_costo_en_tiempo_especifico
    )
    # Función de factibilidad del script de GRASP/HC (considerada robusta)
    from grasp_hc_estocastico_alguna_mejora import es_solucion_factible

except ImportError as e:
    print(f"Error importando módulos necesarios: {e}")
    print("Asegúrate de que 'greedy_estocastico_10_semillas.py' y 'grasp_hc_estocastico_alguna_mejora.py' "
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

    if lim_inf > lim_sup: # Si la ventana es inválida
        nuevo_tiempo = lim_inf
    elif lim_inf == lim_sup:
        nuevo_tiempo = lim_inf
    else:
        try:
            nuevo_tiempo = random.randint(lim_inf, lim_sup)
        except ValueError:
            # Esto puede ocurrir si lim_inf > lim_sup después de todo.
            # Se mantiene el t_seleccionado original en nuevo_tiempo.
            print(
                f"Advertencia en generar_vecino_single_move: lim_inf={lim_inf}, lim_sup={lim_sup} para avión {avion_obj.id_avion}. Se mantiene t_seleccionado.")
            pass

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
    if len(vecino) < 2:
        return copy.deepcopy(solution)

    try:
        idx1, idx2 = random.sample(range(len(vecino)), 2)
    except ValueError:
        return copy.deepcopy(solution)

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

    if not vecino_propuesto:
        return copy.deepcopy(solution)

    idx_avion_a_reinsertar = random.randrange(len(vecino_propuesto))
    avion_k = vecino_propuesto[idx_avion_a_reinsertar]
    t_original_avion_k = avion_k.t_seleccionado

    posibles_tiempos: List[int] = []
    if avion_k.tiempo_aterrizaje_temprano <= avion_k.tiempo_aterrizaje_tardio:
        posibles_tiempos = list(range(avion_k.tiempo_aterrizaje_temprano, avion_k.tiempo_aterrizaje_tardio + 1))

    if not posibles_tiempos:
        return copy.deepcopy(solution)

    random.shuffle(posibles_tiempos)

    for nuevo_tiempo_k in posibles_tiempos:
        avion_k.t_seleccionado = nuevo_tiempo_k
        if es_solucion_factible(vecino_propuesto, num_total_aviones, verbose=False):
            return vecino_propuesto

    avion_k.t_seleccionado = t_original_avion_k
    return copy.deepcopy(solution)


# --- Algoritmo de Simulated Annealing ---

def simulated_annealing_run(
        initial_solution: List[Avion],
        num_total_aviones: int,
        sa_seed: int,
        sa_config: Dict[str, Any],
        config_label: str
) -> Tuple[List[Avion], float, List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Ejecuta una instancia de Simulated Annealing.
    Devuelve: (mejor_solucion_encontrada, mejor_costo_encontrado,
               historia_costo_actual_por_iteracion, historia_mejor_costo_hasta_iteracion)
    """
    random.seed(sa_seed) # Fijar la semilla para esta ejecución específica de SA

    initial_temp = sa_config["initial_temp"]
    final_temp = sa_config["final_temp"]
    cooling_rate = sa_config["cooling_rate"]
    iter_per_temp = sa_config["iter_per_temp"]
    prob_single_move = sa_config.get("prob_single_move", 0.5)
    prob_swap_times = sa_config.get("prob_swap_times", 0.3)

    print_interval = sa_config.get("print_interval", 5000)
    verbose_run = sa_config.get("verbose_sa_run", False)
    verbose_progress = sa_config.get("verbose_sa_progress", False)
    verbose_acceptance = sa_config.get("verbose_sa_acceptance", False)
    verbose_improvement = sa_config.get("verbose_sa_improvement", False)
    verbose_operator = sa_config.get("verbose_sa_operator", False)

    current_solution = copy.deepcopy(initial_solution)

    if not es_solucion_factible(current_solution, num_total_aviones, verbose=True):
        print(
            f"  SA Run (Config: {config_label}, Seed SA: {sa_seed}): ADVERTENCIA - La solución inicial NO es factible. Costo puede ser irreal.")

    current_cost = calcular_costo_total_solucion_estocastico(current_solution)
    best_solution_so_far = copy.deepcopy(current_solution)
    best_cost_so_far = current_cost

    temp = initial_temp
    cost_history_current: List[Tuple[int, float]] = [(0, current_cost)]
    cost_history_best: List[Tuple[int, float]] = [(0, best_cost_so_far)]
    global_iteration_counter = 0

    if verbose_run:
        print(
            f"  SA Run (Config: {config_label}, Seed SA: {sa_seed}): Iniciando SA. Temp Inicial: {initial_temp:.2f}, "
            f"Costo Inicial: {current_cost:.2f}")

    while temp > final_temp:
        for _ in range(iter_per_temp):
            global_iteration_counter += 1

            if verbose_progress and global_iteration_counter % print_interval == 0:
                print(
                    f"    SA Run (Config: {config_label}, Seed SA: {sa_seed}) Iter: {global_iteration_counter}, Temp: {temp:.2f}, "
                    f"Costo Actual: {current_cost:.2f}, Mejor Costo: {best_cost_so_far:.2f}")

            rand_val = random.random()
            neighbor_solution_propuesta: Optional[List[Avion]] = None
            operator_used = ""

            if rand_val < prob_single_move:
                neighbor_solution_propuesta = generar_vecino_single_move(
                    current_solution, num_total_aviones)
                operator_used = "SINGLE_MOVE"
            elif rand_val < prob_single_move + prob_swap_times:
                neighbor_solution_propuesta = generar_vecino_swap_times(
                    current_solution, num_total_aviones)
                operator_used = "SWAP_TIMES"
            else:
                neighbor_solution_propuesta = generar_vecino_reinsert_plane_at_random_valid_spot(
                    current_solution, num_total_aviones)
                operator_used = "REINSERT_PLANE"

            if verbose_operator and global_iteration_counter % print_interval == 0 :
                print(f"      SA Iter {global_iteration_counter}: Intentando Operador -> {operator_used}")

            if neighbor_solution_propuesta is None:
                cost_history_current.append(
                    (global_iteration_counter, current_cost))
                cost_history_best.append(
                    (global_iteration_counter, best_cost_so_far))
                continue

            current_times_map = {
                a.id_avion: a.t_seleccionado for a in current_solution}
            neighbor_times_map = {
                a.id_avion: a.t_seleccionado for a in neighbor_solution_propuesta}

            if current_times_map == neighbor_times_map:
                cost_history_current.append(
                    (global_iteration_counter, current_cost))
                cost_history_best.append(
                    (global_iteration_counter, best_cost_so_far))
                continue

            if es_solucion_factible(neighbor_solution_propuesta, num_total_aviones, verbose=False):
                neighbor_cost = calcular_costo_total_solucion_estocastico(neighbor_solution_propuesta)
                delta_cost = neighbor_cost - current_cost

                if delta_cost < 0:
                    current_solution = copy.deepcopy(neighbor_solution_propuesta)
                    current_cost = neighbor_cost
                    if current_cost < best_cost_so_far:
                        if verbose_improvement:
                            print(
                                f"    SA Iter {global_iteration_counter} (T={temp:.2f}, Config: {config_label}, Seed SA: {sa_seed}): "
                                f"Nueva mejor solución SA. Costo: {current_cost:.2f} (anterior mejor: {best_cost_so_far:.2f})")
                        best_solution_so_far = copy.deepcopy(current_solution)
                        best_cost_so_far = current_cost
                else:
                    acceptance_probability = math.exp(-delta_cost / temp) if temp > 1e-9 else 0.0
                    if random.random() < acceptance_probability:
                        if verbose_acceptance:
                            print(
                                f"    SA Iter {global_iteration_counter} (T={temp:.2f}, Config: {config_label}, Seed SA: {sa_seed}): "
                                f"Aceptado peor sol. Costo Actual: {current_cost:.2f} -> Nuevo Costo: {neighbor_cost:.2f} "
                                f"(Prob: {acceptance_probability:.3f}, Delta: {delta_cost:.2f})")
                        current_solution = copy.deepcopy(neighbor_solution_propuesta)
                        current_cost = neighbor_cost

            cost_history_current.append(
                (global_iteration_counter, current_cost))
            cost_history_best.append(
                (global_iteration_counter, best_cost_so_far))

        temp *= cooling_rate

    if verbose_run:
        print(
            f"  SA Run (Config: {config_label}, Seed SA: {sa_seed}): SA Finalizado. Mejor costo encontrado: {best_cost_so_far:.2f} "
            f"en {global_iteration_counter} iteraciones.")

    return best_solution_so_far, best_cost_so_far, cost_history_current, cost_history_best


# --- Función Principal del Runner SA ---
def main_sa_runner():
    """
    Función principal para:
    1. Generar 10 soluciones iniciales con el greedy estocástico (cada una con su propia semilla y método de búsqueda).
    2. Ejecutar Simulated Annealing para cada solución inicial, probando las 5 configuraciones de SA.
    3. Generar 10 gráficos, uno para cada solución inicial, mostrando la evolución del costo para las 5 configuraciones.
    """
    ruta_archivo_datos = "case1.txt"

    # Volver a la lista completa de 10 semillas
    semillas_greedy_originales = [42, 123, 7, 99, 500, 777, 2024, 1, 100, 314]
    num_soluciones_iniciales = len(semillas_greedy_originales)
    MAX_INTENTOS_GREEDY_POR_SEMILLA = 1000

    # Configuraciones de Simulated Annealing
    sa_configurations = [
        {
            "label": "Config1_Mod_ExploitFocus_From_C2", # Etiqueta modificada para claridad
            "initial_temp": 20000.0,  # De Config2
            "final_temp": 1.0,        # De Config2
            "cooling_rate": 0.97,     # De Config2
            "iter_per_temp": 150,     # Aumentado (original de C2 era 50) para más explotación por temperatura
            "prob_single_move": 0.7,  # Aumentado (original de C2 era 0.4) para favorecer movimientos locales
            "prob_swap_times": 0.1,   # Disminuido (original de C2 era 0.3) para reducir exploración
                                      # Probabilidad implícita de reinsert_plane = 1 - 0.7 - 0.1 = 0.2
            "print_interval": 5000,   # De Config2
            "verbose_sa_run": True,   # De Config2
            "verbose_sa_progress": False, # De Config2
            "verbose_sa_improvement": True, # De Config2
            "verbose_sa_acceptance": False, # Añadido explícitamente para consistencia
            "verbose_sa_operator": False    # Añadido explícitamente para consistencia
        },
        {"label": "Config2_Fast_LowIter_MedT", "initial_temp": 20000.0, "final_temp": 1.0, "cooling_rate": 0.97,
         "iter_per_temp": 50, "prob_single_move": 0.4, "prob_swap_times": 0.3, "print_interval": 5000,
         "verbose_sa_run": True, "verbose_sa_progress": False, "verbose_sa_improvement": True},
        {"label": "Config3_VeryFast_VLowIter_LowT", "initial_temp": 5000.0, "final_temp": 1.0, "cooling_rate": 0.95,
         "iter_per_temp": 30, "prob_single_move": 0.5, "prob_swap_times": 0.25, "print_interval": 3000,
         "verbose_sa_run": True, "verbose_sa_progress": False, "verbose_sa_improvement": True},
        {"label": "Config4_HighIter_BalancedOps", "initial_temp": 50000.0, "final_temp": 0.1, "cooling_rate": 0.99,
         "iter_per_temp": 300, "prob_single_move": 0.33, "prob_swap_times": 0.33, "print_interval": 15000,
         "verbose_sa_run": True, "verbose_sa_progress": True, "verbose_sa_improvement": True},
        {"label": "Config5_Fast_RapidCool_MedIter", "initial_temp": 30000.0, "final_temp": 0.5, "cooling_rate": 0.96,
         "iter_per_temp": 70, "prob_single_move": 0.6, "prob_swap_times": 0.2, "print_interval": 7000,
         "verbose_sa_run": True, "verbose_sa_progress": False, "verbose_sa_improvement": True,
         "verbose_sa_acceptance": False, "verbose_sa_operator": False},
    ]
    num_sa_configurations = len(sa_configurations)

    print(f">>> Iniciando Proceso de SA Estocástico ({num_soluciones_iniciales} Soluciones Iniciales x {num_sa_configurations} Configuraciones) <<<")
    print(f"Archivo de datos: {ruta_archivo_datos}")
    print(f"Número de soluciones iniciales a generar: {num_soluciones_iniciales}")

    try:
        lista_aviones_base = leer_datos_aviones(ruta_archivo_datos)
        if not lista_aviones_base:
            print(f"Error: No se pudieron leer aviones desde '{ruta_archivo_datos}'. Abortando.")
            return
        num_aviones = len(lista_aviones_base)
        print(f"Leídos {num_aviones} aviones desde '{ruta_archivo_datos}'.")

    except FileNotFoundError:
        print(f"Error CRÍTICO: No se pudo encontrar el archivo de datos '{ruta_archivo_datos}'.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Ocurrió un error leyendo los datos: {e}", file=sys.stderr)
        return

    if num_aviones == 0:
        print("Error: No se pudo determinar el número de aviones. Abortando.")
        return

    # --- Paso 1: Generar 10 Soluciones Iniciales con Greedy Estocástico ---
    print(f"\n--- Paso 1: Generando {num_soluciones_iniciales} Solución(es) Inicial(es) con Greedy Estocástico ---")
    lista_soluciones_iniciales_greedy: List[Optional[List[Avion]]] = []
    costos_iniciales_greedy: List[float] = []

    for i, semilla_g in enumerate(semillas_greedy_originales):
        print(f"\nGenerando Solución Inicial Greedy #{i + 1} con semilla: {semilla_g}")
        random.seed(semilla_g)

        aviones_para_greedy_actual = copy.deepcopy(lista_aviones_base)

        solucion_g_obj, costo_g, intentos_g, fue_factible_g = buscar_solucion_factible_estocastica(
            lista_aviones_original_raw=aviones_para_greedy_actual,
            max_intentos_greedy=MAX_INTENTOS_GREEDY_POR_SEMILLA,
            verbose_greedy_internal=False
        )

        if fue_factible_g and solucion_g_obj:
            print(f"  Solución Greedy #{i+1} (semilla {semilla_g}) encontrada en {intentos_g} intento(s) del greedy. Costo: {costo_g:.2f}")
            if not es_solucion_factible(solucion_g_obj, num_aviones, verbose=True):
                 print(f"  ADVERTENCIA ADICIONAL: Solución Greedy #{i + 1} (semilla {semilla_g}) no pasó la verificación de 'es_solucion_factible' externa.")
            imprimir_matriz_aterrizaje_estocastico(solucion_g_obj, f"Solución Inicial Greedy #{i + 1} (Semilla {semilla_g})")
            lista_soluciones_iniciales_greedy.append(copy.deepcopy(solucion_g_obj))
            costos_iniciales_greedy.append(costo_g)
        else:
            print(f"  ADVERTENCIA: No se encontró solución Greedy factible para semilla {semilla_g} después de {MAX_INTENTOS_GREEDY_POR_SEMILLA} intentos.")
            lista_soluciones_iniciales_greedy.append(None)
            costos_iniciales_greedy.append(float('inf'))


    # --- Paso 2: Ejecutar Simulated Annealing para cada Solución Inicial y Configuración ---
    print("\n--- Paso 2: Ejecutando Simulated Annealing para cada Solución Inicial y Configuración ---")
    todos_los_resultados_sa = []

    for idx_sol_inicial, solucion_inicial_g in enumerate(lista_soluciones_iniciales_greedy):
        semilla_greedy_usada = semillas_greedy_originales[idx_sol_inicial]
        costo_inicial_g = costos_iniciales_greedy[idx_sol_inicial]

        if solucion_inicial_g is None:
            print(
                f"\nSaltando SA para la solución inicial Greedy #{idx_sol_inicial + 1} (Semilla: {semilla_greedy_usada}) porque no fue generada/factible.")
            todos_los_resultados_sa.append({
                "greedy_seed": semilla_greedy_usada,
                "resultados_por_config": [],
                "solucion_inicial_greedy": None,
                "costo_inicial_greedy": float('inf')
            })
            continue

        print(
            f"\nEjecutando SA para la solución inicial Greedy #{idx_sol_inicial + 1} (Semilla Greedy: {semilla_greedy_usada}, Costo Inicial: {costo_inicial_g:.2f})")
        resultados_sa_para_esta_inicial = []

        for idx_config, config_sa in enumerate(sa_configurations):
            config_label = config_sa.get("label", f"Config{idx_config + 1}")
            print(f"\n  Configuración SA: {config_label}")

            semilla_sa_para_esta_config = (idx_sol_inicial + 1) * 1000 + (idx_config + 1) * 100

            mejor_sol_sa, mejor_costo_sa, historia_costo_actual_sa, historia_mejor_costo_sa = simulated_annealing_run(
                initial_solution=copy.deepcopy(solucion_inicial_g),
                num_total_aviones=num_aviones,
                sa_seed=semilla_sa_para_esta_config,
                sa_config=config_sa,
                config_label=config_label
            )

            resultados_sa_para_esta_inicial.append({
                "config_label": config_label,
                "sa_seed_usada": semilla_sa_para_esta_config,
                "final_cost_sa": mejor_costo_sa,
                "best_solution_sa": copy.deepcopy(mejor_sol_sa),
                "historia_costo_actual": historia_costo_actual_sa,
                "historia_mejor_costo": historia_mejor_costo_sa,
                "sa_params": config_sa
            })

        todos_los_resultados_sa.append({
            "greedy_seed": semilla_greedy_usada,
            "resultados_por_config": resultados_sa_para_esta_inicial,
            "solucion_inicial_greedy": copy.deepcopy(solucion_inicial_g),
            "costo_inicial_greedy": costo_inicial_g
        })

    # --- Paso 3: Generar Gráficos (ahora hasta 10) ---
    print(f"\n--- Paso 3: Generando {num_soluciones_iniciales} Gráfico(s) de Evolución del Costo ---")

    for idx_sol_inicial_plot, data_resultados_inicial in enumerate(todos_los_resultados_sa):
        semilla_g_plot = data_resultados_inicial["greedy_seed"]
        costo_g_plot = data_resultados_inicial["costo_inicial_greedy"]

        if not data_resultados_inicial["resultados_por_config"]:
            print(f"  No se generará gráfico para Sol. Inicial Greedy #{idx_sol_inicial_plot + 1} (Semilla: {semilla_g_plot}) porque no tuvo ejecuciones de SA.")
            continue

        plt.figure(figsize=(15, 8))
        plot_successful_lines = 0

        for resultado_config_plot in data_resultados_inicial["resultados_por_config"]:
            label_config_plot = resultado_config_plot["config_label"]
            historia_actual_plot = resultado_config_plot["historia_costo_actual"]
            sa_seed_plot = resultado_config_plot["sa_seed_usada"]
            costo_final_sa_plot = resultado_config_plot["final_cost_sa"]

            if historia_actual_plot and historia_actual_plot[0][1] != float('inf') and historia_actual_plot[0][1] < 1e12:
                iteraciones_plot = [item[0] for item in historia_actual_plot]
                costos_plot = [item[1] for item in historia_actual_plot]
                plt.plot(iteraciones_plot, costos_plot,
                         label=f"SA {label_config_plot} (Costo Final: {costo_final_sa_plot:.2f}, Seed SA: {sa_seed_plot})")
                plot_successful_lines +=1

        if plot_successful_lines > 0:
            plt.xlabel("Iteración de SA")
            plt.ylabel("Costo Actual de la Solución")
            costo_g_plot_str = f"{costo_g_plot:.2f}" if costo_g_plot != float('inf') else 'N/A'
            titulo_grafico = (
                f"Evolución del Costo SA para Sol. Inicial Greedy #{idx_sol_inicial_plot + 1} (Seed Greedy: {semilla_g_plot})\n"
                f"Costo Inicial Greedy: {costo_g_plot_str}")
            plt.title(titulo_grafico)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.grid(True)
            plt.tight_layout(rect=[0, 0, 0.75, 1])

            nombre_grafico = f"sa_estocastico_evolucion_costo_greedy_seed_{semilla_g_plot}.png"
            try:
                plt.savefig(nombre_grafico)
                print(f"  Gráfico guardado como: {nombre_grafico}")
            except Exception as e_plot_save:
                print(f"  Error al guardar el gráfico '{nombre_grafico}': {e_plot_save}")
            # plt.show() # Descomentar para mostrar interactivamente
            plt.close()
        else:
            print(
                f"  No se generó gráfico para Sol. Inicial Greedy #{idx_sol_inicial_plot + 1} (Seed: {semilla_g_plot}) "
                f"porque no hubo datos válidos de SA para graficar.")

    # --- Paso 4: Mostrar Resultados Finales ---
    print("\n\n========================= RESULTADOS FINALES DEL PROCESO SA Estocástico =========================")
    print("Resumen de los mejores costos encontrados por cada configuración de SA y Solución Inicial Greedy:")

    mejor_costo_global_de_todo = float('inf')
    mejor_solucion_global_obj = None
    mejor_ejecucion_info = {}

    for idx_res_inicial, resultado_de_inicial in enumerate(todos_los_resultados_sa):
        semilla_g_res = resultado_de_inicial["greedy_seed"]
        costo_g_res = resultado_de_inicial["costo_inicial_greedy"]
        costo_g_res_str = f"{costo_g_res:.2f}" if costo_g_res != float('inf') else 'No generada/factible'
        print(f"\nResultados para la Solución Inicial Greedy #{idx_res_inicial+1} (Semilla: {semilla_g_res}):")
        print(f"  Costo de la Solución Inicial Greedy: {costo_g_res_str}")


        if not resultado_de_inicial["resultados_por_config"]:
            print("    No se ejecutó SA para esta solución inicial.")
            continue

        for resultado_de_config in resultado_de_inicial["resultados_por_config"]:
            config_label_res = resultado_de_config["config_label"]
            costo_final_sa_res = resultado_de_config["final_cost_sa"]
            sa_seed_res = resultado_de_config["sa_seed_usada"]
            print(
                f"    Configuración SA: {config_label_res} (Semilla SA: {sa_seed_res}) -> Costo Final SA: {costo_final_sa_res:.2f}")
            if costo_final_sa_res < mejor_costo_global_de_todo:
                mejor_costo_global_de_todo = costo_final_sa_res
                mejor_solucion_global_obj = resultado_de_config["best_solution_sa"]
                mejor_ejecucion_info = {
                    "greedy_seed": semilla_g_res,
                    "sa_config": config_label_res,
                    "sa_seed": sa_seed_res,
                    "initial_greedy_cost": costo_g_res
                }

    if mejor_solucion_global_obj:
        initial_greedy_cost_info_str = (f"{mejor_ejecucion_info.get('initial_greedy_cost'):.2f}"
                                        if mejor_ejecucion_info.get('initial_greedy_cost') != float('inf')
                                        else 'N/A')
        print(
            f"\nMejor solución global encontrada por SA (sobre todas las soluciones iniciales y configs): Costo = {mejor_costo_global_de_todo:.2f}")
        print(
            f"  Originada por Solución Inicial Greedy con Semilla: {mejor_ejecucion_info.get('greedy_seed', 'N/A')} (Costo Greedy: {initial_greedy_cost_info_str}), "
            f"Configuración SA: {mejor_ejecucion_info.get('sa_config', 'N/A')}, "
            f"Semilla SA: {mejor_ejecucion_info.get('sa_seed', 'N/A')}")

        print("\nHorario de la mejor solución global encontrada:")
        imprimir_matriz_aterrizaje_estocastico(
            mejor_solucion_global_obj, "Mejor Horario Global SA Estocástico")

        if not es_solucion_factible(mejor_solucion_global_obj, num_aviones, verbose=True):
            print("ALERTA CRÍTICA: La mejor solución global reportada NO es factible.")
        else:
            print("La mejor solución global reportada ES factible.")
    else:
        print("\nNo se encontró ninguna solución SA válida en ninguna configuración para ninguna solución inicial Greedy.")


if __name__ == "__main__":
    main_sa_runner()
