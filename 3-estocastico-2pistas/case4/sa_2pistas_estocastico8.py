import sys
import math
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any

# Constante para el número de pistas
NUMERO_DE_PISTAS = 2

# --- Importaciones de tus scripts ---
try:
    # Funciones y clases del script de greedy estocástico para 2 pistas
    from greedyestocastico_2pistas_mas_random import (
        Avion, # Asume que Avion ya tiene pista_seleccionada
        leer_datos_aviones,
        greedy_estocastico_2pistas_mas_random, # Constructor de soluciones iniciales
        calcular_costo_total_2pistas as calcular_costo_total_solucion_2pistas_estocastico
    )
    # Función de factibilidad del script GRASP determinista para 2 pistas (considerada robusta)
    from grasp_hc_determinista_2pistas_alguna_mejora import ( # Asegúrate que el nombre del archivo es el correcto
        es_solucion_factible_2pistas
    )
except ImportError as e:
    print(f"Error importando módulos necesarios: {e}")
    print("Asegúrate de que 'greedyestocastico_2pistas_mas_random.py' y ")
    print("'grasp_hc_determinista_2pistas_alguna_mejora.py' (o el nombre correcto del artefacto que contiene es_solucion_factible_2pistas)")
    print("estén en la misma carpeta o en el PYTHONPATH y contengan las definiciones correctas para 2 pistas.")
    sys.exit(1)

# --- Función de Impresión Local ---
def imprimir_matriz_aterrizaje_local_sa_2pistas(lista_aviones_procesada: List[Avion], cabecera: str) -> None:
    """
    Imprime una matriz con el horario de aterrizaje para 2 pistas, incluyendo cabecera.
    """
    print(f"\n--- {cabecera} ---")
    print("----------------------------------------------------")
    print(f"| {'Tiempo Aterrizaje':<18} | {'ID Avión':<10} | {'Pista':<7} |")
    print("|--------------------|------------|---------|")

    aviones_aterrizados = [
        avion for avion in lista_aviones_procesada
        if avion.t_seleccionado is not None and avion.pista_seleccionada is not None
    ]
    
    if not aviones_aterrizados:
        print("| Sin aterrizajes    | N/A        | N/A     |")
        print("----------------------------------------------------")
        return

    aviones_aterrizados.sort(key=lambda avion: (
        avion.t_seleccionado, 
        avion.pista_seleccionada, 
        avion.id_avion
    ))

    for avion in aviones_aterrizados:
        tiempo_str = f"{avion.t_seleccionado:<18}"
        avion_id_str = f"{avion.id_avion:<10}"
        pista_str = f"{avion.pista_seleccionada:<7}"
        print(f"| {tiempo_str} | {avion_id_str} | {pista_str} |")
    
    print("----------------------------------------------------")

# --- Generadores de Vecinos para Simulated Annealing (2 Pistas) ---
def generar_vecino_single_attribute_change_2pistas(solution: List[Avion], num_total_aviones: int, prob_change_time: float = 0.7) -> Optional[List[Avion]]:
    """
    Genera un vecino cambiando un atributo (tiempo o pista) de un avión seleccionado aleatoriamente.
    """
    if not solution or num_total_aviones == 0:
        return None
    vecino = copy.deepcopy(solution)
    idx_avion_a_modificar = random.randrange(len(vecino))
    avion_obj = vecino[idx_avion_a_modificar]

    if random.random() < prob_change_time: # Cambiar tiempo
        lim_inf = avion_obj.tiempo_aterrizaje_temprano
        lim_sup = avion_obj.tiempo_aterrizaje_tardio
        nuevo_tiempo: Optional[int] = avion_obj.t_seleccionado 
        if lim_inf <= lim_sup:
            try: nuevo_tiempo = random.randint(lim_inf, lim_sup)
            except ValueError: pass 
        avion_obj.t_seleccionado = nuevo_tiempo
    else: # Cambiar pista
        if avion_obj.pista_seleccionada is not None:
            avion_obj.pista_seleccionada = 1 - avion_obj.pista_seleccionada
        else: 
            avion_obj.pista_seleccionada = random.randint(0, NUMERO_DE_PISTAS - 1)
    return vecino

def generar_vecino_swap_all_attributes_2pistas(solution: List[Avion], num_total_aviones: int) -> Optional[List[Avion]]:
    """
    Genera un vecino intercambiando tiempos y pistas de dos aviones.
    """
    if not solution or num_total_aviones < 2: return None
    vecino = copy.deepcopy(solution)
    if len(vecino) < 2: return copy.deepcopy(solution)
    try:
        idx1, idx2 = random.sample(range(len(vecino)), 2)
    except ValueError: return copy.deepcopy(solution)

    vecino[idx1].t_seleccionado, vecino[idx2].t_seleccionado = vecino[idx2].t_seleccionado, vecino[idx1].t_seleccionado
    vecino[idx1].pista_seleccionada, vecino[idx2].pista_seleccionada = vecino[idx2].pista_seleccionada, vecino[idx1].pista_seleccionada
    return vecino

def generar_vecino_reinsert_plane_2pistas(solution: List[Avion], num_total_aviones: int) -> Optional[List[Avion]]:
    """
    Genera un vecino reasignando tiempo y pista a un avión aleatorio.
    """
    if not solution or num_total_aviones == 0: return None
    vecino = copy.deepcopy(solution)
    idx_avion_a_reinsertar = random.randrange(len(vecino))
    avion_k = vecino[idx_avion_a_reinsertar]

    lim_inf_t = avion_k.tiempo_aterrizaje_temprano
    lim_sup_t = avion_k.tiempo_aterrizaje_tardio
    if lim_inf_t <= lim_sup_t:
        try: avion_k.t_seleccionado = random.randint(lim_inf_t, lim_sup_t)
        except ValueError: pass
    
    avion_k.pista_seleccionada = random.randint(0, NUMERO_DE_PISTAS - 1)
    return vecino

# --- Algoritmo de Simulated Annealing (adaptado para 2 pistas) ---
def simulated_annealing_run(
    initial_solution: List[Avion],
    num_total_aviones: int,
    sa_seed: int, 
    sa_config: Dict[str, Any],
    config_label: str
) -> Tuple[List[Avion], float, List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Ejecuta una instancia de Simulated Annealing para 2 pistas.
    """
    random.seed(sa_seed) 

    initial_temp = sa_config["initial_temp"]
    final_temp = sa_config["final_temp"]
    cooling_rate = sa_config["cooling_rate"]
    iter_per_temp = sa_config["iter_per_temp"]
    # Usar las claves actualizadas para las probabilidades de los operadores
    prob_op1 = sa_config.get("prob_op1_single_attr", 0.5) 
    prob_op2 = sa_config.get("prob_op2_swap_attrs", 0.3)

    print_interval = sa_config.get("print_interval", 5000) 
    verbose_run = sa_config.get("verbose_sa_run", False)
    verbose_progress = sa_config.get("verbose_sa_progress", False)
    verbose_acceptance = sa_config.get("verbose_sa_acceptance", False)
    verbose_improvement = sa_config.get("verbose_sa_improvement", False)
    verbose_operator = sa_config.get("verbose_sa_operator", False)

    current_solution = copy.deepcopy(initial_solution) 
    
    if not es_solucion_factible_2pistas(current_solution, num_total_aviones, verbose=True):
        print(f"  SA Run (2 Pistas, Config: {config_label}, Seed SA: {sa_seed}): ADVERTENCIA - La solución inicial NO es factible. Costo puede ser irreal.")
    
    current_cost = calcular_costo_total_solucion_2pistas_estocastico(current_solution) 
    best_solution_so_far = copy.deepcopy(current_solution) 
    best_cost_so_far = current_cost 

    temp = initial_temp 
    cost_history_current: List[Tuple[int, float]] = [(0, current_cost)] 
    cost_history_best: List[Tuple[int, float]] = [(0, best_cost_so_far)] 
    global_iteration_counter = 0 

    if verbose_run:
        print(f"  SA Run (2 Pistas, Config: {config_label}, Seed SA: {sa_seed}): Iniciando. Temp Inicial: {initial_temp:.2f}, "
              f"Costo Inicial: {current_cost:.2f}")

    while temp > final_temp:
        for _ in range(iter_per_temp):
            global_iteration_counter += 1

            if verbose_progress and global_iteration_counter % print_interval == 0:
                 print(f"    SA Run (2 Pistas, Config: {config_label}, Seed SA: {sa_seed}) Iter: {global_iteration_counter}, Temp: {temp:.2f}, "
                       f"Costo Actual: {current_cost:.2f}, Mejor Costo: {best_cost_so_far:.2f}")

            rand_val = random.random()
            neighbor_solution_propuesta: Optional[List[Avion]] = None
            operator_used = ""

            if rand_val < prob_op1:
                neighbor_solution_propuesta = generar_vecino_single_attribute_change_2pistas(current_solution, num_total_aviones)
                operator_used = "SINGLE_ATTR_CHANGE"
            elif rand_val < prob_op1 + prob_op2:
                neighbor_solution_propuesta = generar_vecino_swap_all_attributes_2pistas(current_solution, num_total_aviones)
                operator_used = "SWAP_ALL_ATTRS"
            else: 
                neighbor_solution_propuesta = generar_vecino_reinsert_plane_2pistas(current_solution, num_total_aviones)
                operator_used = "REINSERT_PLANE"
            
            if verbose_operator and global_iteration_counter % print_interval == 0 :
                print(f"      SA Iter {global_iteration_counter}: Intentando Operador -> {operator_used}")

            if neighbor_solution_propuesta is None:
                cost_history_current.append((global_iteration_counter, current_cost))
                cost_history_best.append((global_iteration_counter, best_cost_so_far))
                continue
            
            changed = False 
            if len(current_solution) == len(neighbor_solution_propuesta):
                for i_comp in range(len(current_solution)):
                    if current_solution[i_comp].t_seleccionado != neighbor_solution_propuesta[i_comp].t_seleccionado or \
                       current_solution[i_comp].pista_seleccionada != neighbor_solution_propuesta[i_comp].pista_seleccionada:
                        changed = True; break
            else: changed = True

            if not changed:
                cost_history_current.append((global_iteration_counter, current_cost))
                cost_history_best.append((global_iteration_counter, best_cost_so_far))
                continue

            if es_solucion_factible_2pistas(neighbor_solution_propuesta, num_total_aviones, verbose=False):
                neighbor_cost = calcular_costo_total_solucion_2pistas_estocastico(neighbor_solution_propuesta)
                delta_cost = neighbor_cost - current_cost

                if delta_cost < 0: 
                    current_solution = copy.deepcopy(neighbor_solution_propuesta)
                    current_cost = neighbor_cost
                    if current_cost < best_cost_so_far: 
                        if verbose_improvement:
                             print(f"    SA Iter {global_iteration_counter} (T={temp:.2f}, Cfg: {config_label}, SeedSA: {sa_seed}): "
                                   f"Nueva mejor sol. Costo: {current_cost:.2f} (ant: {best_cost_so_far:.2f}) Op: {operator_used}")
                        best_solution_so_far = copy.deepcopy(current_solution)
                        best_cost_so_far = current_cost
                else: 
                    acceptance_probability = math.exp(-delta_cost / temp) if temp > 1e-9 else 0.0 
                    if random.random() < acceptance_probability: 
                        if verbose_acceptance:
                            print(f"    SA Iter {global_iteration_counter} (T={temp:.2f}, Cfg: {config_label}, SeedSA: {sa_seed}): "
                                  f"Aceptado peor sol. Costo: {current_cost:.2f} -> {neighbor_cost:.2f} "
                                  f"(P: {acceptance_probability:.3f}, D: {delta_cost:.2f}) Op: {operator_used}")
                        current_solution = copy.deepcopy(neighbor_solution_propuesta)
                        current_cost = neighbor_cost
            
            cost_history_current.append((global_iteration_counter, current_cost))
            cost_history_best.append((global_iteration_counter, best_cost_so_far))
        
        temp *= cooling_rate

    if verbose_run:
        print(f"  SA Run (2 Pistas, Config: {config_label}, Seed SA: {sa_seed}): Finalizado. Mejor costo: {best_cost_so_far:.2f} "
              f"en {global_iteration_counter} iteraciones.")

    return best_solution_so_far, best_cost_so_far, cost_history_current, cost_history_best

# --- Función para generar soluciones iniciales (VERSIÓN COMPLETA) ---
def generar_soluciones_iniciales_greedy_2pistas(
    lista_aviones_base_original: List[Avion],
    semillas_greedy: List[int],
    verbose_greedy_constructor: bool = False
) -> Tuple[List[Optional[List[Avion]]], List[float]]:
    """
    Genera múltiples soluciones iniciales usando el greedy estocástico para 2 pistas.
    """
    lista_soluciones_generadas: List[Optional[List[Avion]]] = []
    costos_soluciones_generadas: List[float] = []
    num_aviones = len(lista_aviones_base_original)

    for i, semilla_g in enumerate(semillas_greedy):
        print(f"\nGenerando Solución Inicial Greedy (2 Pistas) #{i + 1} con semilla: {semilla_g}")
        random.seed(semilla_g) 

        aviones_para_greedy_actual = copy.deepcopy(lista_aviones_base_original)
        
        solucion_g_obj, factible_g, _ = greedy_estocastico_2pistas_mas_random(
            aviones_para_greedy_actual,
            verbose_internal=verbose_greedy_constructor
        )

        if factible_g and solucion_g_obj:
            if not es_solucion_factible_2pistas(solucion_g_obj, num_aviones, verbose=True):
                print(f"  ADVERTENCIA: Solución Greedy #{i+1} (semilla {semilla_g}) marcada factible por greedy, "
                      "pero NO pasó la verificación externa de 'es_solucion_factible_2pistas'. Se tratará como infactible.")
                lista_soluciones_generadas.append(None)
                costos_soluciones_generadas.append(float('inf'))
                continue 

            costo_g = calcular_costo_total_solucion_2pistas_estocastico(solucion_g_obj)
            print(f"  Solución Greedy (2 Pistas) #{i+1} (semilla {semilla_g}) encontrada. Costo: {costo_g:.2f}")
            imprimir_matriz_aterrizaje_local_sa_2pistas(solucion_g_obj, f"Sol. Inicial Greedy 2 Pistas #{i + 1} (Semilla {semilla_g})")
            lista_soluciones_generadas.append(copy.deepcopy(solucion_g_obj))
            costos_soluciones_generadas.append(costo_g)
        else:
            print(f"  ADVERTENCIA: No se encontró solución Greedy (2 Pistas) factible para semilla {semilla_g}.")
            lista_soluciones_generadas.append(None)
            costos_soluciones_generadas.append(float('inf'))
            
    return lista_soluciones_generadas, costos_soluciones_generadas

# --- Función Principal del Runner SA (Estocástico para 2 Pistas - VERSIÓN COMPLETA) ---
def main_sa_runner():
    ruta_archivo_datos = "case4.txt"
    semillas_greedy_originales = [42, 123, 7, 99, 500, 777, 2024, 1, 100, 314] 
    num_soluciones_iniciales = len(semillas_greedy_originales)

    # Configuraciones de SA actualizadas según la solicitud del usuario
    sa_configurations = [
        {
            "label": "Config1_Mod_ExploitFocus_From_C2",
            "initial_temp": 20000.0,
            "final_temp": 1.0,
            "cooling_rate": 0.97,
            "iter_per_temp": 150,
            "prob_op1_single_attr": 0.7, # Clave actualizada
            "prob_op2_swap_attrs": 0.1,  # Clave actualizada
            "print_interval": 5000,
            "verbose_sa_run": True,
            "verbose_sa_progress": False,
            "verbose_sa_improvement": True,
            "verbose_sa_acceptance": False,
            "verbose_sa_operator": False
        },
        {
            "label": "Config2_Fast_LowIter_MedT", 
            "initial_temp": 20000.0, "final_temp": 1.0, "cooling_rate": 0.97,
            "iter_per_temp": 50, "prob_op1_single_attr": 0.4, "prob_op2_swap_attrs": 0.3, # Claves actualizadas
            "print_interval": 5000,
            "verbose_sa_run": True, "verbose_sa_progress": False, "verbose_sa_improvement": True
        },
        {
            "label": "Config3_VeryFast_VLowIter_LowT", 
            "initial_temp": 5000.0, "final_temp": 1.0, "cooling_rate": 0.95,
            "iter_per_temp": 30, "prob_op1_single_attr": 0.5, "prob_op2_swap_attrs": 0.25, # Claves actualizadas
            "print_interval": 3000,
            "verbose_sa_run": True, "verbose_sa_progress": False, "verbose_sa_improvement": True
        },
        {
            "label": "Config4_HighIter_BalancedOps", 
            "initial_temp": 50000.0, "final_temp": 0.1, "cooling_rate": 0.99,
            "iter_per_temp": 300, "prob_op1_single_attr": 0.33, "prob_op2_swap_attrs": 0.33, # Claves actualizadas
            "print_interval": 15000,
            "verbose_sa_run": True, "verbose_sa_progress": True, "verbose_sa_improvement": True
        },
        {
            "label": "Config5_Fast_RapidCool_MedIter", 
            "initial_temp": 30000.0, "final_temp": 0.5, "cooling_rate": 0.96,
            "iter_per_temp": 70, "prob_op1_single_attr": 0.6, "prob_op2_swap_attrs": 0.2, # Claves actualizadas
            "print_interval": 7000,
            "verbose_sa_run": True, "verbose_sa_progress": False, "verbose_sa_improvement": True,
            "verbose_sa_acceptance": False, "verbose_sa_operator": False
        },
    ]
    num_sa_configurations = len(sa_configurations)

    print(f">>> Iniciando SA Estocástico (2 Pistas - {num_soluciones_iniciales} Sol. Iniciales) x {num_sa_configurations} Configs SA <<<")
    print(f"Archivo de datos: {ruta_archivo_datos}")

    try:
        lista_aviones_base = leer_datos_aviones(ruta_archivo_datos)
        num_aviones = len(lista_aviones_base)
        print(f"Leídos {num_aviones} aviones.\n")
    except Exception as e:
        print(f"Error CRÍTICO leyendo datos: {e}", file=sys.stderr); return

    print("\n--- Paso 1: Generando Soluciones Iniciales con Greedy Estocástico (2 Pistas) ---")
    lista_soluciones_iniciales_greedy, costos_iniciales_greedy = generar_soluciones_iniciales_greedy_2pistas(
        lista_aviones_base, semillas_greedy_originales, verbose_greedy_constructor=False
    )

    print("\n--- Paso 2: Ejecutando Simulated Annealing para cada Solución Inicial y Configuración (2 Pistas) ---")
    todos_los_resultados_sa = [] 

    for idx_sol_inicial, solucion_inicial_g in enumerate(lista_soluciones_iniciales_greedy):
        semilla_greedy_usada = semillas_greedy_originales[idx_sol_inicial]
        costo_inicial_g = costos_iniciales_greedy[idx_sol_inicial]

        if solucion_inicial_g is None: 
            print(f"\nSaltando SA para la sol. inicial Greedy #{idx_sol_inicial + 1} (Semilla: {semilla_greedy_usada}) porque no fue generada/factible.")
            todos_los_resultados_sa.append({
                "greedy_seed": semilla_greedy_usada, "resultados_por_config": [],
                "solucion_inicial_greedy": None, "costo_inicial_greedy": float('inf')
            })
            continue

        print(f"\nEjecutando SA para sol. inicial Greedy #{idx_sol_inicial + 1} (Semilla Greedy: {semilla_greedy_usada}, Costo Inicial: {costo_inicial_g:.2f})")
        resultados_sa_para_esta_inicial = []

        for idx_config, config_sa in enumerate(sa_configurations):
            config_label = config_sa.get("label", f"Config{idx_config + 1}")
            print(f"\n  Configuración SA: {config_label}")
            semilla_sa_para_esta_config = (idx_sol_inicial + 1) * 1000 + (idx_config + 1) * 100 

            mejor_sol_sa, mejor_costo_sa, historia_actual, historia_mejor = simulated_annealing_run(
                initial_solution=copy.deepcopy(solucion_inicial_g), num_total_aviones=num_aviones,
                sa_seed=semilla_sa_para_esta_config, sa_config=config_sa, config_label=config_label
            )
            resultados_sa_para_esta_inicial.append({
                "config_label": config_label, "sa_seed_usada": semilla_sa_para_esta_config,
                "final_cost_sa": mejor_costo_sa, "best_solution_sa": copy.deepcopy(mejor_sol_sa),
                "historia_costo_actual": historia_actual, "historia_mejor_costo": historia_mejor,
                "sa_params": config_sa
            })
        todos_los_resultados_sa.append({
            "greedy_seed": semilla_greedy_usada, "resultados_por_config": resultados_sa_para_esta_inicial,
            "solucion_inicial_greedy": copy.deepcopy(solucion_inicial_g), "costo_inicial_greedy": costo_inicial_g
        })

    print(f"\n--- Paso 3: Generando {num_soluciones_iniciales} Gráfico(s) de Evolución del Costo (2 Pistas) ---")
    for idx_plot, data_plot in enumerate(todos_los_resultados_sa):
        semilla_g_plot = data_plot["greedy_seed"]
        costo_g_plot = data_plot["costo_inicial_greedy"]
        if not data_plot["resultados_por_config"]: 
            print(f"  No se generará gráfico para Sol. Inicial Greedy #{idx_plot + 1} (Semilla: {semilla_g_plot}) porque no tuvo ejecuciones de SA.")
            continue

        plt.figure(figsize=(15, 8))
        plot_lines_count = 0
        for res_cfg_plot in data_plot["resultados_por_config"]:
            hist_actual_plot = res_cfg_plot["historia_costo_actual"]
            if hist_actual_plot and hist_actual_plot[0][1] != float('inf') and hist_actual_plot[0][1] < 1e12: 
                iters_plot = [item[0] for item in hist_actual_plot]
                costs_plot = [item[1] for item in hist_actual_plot]
                plt.plot(iters_plot, costs_plot, label=f"SA {res_cfg_plot['config_label']} (Costo Final: {res_cfg_plot['final_cost_sa']:.2f}, Seed SA: {res_cfg_plot['sa_seed_usada']})")
                plot_lines_count +=1
        
        if plot_lines_count > 0:
            costo_g_plot_str = f"{costo_g_plot:.2f}" if costo_g_plot != float('inf') else 'N/A'
            titulo = (f"Evolución Costo SA (2 Pistas) para Sol. Inicial Greedy #{idx_plot + 1} (Seed Greedy: {semilla_g_plot})\n"
                      f"Costo Inicial Greedy: {costo_g_plot_str}")
            plt.title(titulo); plt.xlabel("Iteración SA"); plt.ylabel("Costo Actual Solución")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.grid(True); plt.tight_layout(rect=[0, 0, 0.75, 1]) 
            nombre_graf = f"sa_2pistas_estocastico_evolucion_greedy_seed_{semilla_g_plot}.png"
            try: plt.savefig(nombre_graf); print(f"  Gráfico guardado: {nombre_graf}")
            except Exception as e_save: print(f"  Error guardando gráfico '{nombre_graf}': {e_save}")
            plt.close() 
        else:
            print(f"  No se generó gráfico para Sol. Inicial Greedy #{idx_plot + 1} (Seed: {semilla_g_plot}) - sin datos válidos de SA.")

    print("\n\n========================= RESULTADOS FINALES SA Estocástico (2 Pistas) =========================")
    mejor_costo_global = float('inf')
    mejor_sol_global_obj = None
    mejor_info_ejecucion = {}

    for idx_res_ini, res_ini in enumerate(todos_los_resultados_sa):
        semilla_g_res = res_ini["greedy_seed"]
        costo_g_res = res_ini["costo_inicial_greedy"]
        costo_g_str = f"{costo_g_res:.2f}" if costo_g_res != float('inf') else 'No generada/factible'
        print(f"\nResultados para Sol. Inicial Greedy #{idx_res_ini+1} (Semilla: {semilla_g_res}, Costo: {costo_g_str}):")
        if not res_ini["resultados_por_config"]: print("    No se ejecutó SA."); continue

        for res_cfg in res_ini["resultados_por_config"]:
            costo_final_sa = res_cfg["final_cost_sa"]
            print(f"    Config SA: {res_cfg['config_label']} (Seed SA: {res_cfg['sa_seed_usada']}) -> Costo Final SA: {costo_final_sa:.2f}")
            if costo_final_sa < mejor_costo_global:
                mejor_costo_global = costo_final_sa
                mejor_sol_global_obj = res_cfg["best_solution_sa"]
                mejor_info_ejecucion = {"greedy_seed": semilla_g_res, "sa_config": res_cfg['config_label'], 
                                        "sa_seed": res_cfg['sa_seed_usada'], "initial_greedy_cost": costo_g_res}

    if mejor_sol_global_obj:
        costo_greedy_mejor_str = f"{mejor_info_ejecucion.get('initial_greedy_cost'):.2f}" if mejor_info_ejecucion.get('initial_greedy_cost') != float('inf') else 'N/A'
        print(f"\nMejor solución global SA (2 Pistas): Costo = {mejor_costo_global:.2f}")
        print(f"  Originada por Sol. Inicial Greedy (Seed: {mejor_info_ejecucion.get('greedy_seed', 'N/A')}, Costo: {costo_greedy_mejor_str}), "
              f"Config SA: {mejor_info_ejecucion.get('sa_config', 'N/A')}, Seed SA: {mejor_info_ejecucion.get('sa_seed', 'N/A')}")
        imprimir_matriz_aterrizaje_local_sa_2pistas(mejor_sol_global_obj, "Mejor Horario Global SA Estocástico (2 Pistas)")
        if not es_solucion_factible_2pistas(mejor_sol_global_obj, num_aviones, verbose=True):
            print("ALERTA CRÍTICA: La mejor solución SA global (2 Pistas) reportada NO es factible.")
        else:
            print("La mejor solución SA global (2 Pistas) reportada ES factible.")
    else:
        print("\nNo se encontró ninguna solución SA válida.")

if __name__ == "__main__":
    main_sa_runner()
