import sys
import math
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any

# Constante para el número de pistas
NUMERO_DE_PISTAS = 2

# --- Importaciones de scripts ---
try:
    # Funciones y clases del script de greedy determinista para 2 pistas
    from greedydeterminista_2pistas import (
        Avion, # Asume que Avion ya tiene pista_seleccionada
        leer_datos_aviones,
        greedy_determinista_2pistas
    )
    # Funciones de costo y factibilidad del script GRASP determinista para 2 pistas
    from grasp_hc_determinista_2pistas_alguna_mejora import ( # Nombre del archivo como fue creado previamente
        es_solucion_factible_2pistas,
        calcular_costo_total_solucion
    )
except ImportError as e:
    print(f"Error importando módulos necesarios: {e}")
    print("Asegúrate de que 'greedydeterminista_2pistas.py' y ")
    print("'grasp_hc_determinista_2pistas_alguna_mejora.py'")
    print("estén en la misma carpeta o en el PYTHONPATH y contengan las definiciones correctas para 2 pistas.")
    sys.exit(1)

# --- Función de Impresión Local ---
def imprimir_matriz_aterrizaje_local_sa(lista_aviones_procesada: List[Avion], cabecera: str) -> None:
    """
    Imprime una matriz con el horario de aterrizaje, incluyendo pista y cabecera.
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
# Estos son los mismos operadores que se usan en sa_2pistas_estocastico8.py

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


# --- Algoritmo de Simulated Annealing ---
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
    # Usar las claves como están definidas en las nuevas configuraciones
    prob_op1 = sa_config.get("prob_single_move", sa_config.get("prob_op1_single_attr", 0.5)) 
    prob_op2 = sa_config.get("prob_swap_times", sa_config.get("prob_op2_swap_attrs", 0.3))

    print_interval = sa_config.get("print_interval", 5000) 
    
    verbose_run = sa_config.get("verbose_sa_run", False)
    verbose_progress = sa_config.get("verbose_sa_progress", False)
    verbose_acceptance = sa_config.get("verbose_sa_acceptance", False)
    verbose_improvement = sa_config.get("verbose_sa_improvement", False)
    verbose_operator = sa_config.get("verbose_sa_operator", False)

    current_solution = copy.deepcopy(initial_solution) 
    
    if not es_solucion_factible_2pistas(current_solution, num_total_aviones, verbose=True):
        print(f"  SA Run (Config: {config_label}, Seed {sa_seed}): ADVERTENCIA - La solución inicial NO es factible. Costo puede ser irreal.")
    
    current_cost = calcular_costo_total_solucion(current_solution) 
    best_solution_so_far = copy.deepcopy(current_solution) 
    best_cost_so_far = current_cost 

    temp = initial_temp 
    
    cost_history_current: List[Tuple[int, float]] = [(0, current_cost)] 
    cost_history_best: List[Tuple[int, float]] = [(0, best_cost_so_far)] 
    
    global_iteration_counter = 0 

    if verbose_run:
        print(f"  SA Run (Config: {config_label}, Seed {sa_seed}): Iniciando SA 2 Pistas. Temp Inicial: {initial_temp:.2f}, "
              f"Costo Inicial: {current_cost:.2f}")

    while temp > final_temp:
        for _ in range(iter_per_temp):
            global_iteration_counter += 1

            if verbose_progress and global_iteration_counter % print_interval == 0:
                 print(f"    SA Run (Config: {config_label}, Seed {sa_seed}) Iter: {global_iteration_counter}, Temp: {temp:.2f}, "
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
            
            if verbose_operator and global_iteration_counter % print_interval == 0:
                print(f"      SA Iter {global_iteration_counter}: Intentando Operador -> {operator_used}")

            if neighbor_solution_propuesta is None:
                cost_history_current.append((global_iteration_counter, current_cost))
                cost_history_best.append((global_iteration_counter, best_cost_so_far))
                continue
            
            changed = False
            if len(current_solution) == len(neighbor_solution_propuesta):
                for i in range(len(current_solution)):
                    if current_solution[i].t_seleccionado != neighbor_solution_propuesta[i].t_seleccionado or \
                       current_solution[i].pista_seleccionada != neighbor_solution_propuesta[i].pista_seleccionada:
                        changed = True
                        break
            else: 
                changed = True

            if not changed:
                cost_history_current.append((global_iteration_counter, current_cost))
                cost_history_best.append((global_iteration_counter, best_cost_so_far))
                continue

            if es_solucion_factible_2pistas(neighbor_solution_propuesta, num_total_aviones, verbose=False):
                neighbor_cost = calcular_costo_total_solucion(neighbor_solution_propuesta)
                delta_cost = neighbor_cost - current_cost

                if delta_cost < 0: 
                    current_solution = copy.deepcopy(neighbor_solution_propuesta)
                    current_cost = neighbor_cost
                    if current_cost < best_cost_so_far: 
                        if verbose_improvement:
                             print(f"    SA Iter {global_iteration_counter} (T={temp:.2f}, Config: {config_label}): "
                                   f"Nueva mejor solución SA. Costo: {current_cost:.2f} (anterior mejor: {best_cost_so_far:.2f}) Op: {operator_used}")
                        best_solution_so_far = copy.deepcopy(current_solution)
                        best_cost_so_far = current_cost
                else: 
                    acceptance_probability = math.exp(-delta_cost / temp) if temp > 1e-9 else 0.0 
                    if random.random() < acceptance_probability: 
                        if verbose_acceptance:
                            print(f"    SA Iter {global_iteration_counter} (T={temp:.2f}, Config: {config_label}): "
                                  f"Aceptado peor sol. Costo Actual: {current_cost:.2f} -> Nuevo Costo: {neighbor_cost:.2f} "
                                  f"(Prob: {acceptance_probability:.3f}, Delta: {delta_cost:.2f}) Op: {operator_used}")
                        current_solution = copy.deepcopy(neighbor_solution_propuesta)
                        current_cost = neighbor_cost
            
            cost_history_current.append((global_iteration_counter, current_cost))
            cost_history_best.append((global_iteration_counter, best_cost_so_far))
        
        temp *= cooling_rate

    if verbose_run:
        print(f"  SA Run (Config: {config_label}, Seed {sa_seed}): SA 2 Pistas Finalizado. Mejor costo: {best_cost_so_far:.2f} "
              f"en {global_iteration_counter} iteraciones.")

    return best_solution_so_far, best_cost_so_far, cost_history_current, cost_history_best

# --- Función Principal del Runner SA ---
def main_sa_runner():
    """
    Función principal para SA con Greedy Determinista para 2 Pistas.
    """
    ruta_archivo_datos = "case1.txt" 
    SEMILLA_GLOBAL_REPLICACION = 42 

    # Configuraciones de SA tomadas de sa_estocastico5.py / sa_2pistas_estocastico8.py
    sa_configurations = [
        {
            "label": "Config1_Mod_ExploitFocus_From_C2",
            "initial_temp": 20000.0,
            "final_temp": 1.0,
            "cooling_rate": 0.97,
            "iter_per_temp": 150,
            "prob_op1_single_attr": 0.7, # Clave para el primer operador
            "prob_op2_swap_attrs": 0.1,  # Clave para el segundo operador
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
            "iter_per_temp": 50, "prob_op1_single_attr": 0.4, "prob_op2_swap_attrs": 0.3, 
            "print_interval": 5000,
            "verbose_sa_run": True, "verbose_sa_progress": False, "verbose_sa_improvement": True
        },
        {
            "label": "Config3_VeryFast_VLowIter_LowT", 
            "initial_temp": 5000.0, "final_temp": 1.0, "cooling_rate": 0.95,
            "iter_per_temp": 30, "prob_op1_single_attr": 0.5, "prob_op2_swap_attrs": 0.25, 
            "print_interval": 3000,
            "verbose_sa_run": True, "verbose_sa_progress": False, "verbose_sa_improvement": True
        },
        {
            "label": "Config4_HighIter_BalancedOps", 
            "initial_temp": 50000.0, "final_temp": 0.1, "cooling_rate": 0.99,
            "iter_per_temp": 300, "prob_op1_single_attr": 0.33, "prob_op2_swap_attrs": 0.33, 
            "print_interval": 15000,
            "verbose_sa_run": True, "verbose_sa_progress": True, "verbose_sa_improvement": True
        },
        {
            "label": "Config5_Fast_RapidCool_MedIter", 
            "initial_temp": 30000.0, "final_temp": 0.5, "cooling_rate": 0.96,
            "iter_per_temp": 70, "prob_op1_single_attr": 0.6, "prob_op2_swap_attrs": 0.2, 
            "print_interval": 7000,
            "verbose_sa_run": True, "verbose_sa_progress": False, "verbose_sa_improvement": True,
            "verbose_sa_acceptance": False, "verbose_sa_operator": False
        },
    ]
    
    print(">>> Iniciando Proceso de Simulated Annealing (con Greedy Determinista para 2 Pistas) <<<")
    print(f"Archivo de datos: {ruta_archivo_datos}")
    print(f"USANDO SEMILLA GLOBAL PARA REPLICACIÓN EN TODAS LAS CONFIGURACIONES DE SA: {SEMILLA_GLOBAL_REPLICACION}")

    print("\n--- Paso 1: Generando Solución Inicial con Greedy Determinista para 2 Pistas ---")
    try:
        lista_aviones_base = leer_datos_aviones(ruta_archivo_datos)
        if not lista_aviones_base:
            print(f"Error: No se pudieron leer aviones desde '{ruta_archivo_datos}'. Abortando.")
            return
        num_aviones = len(lista_aviones_base)
        print(f"Leídos {num_aviones} aviones desde '{ruta_archivo_datos}'.")

        solucion_greedy_inicial, factible_greedy, _ = greedy_determinista_2pistas(copy.deepcopy(lista_aviones_base))

        if not factible_greedy:
            print("ADVERTENCIA: La solución inicial del Greedy Determinista (2 Pistas) NO es factible.")
        if not es_solucion_factible_2pistas(solucion_greedy_inicial, num_aviones, verbose=True):
            print("ADVERTENCIA ADICIONAL: Solución Greedy (2 Pistas) no pasó la verificación de 'es_solucion_factible_2pistas'. SA podría no funcionar correctamente.")
        
        costo_greedy_inicial = calcular_costo_total_solucion(solucion_greedy_inicial)
        print(f"Solución Inicial del Greedy Determinista (2 Pistas) (Factible por greedy: {factible_greedy}):")
        print(f"Costo del Greedy (2 Pistas): {costo_greedy_inicial:.2f}")
        imprimir_matriz_aterrizaje_local_sa(solucion_greedy_inicial, "Matriz Greedy Inicial (2 Pistas)")

    except FileNotFoundError:
        print(f"Error CRÍTICO: No se pudo encontrar el archivo de datos '{ruta_archivo_datos}'.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Ocurrió un error preparando la solución inicial: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    if num_aviones == 0: 
        print("Error: No se pudo determinar el número de aviones. Abortando SA.")
        return

    print("\n--- Paso 2: Ejecutando Simulated Annealing para Múltiples Configuraciones (2 Pistas) ---")
    
    todas_las_historias_costos_actuales_sa: List[Tuple[str, List[Tuple[int, float]]]] = [] # Para el gráfico de costo actual
    resultados_finales_sa: List[Dict[str, Any]] = []
    
    for i, config in enumerate(sa_configurations): 
        config_label = config.get("label", f"Config{i+1}")
        print(f"\nProcesando con SA (2 Pistas) - {config_label}:")
        
        mejor_sol_sa, mejor_costo_sa, historia_actual_run, historia_mejor_costo_run = simulated_annealing_run(
            initial_solution=copy.deepcopy(solucion_greedy_inicial), 
            num_total_aviones=num_aviones,
            sa_seed=SEMILLA_GLOBAL_REPLICACION, 
            sa_config=config,
            config_label=config_label
        )
        
        todas_las_historias_costos_actuales_sa.append((config_label, historia_actual_run)) # Guardar historia del costo actual
        resultados_finales_sa.append({
            "config_label": config_label,
            "sa_seed_usada": SEMILLA_GLOBAL_REPLICACION, 
            "final_cost_sa": mejor_costo_sa,
            "best_solution_sa": copy.deepcopy(mejor_sol_sa), 
            "sa_params": config 
        })

    print("\n\n========================= RESULTADOS FINALES DEL PROCESO SA (Determinista 2 Pistas) =========================")
    print(f"Costo Inicial (Greedy Determinista 2 Pistas): {costo_greedy_inicial:.2f}")
    print(f"Semilla Global de Replicación Utilizada para TODAS las corridas de SA: {SEMILLA_GLOBAL_REPLICACION}")
    print("\nResumen de costos por configuración de SA:")
    
    mejor_costo_global_todas_sa = float('inf')
    mejor_solucion_global_obj_sa = None
    mejor_run_info_sa = {}

    for res in resultados_finales_sa:
        costo_sa_str = f"{res['final_cost_sa']:.2f}" if res['final_cost_sa'] != float('inf') else "Inf"
        print(f"  Config SA: {res['config_label']} (SA Seed Usada: {res['sa_seed_usada']}) -> Costo SA Final = {costo_sa_str}")
        if res['final_cost_sa'] < mejor_costo_global_todas_sa:
            mejor_costo_global_todas_sa = res['final_cost_sa']
            mejor_solucion_global_obj_sa = res['best_solution_sa']
            mejor_run_info_sa = res 

    if mejor_solucion_global_obj_sa:
        print(f"\nMejor solución global encontrada por SA (2 Pistas, todas configs, semilla {SEMILLA_GLOBAL_REPLICACION}): Costo = {mejor_costo_global_todas_sa:.2f}")
        print(f"  Originada por Config SA: {mejor_run_info_sa.get('config_label', 'N/A')}, SA Seed Usada: {mejor_run_info_sa.get('sa_seed_usada', 'N/A')}")
        
        print(f"\nHorario de la mejor solución global SA (Config: {mejor_run_info_sa.get('config_label', 'N/A')})")
        imprimir_matriz_aterrizaje_local_sa(mejor_solucion_global_obj_sa, f"Mejor Horario Global SA (2 Pistas, Config: {mejor_run_info_sa.get('config_label', 'N/A')})")
        
        if not es_solucion_factible_2pistas(mejor_solucion_global_obj_sa, num_aviones, verbose=True):
             print("ALERTA CRÍTICA: La mejor solución SA global (2 Pistas) reportada NO es factible.")
        else:
             print("La mejor solución SA global (2 Pistas) reportada ES factible.")
    else:
        print("\nNo se encontró ninguna solución SA válida en ninguna configuración.")

    # --- Gráfico: Evolución del Costo Actual Encontrado por Iteración (Consolidado) ---
    plt.figure(figsize=(15, 8))
    plot_successful_current_cost = False
    
    for config_label, historia_costo_actual in todas_las_historias_costos_actuales_sa:
        if historia_costo_actual and historia_costo_actual[0][1] != float('inf') and historia_costo_actual[0][1] < 1e11 : 
            iteraciones = [item[0] for item in historia_costo_actual]
            costos_actuales = [item[1] for item in historia_costo_actual]
            plt.plot(iteraciones, costos_actuales, label=f"Costo Actual SA - {config_label}")
            plot_successful_current_cost = True
        else:
            print(f"Nota: Historia del costo actual para SA (2 Pistas) con config '{config_label}' no se grafica.")

    if plot_successful_current_cost:
        plt.xlabel("Iteración Global de SA")
        plt.ylabel("Costo Actual de la Solución")
        titulo_grafico_actual = (f"Evolución del Costo Actual en SA (2 Pistas) para {ruta_archivo_datos}\n"
                                 f"(Desde Greedy Determinista 2 Pistas, Costo Inicial: {costo_greedy_inicial:.2f})\n"
                                 f"Semilla Global para todas las corridas SA: {SEMILLA_GLOBAL_REPLICACION}")
        plt.title(titulo_grafico_actual)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) 
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.80, 1]) 

        nombre_grafico_actual = f"sa_2pistas_determinista_costo_actual_evolucion_{ruta_archivo_datos.split('.')[0]}_sem{SEMILLA_GLOBAL_REPLICACION}.png"
        try:
            plt.savefig(nombre_grafico_actual)
            print(f"\nGráfico de costo actual guardado como: {nombre_grafico_actual}")
        except Exception as e_plot:
            print(f"Error al guardar el gráfico de costo actual: {e_plot}")
        plt.show() 
    else:
        print("\nNo se generó ningún gráfico de costo actual porque no hubo ejecuciones de SA válidas para graficar.")

if __name__ == "__main__":
    main_sa_runner()
