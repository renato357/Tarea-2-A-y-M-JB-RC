import sys
import math
import random
import copy
from typing import List, Optional, Tuple, Set
import matplotlib.pyplot as plt # <--- AÑADIDO para graficar

# Asegúrate de que 'greedy_estocastico_10_semillas.py' esté en la misma carpeta o en el PYTHONPATH
try:
    from greedy_estocastico_10_semillas import (
        Avion,
        leer_datos_aviones,
        greedy_stochastico_Ek_ponderado_costo,
        calcular_costo_total as calcular_costo_total_solucion_gs, # Usaremos esta para consistencia
        imprimir_matriz_aterrizaje as imprimir_horario_aterrizaje_gs,
        calcular_costo_en_tiempo_especifico # Necesaria para el costo individual
    )
except ImportError as e:
    print(f"Error importando desde 'greedy_estocastico_10_semillas.py': {e}")
    print("Asegúrate de que el archivo 'greedy_estocastico_10_semillas.py' esté en la misma carpeta.")
    sys.exit(1)

# --- Funciones de Cálculo de Costo y Factibilidad (Adaptadas y/o Reutilizadas) ---

def calcular_costo_individual_avion(avion: Avion, tiempo_aterrizaje_propuesto: int) -> float:
    """Calcula el costo de penalización si el avión aterriza en un tiempo específico."""
    return calcular_costo_en_tiempo_especifico(avion, tiempo_aterrizaje_propuesto)

def calcular_costo_total_solucion(solucion: List[Avion]) -> float:
    """Calcula el costo total de la programación de aterrizajes para una solución dada."""
    return calcular_costo_total_solucion_gs(solucion)

def es_solucion_factible(solucion: List[Avion], numero_total_aviones_problema: int, verbose: bool = False) -> bool:
    """
    Verifica si una solución (lista de aviones con t_seleccionado) es factible.
    Adaptado de hc-alguna-mejora_estocastico.py.
    """
    if not solucion:
        if verbose: print("  [Factibilidad] Falla: La solución está vacía.")
        return False

    aviones_programados_count = 0
    for avion_obj in solucion:
        if avion_obj.t_seleccionado is not None:
            aviones_programados_count += 1
        else:
            if verbose: print(f"  [Factibilidad] Falla: Avión ID {avion_obj.id_avion} no tiene t_seleccionado.")
            return False 

    if aviones_programados_count != numero_total_aviones_problema:
        if verbose: print(f"  [Factibilidad] Falla: Se programaron {aviones_programados_count} aviones, se esperaban {numero_total_aviones_problema}.")
        return False
    
    try:
        # Asegurarse de que todos los t_seleccionado son números antes de ordenar
        for a_sort_check in solucion:
            if not isinstance(a_sort_check.t_seleccionado, (int, float)): # type: ignore
                if verbose: print(f"  [Factibilidad] Falla: Avión ID {a_sort_check.id_avion} tiene t_seleccionado no numérico: {a_sort_check.t_seleccionado}.")
                return False
        secuencia_ordenada = sorted(solucion, key=lambda a: a.t_seleccionado) # type: ignore
    except TypeError: 
        if verbose: print("  [Factibilidad] Error al ordenar: t_seleccionado es None inesperadamente.")
        return False

    for avion in secuencia_ordenada: 
        if not (avion.tiempo_aterrizaje_temprano <= avion.t_seleccionado <= avion.tiempo_aterrizaje_tardio): # type: ignore
            if verbose: print(f"  [Factibilidad] Falla: Avión ID {avion.id_avion} t={avion.t_seleccionado} fuera de ventana [{avion.tiempo_aterrizaje_temprano}, {avion.tiempo_aterrizaje_tardio}].")
            return False

    for i in range(len(secuencia_ordenada) - 1):
        avion_i = secuencia_ordenada[i]
        avion_j = secuencia_ordenada[i+1]
        separacion_requerida = avion_i.get_separation_to(avion_j)
        
        if avion_j.t_seleccionado < avion_i.t_seleccionado + separacion_requerida: # type: ignore
            if verbose: print(f"  [Factibilidad] Falla: Separación entre avión ID {avion_i.id_avion} (t={avion_i.t_seleccionado}) y "
                               f"avión ID {avion_j.id_avion} (t={avion_j.t_seleccionado}). "
                               f"Real: {avion_j.t_seleccionado - avion_i.t_seleccionado}, Requerida: {separacion_requerida}.") # type: ignore
            return False
    if verbose: print("  [Factibilidad] Solución es factible.")
    return True

# --- Hill Climbing ---
def ejecutar_hill_climbing(
    solucion_inicial_hc: List[Avion],
    numero_total_aviones: int,
    paso_tiempo_delta: int,
    max_iter_local_hc: int,
    verbose_hc: bool = False
    ) -> Tuple[List[Avion], float, List[Tuple[int, float]]]: # MODIFICADO: Devuelve historial de costos HC
    """
    Aplica Hill Climbing (Alguna-Mejora Determinista) a una solución.
    Devuelve la solución mejorada, su costo y el historial de costos de HC.
    """
    solucion_hc_actual = copy.deepcopy(solucion_inicial_hc)
    costo_hc_actual = calcular_costo_total_solucion(solucion_hc_actual)
    
    hc_cost_history: List[Tuple[int, float]] = []
    hc_cost_history.append((0, costo_hc_actual)) # Costo inicial antes de la primera iteración de HC

    if verbose_hc: print(f"    HC Iniciando con costo local: {costo_hc_actual:.2f}")

    for iter_local in range(max_iter_local_hc):
        mejora_encontrada_iter_local = False
        indices_aviones = list(range(len(solucion_hc_actual)))
        random.shuffle(indices_aviones) 

        for i in indices_aviones:
            avion_a_modificar = solucion_hc_actual[i]
            t_original_avion = avion_a_modificar.t_seleccionado
            
            if t_original_avion is None: 
                if verbose_hc: print(f"    HC Advertencia: Avión ID {avion_a_modificar.id_avion} sin t_seleccionado en HC.")
                continue

            tiempos_candidatos: Set[int] = set()
            for k_delta in range(1, 4): 
                tiempos_candidatos.add(t_original_avion - (paso_tiempo_delta * k_delta))
                tiempos_candidatos.add(t_original_avion + (paso_tiempo_delta * k_delta))
            tiempos_candidatos.add(avion_a_modificar.tiempo_aterrizaje_preferente)
            tiempos_candidatos.add(avion_a_modificar.tiempo_aterrizaje_temprano)
            tiempos_candidatos.add(avion_a_modificar.tiempo_aterrizaje_tardio)
            
            tiempos_validos_ordenados = sorted(list(filter(
                lambda t: t != t_original_avion and \
                          avion_a_modificar.tiempo_aterrizaje_temprano <= t <= avion_a_modificar.tiempo_aterrizaje_tardio,
                tiempos_candidatos
            )))
            
            for nuevo_tiempo in tiempos_validos_ordenados:
                vecino = copy.deepcopy(solucion_hc_actual)
                avion_mod_vecino = next(a for a in vecino if a.id_avion == avion_a_modificar.id_avion)
                avion_mod_vecino.t_seleccionado = nuevo_tiempo

                if es_solucion_factible(vecino, numero_total_aviones, verbose=False): 
                    costo_vecino = calcular_costo_total_solucion(vecino)
                    if costo_vecino < costo_hc_actual:
                        solucion_hc_actual = copy.deepcopy(vecino)
                        costo_hc_actual = costo_vecino
                        mejora_encontrada_iter_local = True
                        if verbose_hc: print(f"      HC step (Iter local HC {iter_local+1}): Mejora local a {costo_hc_actual:.2f} (Avión {avion_a_modificar.id_avion} a t={nuevo_tiempo})")
                        break 
            
            if mejora_encontrada_iter_local:
                break 
        
        hc_cost_history.append((iter_local + 1, costo_hc_actual)) # Registrar costo al final de la iteración local de HC

        if not mejora_encontrada_iter_local:
            if verbose_hc: print(f"    HC óptimo local alcanzado en {iter_local+1} iteraciones locales. Costo local: {costo_hc_actual:.2f}")
            break 
    
    if iter_local + 1 >= max_iter_local_hc and mejora_encontrada_iter_local: # type: ignore
         if verbose_hc: print(f"    HC max iteraciones locales ({max_iter_local_hc}) alcanzado. Costo local final: {costo_hc_actual:.2f}")
    elif iter_local +1 >= max_iter_local_hc and not mejora_encontrada_iter_local: # type: ignore
         if verbose_hc: print(f"    HC max iteraciones locales ({max_iter_local_hc}) alcanzado, ya en óptimo local. Costo: {costo_hc_actual:.2f}")

    if not es_solucion_factible(solucion_hc_actual, numero_total_aviones, verbose=True):
        print(f"    ALERTA HC: La solución final del HC NO es factible. Costo reportado: {costo_hc_actual:.2f}")
        if es_solucion_factible(solucion_inicial_hc, numero_total_aviones, verbose=False):
             print(f"    HC revirtiendo a la solución inicial factible debido a que HC produjo infactibilidad.")
             # Recalcular el historial de HC para reflejar la reversión (o simplemente devolver el inicial)
             initial_hc_hist = [(0, calcular_costo_total_solucion(solucion_inicial_hc))]
             return copy.deepcopy(solucion_inicial_hc), calcular_costo_total_solucion(solucion_inicial_hc), initial_hc_hist

    return solucion_hc_actual, costo_hc_actual, hc_cost_history # MODIFICADO

# --- Perturbación ---
def perturbar_solucion_con_switches(
    solucion_original: List[Avion], 
    num_switches_deseados: int, 
    numero_total_aviones: int,
    verbose_perturb: bool = False
    ) -> List[Avion]:
    """
    Perturba una solución realizando un número deseado de intercambios (switches)
    de tiempos de aterrizaje entre pares de aviones, asegurando factibilidad.
    """
    solucion_perturbada = copy.deepcopy(solucion_original)
    
    indices_aviones_programados = [idx for idx, avion in enumerate(solucion_perturbada) if avion.t_seleccionado is not None]

    if len(indices_aviones_programados) < 2:
        if verbose_perturb: print("  Perturbación: No hay suficientes aviones programados para hacer switches.")
        return solucion_perturbada 

    switches_exitosos = 0
    max_intentos_globales_para_switches = num_switches_deseados * 20 

    for _ in range(max_intentos_globales_para_switches):
        if switches_exitosos >= num_switches_deseados:
            break

        idx_en_lista_avion1, idx_en_lista_avion2 = random.sample(indices_aviones_programados, 2)
        
        solucion_temporal_switch = copy.deepcopy(solucion_perturbada)
        
        avion1_obj_temp = solucion_temporal_switch[idx_en_lista_avion1]
        avion2_obj_temp = solucion_temporal_switch[idx_en_lista_avion2]

        t_original_avion1 = avion1_obj_temp.t_seleccionado
        avion1_obj_temp.t_seleccionado = avion2_obj_temp.t_seleccionado
        avion2_obj_temp.t_seleccionado = t_original_avion1
        
        if es_solucion_factible(solucion_temporal_switch, numero_total_aviones, verbose=False):
            solucion_perturbada = solucion_temporal_switch 
            switches_exitosos += 1
            if verbose_perturb: print(f"    Perturbación: Switch exitoso #{switches_exitosos} entre avión ID {avion1_obj_temp.id_avion} y ID {avion2_obj_temp.id_avion}.")
    
    if verbose_perturb:
        if switches_exitosos < num_switches_deseados:
            print(f"  Perturbación: Se realizaron {switches_exitosos} de {num_switches_deseados} switches deseados.")
        else:
            print(f"  Perturbación: Se realizaron los {switches_exitosos} switches deseados.")
            
    return solucion_perturbada

# --- Ejecución GRASP Individual ---
def ejecutar_grasp_una_vez(
    semilla: int,
    aviones_base_datos: List[Avion], 
    numero_total_aviones: int,
    params_grasp: dict,
    verbose_grasp: bool = False
    ) -> Tuple[Optional[List[Avion]], float, float, List[Tuple[int, float]]]: # MODIFICADO: Devuelve historial GRASP
    """
    Ejecuta una instancia completa de GRASP con ILS.
    Devuelve: (mejor_solucion_esta_grasp, mejor_costo_esta_grasp, costo_construccion_inicial, grasp_run_cost_history)
    """
    random.seed(semilla)
    if verbose_grasp: print(f"\n--- Iniciando Ejecución GRASP con Semilla: {semilla} ---")

    grasp_run_cost_history: List[Tuple[int, float]] = []
    global_grasp_step_counter = 0
    
    # --- 1. Fase de Construcción (Greedy Estocástico con Reintentos) ---
    if verbose_grasp: print("  1. Fase de Construcción (Greedy Estocástico)...")
    
    solucion_construida_obj: Optional[List[Avion]] = None
    costo_construccion_inicial = float('inf') 
    mejor_costo_esta_grasp = float('inf') # Mejor costo encontrado en esta ejecución GRASP
    mejor_solucion_esta_grasp: Optional[List[Avion]] = None
    es_factible_construccion_final = False
    intentos_construccion_realizados = 0

    for intento_construccion in range(params_grasp["max_intentos_greedy_construccion"]):
        intentos_construccion_realizados = intento_construccion + 1
        aviones_para_intento_actual = copy.deepcopy(aviones_base_datos)
        
        lista_procesada_intento, factibilidad_intento_greedy, _ = greedy_stochastico_Ek_ponderado_costo(
            aviones_para_intento_actual, 
            verbose_internal=params_grasp.get("verbose_greedy_construccion", False)
        )
        
        if factibilidad_intento_greedy:
            if es_solucion_factible(lista_procesada_intento, numero_total_aviones, verbose=params_grasp.get("verbose_greedy_construccion", False)):
                solucion_construida_obj = lista_procesada_intento
                es_factible_construccion_final = True
                costo_construccion_inicial = calcular_costo_total_solucion(solucion_construida_obj)
                mejor_costo_esta_grasp = costo_construccion_inicial
                mejor_solucion_esta_grasp = copy.deepcopy(solucion_construida_obj)
                
                grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp))
                global_grasp_step_counter += 1
                
                if verbose_grasp: 
                    print(f"    Construcción GRASP exitosa en intento {intentos_construccion_realizados}/{params_grasp['max_intentos_greedy_construccion']}. Costo: {costo_construccion_inicial:.2f}")
                break 
            else:
                 if verbose_grasp and params_grasp.get("verbose_greedy_construccion", False):
                    print(f"    Intento construcción {intentos_construccion_realizados}: Factible por greedy, pero no por chequeo global GRASP.")
        
        if verbose_grasp and params_grasp.get("verbose_greedy_construccion", False) and \
           (intentos_construccion_realizados % 100 == 0 or intentos_construccion_realizados == params_grasp["max_intentos_greedy_construccion"]):
             print(f"    Intento de construcción GRASP {intentos_construccion_realizados}/{params_grasp['max_intentos_greedy_construccion']} no fue factible.")

    if not es_factible_construccion_final or mejor_solucion_esta_grasp is None:
        if verbose_grasp: print(f"    Construcción fallida tras {intentos_construccion_realizados} intentos: No se generó solución inicial factible.")
        grasp_run_cost_history.append((global_grasp_step_counter, float('inf'))) # Registrar inf si no hay solución
        return None, float('inf'), float('inf'), grasp_run_cost_history

    # --- 2. Búsqueda Local Inicial (Hill Climbing) ---
    if verbose_grasp: print(f"\n  2. Búsqueda Local Inicial (Hill Climbing) tras construcción...")
    
    solucion_para_hc_inicial = copy.deepcopy(mejor_solucion_esta_grasp)
    # MODIFICADO: Recibir historial de HC
    solucion_tras_hc_inicial, costo_tras_hc_inicial, hc_init_history = ejecutar_hill_climbing(
        solucion_para_hc_inicial, 
        numero_total_aviones,
        params_grasp["paso_delta_hc"],
        params_grasp["max_iter_hc"],
        verbose_hc=params_grasp.get("verbose_hc", False)
    )
    
    # Registrar evolución durante HC inicial
    for hc_iter, hc_cost_at_iter in hc_init_history:
        if hc_iter > 0 : # El costo en iter 0 ya fue capturado por la construcción o es el mismo
            current_best_for_plot = min(mejor_costo_esta_grasp, hc_cost_at_iter if es_solucion_factible(solucion_tras_hc_inicial, numero_total_aviones) else float('inf'))
            grasp_run_cost_history.append((global_grasp_step_counter, current_best_for_plot))
            global_grasp_step_counter +=1

    if es_solucion_factible(solucion_tras_hc_inicial, numero_total_aviones, verbose=params_grasp.get("verbose_hc", False)):
        if costo_tras_hc_inicial < mejor_costo_esta_grasp:
            mejor_solucion_esta_grasp = copy.deepcopy(solucion_tras_hc_inicial)
            mejor_costo_esta_grasp = costo_tras_hc_inicial
            if verbose_grasp: print(f"    HC inicial mejoró la solución. Nuevo mejor costo GRASP: {mejor_costo_esta_grasp:.2f}")
    else:
        if verbose_grasp: print(f"    HC inicial produjo una solución infactible. Se descarta la mejora de HC inicial.")
    
    # Asegurar que el último estado de mejor_costo_esta_grasp se registre si no hubo pasos en HC o si HC no mejoró
    if not hc_init_history or hc_init_history[-1][0] == 0 : # Si HC no corrió o solo tuvo el paso 0
         grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp))
         # No se incrementa global_grasp_step_counter aquí si ya se hizo o no hubo HC real.
    
    # --- 3. Búsqueda Local Iterada (ILS) con Perturbación y HC ---
    if verbose_grasp: print(f"\n  3. Búsqueda Local Iterada (ILS) - {params_grasp['num_restarts_ils']} restarts de perturbación...")
    
    solucion_actual_para_ils = copy.deepcopy(mejor_solucion_esta_grasp) 

    for i_restart_ils in range(params_grasp["num_restarts_ils"]):
        if verbose_grasp: print(f"\n    ILS Restart #{i_restart_ils + 1}/{params_grasp['num_restarts_ils']}:")
        
        costo_base_perturbacion = calcular_costo_total_solucion(solucion_actual_para_ils) # type: ignore
        if verbose_grasp: print(f"      3a. Perturbando solución actual (costo: {costo_base_perturbacion:.2f})...")
        solucion_perturbada = perturbar_solucion_con_switches(
            solucion_actual_para_ils, # type: ignore
            params_grasp["num_switches_perturbacion"],
            numero_total_aviones,
            verbose_perturb=params_grasp.get("verbose_perturbacion", False)
        )
        
        if not es_solucion_factible(solucion_perturbada, numero_total_aviones, verbose=True): 
            if verbose_grasp: print("      ALERTA: Solución perturbada NO es factible. Saltando HC para este restart de ILS.")
            solucion_actual_para_ils = copy.deepcopy(mejor_solucion_esta_grasp) 
            grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp)) # Registrar estado actual
            global_grasp_step_counter += 1
            continue 

        costo_perturbada = calcular_costo_total_solucion(solucion_perturbada)
        if verbose_grasp: print(f"      Solución perturbada (factible). Costo: {costo_perturbada:.2f}")
        
        # Registrar costo después de perturbación (refleja el mejor actual antes de este HC)
        grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp))
        global_grasp_step_counter += 1

        if verbose_grasp: print(f"      3b. Aplicando HC a solución perturbada...")
        # MODIFICADO: Recibir historial de HC
        solucion_tras_hc_perturb, costo_tras_hc_perturb, hc_perturb_history = ejecutar_hill_climbing(
            solucion_perturbada,
            numero_total_aviones,
            params_grasp["paso_delta_hc"],
            params_grasp["max_iter_hc"],
            verbose_hc=params_grasp.get("verbose_hc", False)
        )

        # Registrar evolución durante HC de perturbación
        for hc_iter, hc_cost_at_iter in hc_perturb_history:
            if hc_iter > 0: # El costo en iter 0 es el de la solución perturbada
                current_best_for_plot = min(mejor_costo_esta_grasp, hc_cost_at_iter if es_solucion_factible(solucion_tras_hc_perturb, numero_total_aviones) else float('inf'))
                grasp_run_cost_history.append((global_grasp_step_counter, current_best_for_plot))
                global_grasp_step_counter +=1
        
        if es_solucion_factible(solucion_tras_hc_perturb, numero_total_aviones, verbose=params_grasp.get("verbose_hc", False)):
            # Decisión de aceptación para ILS (si mejora la solución actual de ILS)
            if costo_tras_hc_perturb < calcular_costo_total_solucion(solucion_actual_para_ils): # type: ignore
                 solucion_actual_para_ils = copy.deepcopy(solucion_tras_hc_perturb)

            # Actualizar la mejor solución global de esta corrida GRASP
            if costo_tras_hc_perturb < mejor_costo_esta_grasp:
                mejor_solucion_esta_grasp = copy.deepcopy(solucion_tras_hc_perturb)
                mejor_costo_esta_grasp = costo_tras_hc_perturb
                solucion_actual_para_ils = copy.deepcopy(mejor_solucion_esta_grasp) # Asegurar que ILS continúe desde la nueva mejor
                if verbose_grasp: print(f"      ILS encontró nueva mejor solución para esta GRASP. Costo: {mejor_costo_esta_grasp:.2f}")
            else:
                if verbose_grasp: print(f"      ILS no mejoró la mejor solución de esta GRASP (Mejor actual GRASP: {mejor_costo_esta_grasp:.2f})")
        else:
            if verbose_grasp: print(f"      HC tras perturbación produjo solución infactible. Se descarta este camino de ILS.")
            solucion_actual_para_ils = copy.deepcopy(mejor_solucion_esta_grasp)
        
        # Asegurar que el último estado de mejor_costo_esta_grasp se registre si no hubo pasos en HC o si HC no mejoró
        if not hc_perturb_history or hc_perturb_history[-1][0] == 0 :
            grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp))
            # No se incrementa aquí si ya se hizo o no hubo HC real

    if verbose_grasp: print(f"\n--- Fin Ejecución GRASP (Semilla: {semilla}). Mejor Costo Final Encontrado: {mejor_costo_esta_grasp:.2f} (Costo Inicial Construcción: {costo_construccion_inicial:.2f}) ---")
    
    if mejor_solucion_esta_grasp is None or not es_solucion_factible(mejor_solucion_esta_grasp, numero_total_aviones, verbose=True):
        print(f"  ALERTA FINAL GRASP (Semilla {semilla}): La mejor solución reportada por esta ejecución GRASP NO es factible o es None. Costo Final: {mejor_costo_esta_grasp}, Costo Inicial: {costo_construccion_inicial}")
        # Añadir un punto final al historial si no existe o es infactible
        if not grasp_run_cost_history or grasp_run_cost_history[-1][0] < global_grasp_step_counter:
            grasp_run_cost_history.append((global_grasp_step_counter, float('inf')))
        return None, float('inf'), costo_construccion_inicial, grasp_run_cost_history

    # Asegurar un último punto en el historial con el costo final
    if not grasp_run_cost_history or grasp_run_cost_history[-1][0] < global_grasp_step_counter :
         grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp))
    elif grasp_run_cost_history and grasp_run_cost_history[-1][1] > mejor_costo_esta_grasp: # Actualizar el último punto si se encontró mejor costo
        grasp_run_cost_history[-1] = (grasp_run_cost_history[-1][0], mejor_costo_esta_grasp)


    return mejor_solucion_esta_grasp, mejor_costo_esta_grasp, costo_construccion_inicial, grasp_run_cost_history

# --- Función Principal del Runner GRASP con ILS ---
def main_grasp_ils_runner():
    """
    Función principal para ejecutar el GRASP con ILS múltiples veces y graficar resultados.
    """
    ruta_archivo_datos = "case1.txt" # O "case1.txt" etc.
    semillas_predefinidas = [42, 123, 7, 99, 500, 777, 2024, 1, 100, 314] 
    # semillas_predefinidas = [42, 123] # Para pruebas rápidas
    
    parametros_grasp = {
        "max_intentos_greedy_construccion": 1000, 
        "paso_delta_hc": 50,                   
        "max_iter_hc": 500, # Reducido para que los gráficos no sean demasiado largos por HC                   
        "num_restarts_ils": 5, # Reducido para ILS                 
        "num_switches_perturbacion": 3,        
        "verbose_grasp": True, # Poner en False para menos output                 
        "verbose_greedy_construccion": False,  
        "verbose_hc": False,                   
        "verbose_perturbacion": False          
    }

    print(">>> Iniciando Proceso GRASP con Búsqueda Local Iterada (ILS) <<<")
    print(f"Archivo de datos: {ruta_archivo_datos}")
    print(f"Número de ejecuciones GRASP separadas: {len(semillas_predefinidas)}")
    print(f"Parámetros GRASP/ILS: {parametros_grasp}")

    try:
        aviones_base_originales = leer_datos_aviones(ruta_archivo_datos)
        numero_aviones = len(aviones_base_originales)
        if numero_aviones == 0:
            print("Error: No se leyeron aviones del archivo.")
            return
        print(f"Número total de aviones en el problema: {numero_aviones}")
    except Exception as e:
        print(f"Error crítico leyendo datos de aviones: {e}")
        return

    mejor_solucion_global_obj: Optional[List[Avion]] = None
    costo_mejor_solucion_global: float = float('inf')
    semilla_mejor_solucion_global: Optional[int] = None
    costo_inicial_mejor_solucion_global: float = float('inf')

    resultados_por_semilla = []
    historias_de_costos_grasp: List[Tuple[int, List[Tuple[int, float]]]] = [] # Para graficar

    for idx_seed, semilla_actual in enumerate(semillas_predefinidas):
        print(f"\n========================= EJECUCIÓN GRASP GLOBAL #{idx_seed + 1}/{len(semillas_predefinidas)} (Semilla: {semilla_actual}) =========================")
        
        # MODIFICADO: Desempaquetar también grasp_run_history
        sol_grasp_actual, costo_final_grasp_actual, costo_inicial_grasp_actual, grasp_run_history = ejecutar_grasp_una_vez(
            semilla_actual,
            copy.deepcopy(aviones_base_originales), 
            numero_aviones,
            parametros_grasp,
            verbose_grasp=parametros_grasp["verbose_grasp"]
        )
        
        resultados_por_semilla.append({
            "semilla": semilla_actual, 
            "costo_inicial": costo_inicial_grasp_actual, 
            "costo_final": costo_final_grasp_actual, 
            "solucion_obj_presente": sol_grasp_actual is not None 
        })
        historias_de_costos_grasp.append((semilla_actual, grasp_run_history)) # Guardar historial para graficar

        if sol_grasp_actual is not None and costo_final_grasp_actual < costo_mejor_solucion_global:
            if es_solucion_factible(sol_grasp_actual, numero_aviones, verbose=True): 
                print(f"¡NUEVA MEJOR SOLUCIÓN GLOBAL encontrada por semilla {semilla_actual}!")
                print(f"  Costo Inicial (Construcción): {costo_inicial_grasp_actual:.2f}")
                print(f"  Costo Final Anterior: {costo_mejor_solucion_global:.2f}, Nuevo Costo Final: {costo_final_grasp_actual:.2f}")
                costo_mejor_solucion_global = costo_final_grasp_actual
                mejor_solucion_global_obj = copy.deepcopy(sol_grasp_actual)
                semilla_mejor_solucion_global = semilla_actual
                costo_inicial_mejor_solucion_global = costo_inicial_grasp_actual
            else:
                print(f"Ejecución GRASP con semilla {semilla_actual} reportó costo final {costo_final_grasp_actual:.2f} pero la solución NO ES FACTIBLE globalmente. Descartada.")
                for res in resultados_por_semilla:
                    if res["semilla"] == semilla_actual:
                        res["costo_final"] = float('inf')
                        res["solucion_obj_presente"] = False
                        break
        elif sol_grasp_actual is None:
            print(f"Ejecución GRASP con semilla {semilla_actual} no produjo una solución factible.")
        else: 
            if not es_solucion_factible(sol_grasp_actual, numero_aviones, verbose=False):
                 print(f"Ejecución GRASP con semilla {semilla_actual} finalizada. Costo Inicial: {costo_inicial_grasp_actual:.2f}, Costo Final: {costo_final_grasp_actual:.2f}, pero la solución NO ES FACTIBLE. (No superó el mejor global: {costo_mejor_solucion_global:.2f})")
                 for res in resultados_por_semilla:
                    if res["semilla"] == semilla_actual:
                        res["costo_final"] = float('inf')
                        res["solucion_obj_presente"] = False
                        break
            else:
                print(f"Ejecución GRASP con semilla {semilla_actual} finalizada. Costo Inicial: {costo_inicial_grasp_actual:.2f}, Costo Final: {costo_final_grasp_actual:.2f}. (No superó el mejor global: {costo_mejor_solucion_global:.2f})")


    print("\n\n========================= RESULTADOS FINALES DEL PROCESO GRASP-ILS =========================")
    print("\nResumen de costos por semilla (Inicial -> Final):")
    for res in resultados_por_semilla:
        costo_inicial_str = f"{res['costo_inicial']:.2f}" if res['costo_inicial'] != float('inf') else "Inf (No factible)"
        costo_final_str = f"{res['costo_final']:.2f}" if res['costo_final'] != float('inf') else "Inf (No factible)"
        print(f"  Semilla: {res['semilla']:<4} -> Costo Inicial: {costo_inicial_str}, Costo Final: {costo_final_str}")

    if mejor_solucion_global_obj:
        print(f"\nMejor solución global encontrada con costo inicial: {costo_inicial_mejor_solucion_global:.2f} y costo final: {costo_mejor_solucion_global:.2f} (originada por semilla: {semilla_mejor_solucion_global})")
        print("Horario de la mejor solución global:")
        imprimir_horario_aterrizaje_gs(mejor_solucion_global_obj, f"Mejor Horario Global GRASP-ILS (Semilla {semilla_mejor_solucion_global})")
        
        if not es_solucion_factible(mejor_solucion_global_obj, numero_aviones, verbose=True):
            print("\nALERTA CRÍTICA: La mejor solución global reportada NO es factible según la verificación final.")
        else:
            print("\nLa mejor solución global reportada ES factible según la verificación final.")
            
    else:
        print("\nNo se encontró ninguna solución factible en ninguna de las ejecuciones GRASP.")

    # --- AÑADIDO: Sección de Graficación ---
    plt.figure(figsize=(15, 8))
    plot_successful = False
    max_steps_overall = 0

    for semilla, historia_costos_run in historias_de_costos_grasp:
        if historia_costos_run and historia_costos_run[0][1] != float('inf') : # Solo graficar si hay datos y el costo inicial no es inf
            pasos = [item[0] for item in historia_costos_run]
            costos = [item[1] for item in historia_costos_run]
            if pasos: # Asegurarse que hay datos
                 max_steps_overall = max(max_steps_overall, pasos[-1])
                 plt.plot(pasos, costos, marker='.', linestyle='-', alpha=0.7, label=f"Semilla {semilla}")
                 plot_successful = True
        else:
            print(f"Nota: Historia de costos para GRASP con semilla '{semilla}' no se grafica (costo inicial inf o sin datos).")

    if plot_successful:
        plt.xlabel("Paso de Evaluación en GRASP-ILS")
        plt.ylabel("Mejor Costo Encontrado (hasta el paso)")
        titulo_grafico = (f"Evolución del Costo en GRASP-ILS para '{ruta_archivo_datos}'\n"
                          f"({len(semillas_predefinidas)} Semillas)")
        plt.title(titulo_grafico)
        plt.legend(title="Ejecuciones GRASP", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.80, 1]) # Ajustar para que la leyenda no se corte si está fuera

        nombre_grafico = f"grasp_ils_costo_evolucion_{ruta_archivo_datos.split('.')[0]}.png"
        try:
            plt.savefig(nombre_grafico)
            print(f"\nGráfico de evolución de costos GRASP-ILS guardado como: {nombre_grafico}")
        except Exception as e_plot:
            print(f"Error al guardar el gráfico de GRASP-ILS: {e_plot}")
        plt.show()
    else:
        print("\nNo se generó ningún gráfico de evolución de costos porque no hubo ejecuciones GRASP válidas para graficar.")


if __name__ == "__main__":
    main_grasp_ils_runner()
