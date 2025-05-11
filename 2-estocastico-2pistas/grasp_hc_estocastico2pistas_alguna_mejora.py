import sys
import math
import copy # Para deepcopy
import random # Para GRASP y el constructor estocástico
from typing import List, Tuple, Optional, Set
import matplotlib.pyplot as plt # <--- AÑADIDO para graficar

# Constante para el número de pistas
NUMERO_DE_PISTAS = 2

# Intentar importar clases y funciones necesarias
try:
    from greedyestocastico_2pistas_mas_random import (
        Avion,
        leer_datos_aviones,
        greedy_estocastico_2pistas_mas_random,
    )
except ImportError:
    print("Error: No se pudo importar de 'greedyestocastico_2pistas_mas_random.py'.")
    print("Asegúrate de que dicho archivo exista en el mismo directorio y contenga las definiciones necesarias.")
    sys.exit(1)

# --- Funciones de Cálculo de Costo y Factibilidad ---
def calcular_costo_individual_avion(avion: Avion, tiempo_aterrizaje_propuesto: int) -> float:
    if tiempo_aterrizaje_propuesto is None:
        return float('inf') 
    diferencia = tiempo_aterrizaje_propuesto - avion.tiempo_aterrizaje_preferente
    costo = 0.0
    if diferencia < 0:
        costo = abs(diferencia) * avion.penalizacion_antes_preferente
    elif diferencia > 0:
        costo = diferencia * avion.penalizacion_despues_preferente
    return costo

def calcular_costo_total_solucion(solucion: List[Avion]) -> float:
    costo_total = 0.0
    for avion in solucion:
        if avion.t_seleccionado is None or avion.pista_seleccionada is None:
            # Penalización alta si un avión no está completamente programado en una solución "completa"
            costo_total += 1_000_000_000 
        else:
            costo_total += calcular_costo_individual_avion(avion, avion.t_seleccionado)
    return costo_total

def es_solucion_factible_2pistas(solucion: List[Avion], numero_total_aviones_problema: int, verbose: bool = False) -> bool:
    if not solucion:
        if verbose: print("  [Factibilidad] Falla: La solución está vacía.")
        return False

    aviones_completamente_programados = 0
    ids_aviones_en_solucion = set()

    for avion_obj in solucion:
        if avion_obj.t_seleccionado is not None and avion_obj.pista_seleccionada is not None:
            if not (0 <= avion_obj.pista_seleccionada < NUMERO_DE_PISTAS):
                if verbose: print(f"  [Factibilidad] Falla: Avión ID {avion_obj.id_avion} con pista inválida {avion_obj.pista_seleccionada}.")
                return False
            aviones_completamente_programados += 1
            ids_aviones_en_solucion.add(avion_obj.id_avion)
        else: 
            if verbose: print(f"  [Factibilidad] Falla: Avión ID {avion_obj.id_avion} no tiene tiempo ({avion_obj.t_seleccionado}) o pista ({avion_obj.pista_seleccionada}) asignada.")
            return False 

    if aviones_completamente_programados != numero_total_aviones_problema:
        if verbose: print(f"  [Factibilidad] Falla: Número de aviones programados ({aviones_completamente_programados}) no coincide con el total ({numero_total_aviones_problema}).")
        return False
    
    if len(ids_aviones_en_solucion) != numero_total_aviones_problema: 
        if verbose: print(f"  [Factibilidad] Falla: Discrepancia en IDs de aviones programados. Esperados: {numero_total_aviones_problema}, Encontrados únicos: {len(ids_aviones_en_solucion)}.")
        return False

    for avion in solucion: 
        if not (avion.tiempo_aterrizaje_temprano <= avion.t_seleccionado <= avion.tiempo_aterrizaje_tardio): # type: ignore
            if verbose: print(f"  [Factibilidad] Falla: Avión ID {avion.id_avion} t={avion.t_seleccionado} fuera de ventana [{avion.tiempo_aterrizaje_temprano}, {avion.tiempo_aterrizaje_tardio}].")
            return False

    for pista_idx_verif in range(NUMERO_DE_PISTAS):
        aviones_en_esta_pista = sorted(
            [a for a in solucion if a.pista_seleccionada == pista_idx_verif],
            key=lambda x: x.t_seleccionado # type: ignore
        )
        for i in range(len(aviones_en_esta_pista) - 1):
            avion_i = aviones_en_esta_pista[i]
            avion_j = aviones_en_esta_pista[i+1]
            separacion_requerida_S_ij = avion_i.get_separation_to(avion_j)
            separacion_real = avion_j.t_seleccionado - avion_i.t_seleccionado # type: ignore
            if separacion_real < separacion_requerida_S_ij:
                if verbose: print(f"  [Factibilidad] Falla: Separación en Pista {pista_idx_verif} entre ID {avion_i.id_avion} (t={avion_i.t_seleccionado}) y "
                               f"ID {avion_j.id_avion} (t={avion_j.t_seleccionado}). Real: {separacion_real}, Req: {separacion_requerida_S_ij}.")
                return False
    if verbose: print("  [Factibilidad] Solución es factible.")
    return True

def _calcular_tiempo_min_aterrizaje_en_pista_hc(
    avion_k: Avion, 
    pista_idx: int, 
    otros_aviones_en_pista: List[Avion], 
    tiempo_referencia_aterrizaje: Optional[int] = None
    ) -> int:
    """Calcula el tiempo más temprano posible para avion_k en pista_idx considerando otros_aviones_en_pista."""
    t_min_calculado = avion_k.tiempo_aterrizaje_temprano
    if tiempo_referencia_aterrizaje is not None:
        t_min_calculado = max(t_min_calculado, tiempo_referencia_aterrizaje)

    for avion_ya_en_pista in otros_aviones_en_pista: 
        if avion_ya_en_pista.t_seleccionado is None: continue 
        t_min_calculado = max(t_min_calculado, avion_ya_en_pista.t_seleccionado + avion_ya_en_pista.get_separation_to(avion_k))
    return t_min_calculado

def imprimir_matriz_aterrizaje_local_con_cabecera(lista_aviones_procesada: List[Avion], cabecera: str) -> None:
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
    else:
        aviones_aterrizados.sort(key=lambda avion: (avion.t_seleccionado, avion.pista_seleccionada, avion.id_avion))
        for avion in aviones_aterrizados:
            tiempo_str = f"{avion.t_seleccionado:<18}"
            avion_id_str = f"{avion.id_avion:<10}"
            pista_str = f"{avion.pista_seleccionada:<7}" 
            print(f"| {tiempo_str} | {avion_id_str} | {pista_str} |")
    print("----------------------------------------------------")

# --- Hill Climbing (Alguna-Mejora Determinista para 2 Pistas) ---
def hill_climbing_alguna_mejora_2pistas(
    solucion_inicial: List[Avion],
    numero_total_aviones_problema: int,
    paso_tiempo_delta: int = 10,
    max_iter_hc: int = 100, 
    verbose_hc_detalle: bool = False 
    ) -> Tuple[List[Avion], float, List[Tuple[int, float]]]: # MODIFICADO: Devuelve historial HC
    
    solucion_actual_hc = copy.deepcopy(solucion_inicial)
    hc_cost_history: List[Tuple[int, float]] = []

    if not es_solucion_factible_2pistas(solucion_actual_hc, numero_total_aviones_problema, verbose=verbose_hc_detalle):
        if verbose_hc_detalle: print("  HC Error: La solución inicial para Hill Climbing no es factible. Devolviendo la original.")
        costo_infactible = calcular_costo_total_solucion(solucion_inicial) # Podría ser inf
        hc_cost_history.append((0, costo_infactible))
        return solucion_inicial, costo_infactible, hc_cost_history

    costo_actual_hc = calcular_costo_total_solucion(solucion_actual_hc)
    hc_cost_history.append((0, costo_actual_hc)) # Costo antes de la primera iteración de HC

    if verbose_hc_detalle: print(f"    HC Iniciando con costo: {costo_actual_hc:.2f}")

    iteraciones_hc_global = 0
    for iter_g in range(max_iter_hc): # Reemplazo del while con for para asegurar fin
        iteraciones_hc_global = iter_g + 1
        mejora_encontrada_en_esta_iteracion_global = False
        
        aviones_indices_barajados = list(range(numero_total_aviones_problema))
        random.shuffle(aviones_indices_barajados)

        for avion_idx in aviones_indices_barajados:
            avion_a_modificar = solucion_actual_hc[avion_idx] 
            
            t_original_avion = avion_a_modificar.t_seleccionado
            p_original_avion = avion_a_modificar.pista_seleccionada

            if t_original_avion is None or p_original_avion is None: continue

            # Movimiento Tipo 1: Cambiar tiempo de aterrizaje en la MISMA PISTA
            tiempos_a_probar_misma_pista: Set[int] = set()
            for t_clave in [avion_a_modificar.tiempo_aterrizaje_preferente,
                            avion_a_modificar.tiempo_aterrizaje_temprano,
                            avion_a_modificar.tiempo_aterrizaje_tardio]:
                if t_clave != t_original_avion and \
                   avion_a_modificar.tiempo_aterrizaje_temprano <= t_clave <= avion_a_modificar.tiempo_aterrizaje_tardio:
                    tiempos_a_probar_misma_pista.add(t_clave)
            for k_delta in range(1, 4): 
                for sign in [-1, 1]:
                    t_delta = t_original_avion + sign * k_delta * paso_tiempo_delta
                    if t_delta != t_original_avion and \
                       avion_a_modificar.tiempo_aterrizaje_temprano <= t_delta <= avion_a_modificar.tiempo_aterrizaje_tardio:
                        tiempos_a_probar_misma_pista.add(t_delta)
            
            tiempos_ordenados_misma_pista = sorted(list(tiempos_a_probar_misma_pista))

            for nuevo_t_propuesto in tiempos_ordenados_misma_pista:
                vecino_potencial = copy.deepcopy(solucion_actual_hc)
                avion_mod_vecino = next(a for a in vecino_potencial if a.id_avion == avion_a_modificar.id_avion)
                avion_mod_vecino.t_seleccionado = nuevo_t_propuesto

                if es_solucion_factible_2pistas(vecino_potencial, numero_total_aviones_problema, verbose=False):
                    costo_vecino = calcular_costo_total_solucion(vecino_potencial)
                    if costo_vecino < costo_actual_hc:
                        if verbose_hc_detalle: print(f"      HC Mejora (Misma Pista): Avión ID {avion_mod_vecino.id_avion} t={t_original_avion}->{nuevo_t_propuesto} P{p_original_avion}. Costo {costo_actual_hc:.2f}->{costo_vecino:.2f}")
                        solucion_actual_hc = vecino_potencial
                        costo_actual_hc = costo_vecino
                        mejora_encontrada_en_esta_iteracion_global = True
                        break 
            
            if mejora_encontrada_en_esta_iteracion_global: break 

            # Movimiento Tipo 2: Cambiar a la OTRA PISTA
            if NUMERO_DE_PISTAS > 1:
                otra_pista = 1 - p_original_avion 
                
                aviones_fijos_en_otra_pista = [
                    a for a in solucion_actual_hc 
                    if a.pista_seleccionada == otra_pista and a.id_avion != avion_a_modificar.id_avion
                ]
                
                tiempos_a_probar_otra_pista: Set[int] = set()
                t_min_base_otra_pista = _calcular_tiempo_min_aterrizaje_en_pista_hc(avion_a_modificar, otra_pista, aviones_fijos_en_otra_pista)

                if t_min_base_otra_pista <= avion_a_modificar.tiempo_aterrizaje_tardio:
                    tiempos_a_probar_otra_pista.add(t_min_base_otra_pista)
                
                pk_avion = avion_a_modificar.tiempo_aterrizaje_preferente
                if pk_avion >= t_min_base_otra_pista and pk_avion <= avion_a_modificar.tiempo_aterrizaje_tardio:
                     tiempos_a_probar_otra_pista.add(pk_avion)

                if t_original_avion >= t_min_base_otra_pista and t_original_avion <= avion_a_modificar.tiempo_aterrizaje_tardio:
                    tiempos_a_probar_otra_pista.add(t_original_avion)
                    for k_delta in range(1, 3): 
                        for sign in [-1, 1]:
                            t_delta_otra = t_original_avion + sign * k_delta * paso_tiempo_delta
                            if t_delta_otra >= t_min_base_otra_pista and t_delta_otra <= avion_a_modificar.tiempo_aterrizaje_tardio:
                                tiempos_a_probar_otra_pista.add(t_delta_otra)
                
                tiempos_ordenados_otra_pista = sorted(list(tiempos_a_probar_otra_pista))

                for nuevo_t_otra_pista in tiempos_ordenados_otra_pista:
                    vecino_potencial = copy.deepcopy(solucion_actual_hc)
                    avion_mod_vecino = next(a for a in vecino_potencial if a.id_avion == avion_a_modificar.id_avion)
                    avion_mod_vecino.t_seleccionado = nuevo_t_otra_pista
                    avion_mod_vecino.pista_seleccionada = otra_pista

                    if es_solucion_factible_2pistas(vecino_potencial, numero_total_aviones_problema, verbose=False):
                        costo_vecino = calcular_costo_total_solucion(vecino_potencial)
                        if costo_vecino < costo_actual_hc:
                            if verbose_hc_detalle: print(f"      HC Mejora (Cambio Pista): Avión ID {avion_mod_vecino.id_avion} (t={t_original_avion},p={p_original_avion})->(t={nuevo_t_otra_pista},p={otra_pista}). Costo {costo_actual_hc:.2f}->{costo_vecino:.2f}")
                            solucion_actual_hc = vecino_potencial
                            costo_actual_hc = costo_vecino
                            mejora_encontrada_en_esta_iteracion_global = True
                            break 
                if mejora_encontrada_en_esta_iteracion_global: break 
        
        hc_cost_history.append((iteraciones_hc_global, costo_actual_hc)) # Registrar costo al final de la iteración global de HC

        if not mejora_encontrada_en_esta_iteracion_global:
            if verbose_hc_detalle: print(f"    HC: No se encontró mejora en iteración global {iteraciones_hc_global}. Óptimo local.")
            break 
    
    if verbose_hc_detalle:
        if iteraciones_hc_global >= max_iter_hc : print(f"    HC: Límite de iteraciones ({max_iter_hc}) alcanzado.")
        print(f"    HC Finalizado. Costo final HC: {costo_actual_hc:.2f}")
    
    # Asegurar que la solución devuelta sea factible
    if not es_solucion_factible_2pistas(solucion_actual_hc, numero_total_aviones_problema, verbose=True):
        print(f"    ALERTA HC: La solución final del HC NO es factible. Costo reportado: {costo_actual_hc:.2f}")
        # Podría revertir a la solucion_inicial si era factible, o devolver la infactible con su historial.
        # Por ahora, devolvemos la solución (potencialmente infactible) y su historial tal cual.
        # Si se quisiera revertir:
        # if es_solucion_factible_2pistas(solucion_inicial, numero_total_aviones_problema):
        #     print("     HC revirtiendo a la solución inicial que era factible.")
        #     costo_inicial_calc = calcular_costo_total_solucion(solucion_inicial)
        #     return solucion_inicial, costo_inicial_calc, [(0, costo_inicial_calc)]
    
    return solucion_actual_hc, costo_actual_hc, hc_cost_history # MODIFICADO

# --- Perturbación para ILS (2 Pistas) ---
def perturbar_solucion_2pistas_con_switches(
    solucion_original: List[Avion],
    num_switches_deseados: int,
    numero_total_aviones: int,
    verbose_perturb: bool = False
) -> List[Avion]:
    """
    Perturba una solución realizando un número deseado de intercambios (switches)
    de tiempos y pistas de aterrizaje entre pares de aviones, asegurando factibilidad.
    """
    solucion_perturbada = copy.deepcopy(solucion_original)
    
    if not es_solucion_factible_2pistas(solucion_original, numero_total_aviones, verbose=False):
        if verbose_perturb: print("  Perturbación ILS: Solución original no es factible. No se perturbará.")
        return solucion_original 

    indices_aviones = list(range(numero_total_aviones))
    if len(indices_aviones) < 2:
        if verbose_perturb: print("  Perturbación ILS: No hay suficientes aviones para hacer switches.")
        return solucion_perturbada 

    switches_exitosos = 0
    max_intentos_por_switch_valido = 30 
    intentos_globales_para_switches = num_switches_deseados * max_intentos_por_switch_valido

    for _ in range(intentos_globales_para_switches):
        if switches_exitosos >= num_switches_deseados:
            break

        idx1, idx2 = random.sample(indices_aviones, 2)
        
        solucion_temporal_switch = copy.deepcopy(solucion_perturbada)
        
        avion1_obj_temp = solucion_temporal_switch[idx1]
        avion2_obj_temp = solucion_temporal_switch[idx2]

        t_original_avion1 = avion1_obj_temp.t_seleccionado
        p_original_avion1 = avion1_obj_temp.pista_seleccionada
        t_original_avion2 = avion2_obj_temp.t_seleccionado
        p_original_avion2 = avion2_obj_temp.pista_seleccionada

        avion1_obj_temp.t_seleccionado = t_original_avion2
        avion1_obj_temp.pista_seleccionada = p_original_avion2
        avion2_obj_temp.t_seleccionado = t_original_avion1
        avion2_obj_temp.pista_seleccionada = p_original_avion1
        
        if es_solucion_factible_2pistas(solucion_temporal_switch, numero_total_aviones, verbose=False):
            solucion_perturbada = solucion_temporal_switch 
            switches_exitosos += 1
            if verbose_perturb: 
                print(f"    Perturbación ILS: Switch #{switches_exitosos} entre ID {avion1_obj_temp.id_avion} (orig t={t_original_avion1},p={p_original_avion1}) y "
                      f"ID {avion2_obj_temp.id_avion} (orig t={t_original_avion2},p={p_original_avion2}).")
    
    if verbose_perturb:
        if switches_exitosos < num_switches_deseados:
            print(f"  Perturbación ILS: {switches_exitosos}/{num_switches_deseados} switches deseados realizados.")
        else:
            print(f"  Perturbación ILS: {switches_exitosos} switches deseados realizados.")
            
    return solucion_perturbada

# --- Ejecución GRASP Individual con ILS ---
def ejecutar_grasp_una_vez_con_ils(
    semilla_construccion: int,
    aviones_base_datos: List[Avion],
    numero_total_aviones: int,
    params_grasp: dict,
    verbose_grasp_general: bool,
    verbose_greedy_constructor_detalle: bool,
    verbose_hc_aplicado_resumen: bool, 
    verbose_hc_detallado_interno: bool, 
    verbose_perturbacion_detalle: bool
) -> Tuple[Optional[List[Avion]], float, float, List[Tuple[int, float]]]: # MODIFICADO: Devuelve historial GRASP
    
    random.seed(semilla_construccion) 
    grasp_run_cost_history: List[Tuple[int, float]] = []
    global_grasp_step_counter = 0

    # --- 1. Fase de Construcción ---
    solucion_construida_obj: Optional[List[Avion]] = None
    costo_inicial_construido = float('inf')
    mejor_costo_esta_grasp_iter = float('inf')
    mejor_solucion_esta_grasp_iter: Optional[List[Avion]] = None
    max_intentos_constr = params_grasp.get("max_intentos_construccion_por_grasp", 1000)
    
    if verbose_grasp_general: print("  GRASP: Iniciando Fase de Construcción (2 Pistas)...")
    for intento_constr in range(max_intentos_constr):
        aviones_para_intento_construccion = copy.deepcopy(aviones_base_datos)
        
        sol_intento, fact_intento, _ = greedy_estocastico_2pistas_mas_random(
            aviones_para_intento_construccion, 
            verbose_internal=verbose_greedy_constructor_detalle
        )
        
        if fact_intento and es_solucion_factible_2pistas(sol_intento, numero_total_aviones, verbose=verbose_greedy_constructor_detalle):
            solucion_construida_obj = sol_intento
            costo_inicial_construido = calcular_costo_total_solucion(solucion_construida_obj)
            mejor_costo_esta_grasp_iter = costo_inicial_construido
            mejor_solucion_esta_grasp_iter = copy.deepcopy(solucion_construida_obj)
            
            grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp_iter))
            global_grasp_step_counter += 1
            
            if verbose_grasp_general: print(f"    Construcción GRASP (2 Pistas) exitosa en intento {intento_constr + 1}. Costo: {costo_inicial_construido:.2f}")
            break 
        
        if verbose_greedy_constructor_detalle and (intento_constr + 1) % 50 == 0:
             print(f"    Intento de construcción GRASP (2 Pistas) {intento_constr + 1} no fue factible.")
    
    if mejor_solucion_esta_grasp_iter is None: # Si no se pudo construir
        if verbose_grasp_general: print(f"  GRASP (2 Pistas): Construcción fallida tras {max_intentos_constr} intentos.")
        grasp_run_cost_history.append((global_grasp_step_counter, float('inf')))
        return None, float('inf'), float('inf'), grasp_run_cost_history
    
    # --- 2. Búsqueda Local Inicial (Hill Climbing) ---
    if verbose_hc_aplicado_resumen: print(f"  GRASP (2 Pistas): Aplicando HC inicial (Costo: {mejor_costo_esta_grasp_iter:.2f})...")
    
    sol_hc_inicial, costo_hc_inicial, hist_hc_inicial = hill_climbing_alguna_mejora_2pistas(
        copy.deepcopy(mejor_solucion_esta_grasp_iter), # Pasar copia
        numero_total_aviones,
        paso_tiempo_delta=params_grasp.get("paso_delta_hc", 10),
        max_iter_hc=params_grasp.get("max_iter_hc", 100),
        verbose_hc_detalle=verbose_hc_detallado_interno
    )

    for hc_iter, hc_cost_at_iter in hist_hc_inicial:
        if hc_iter > 0: # El costo en iter 0 ya fue capturado o es el mismo
            current_best_for_plot = min(mejor_costo_esta_grasp_iter, hc_cost_at_iter if es_solucion_factible_2pistas(sol_hc_inicial, numero_total_aviones) else float('inf'))
            grasp_run_cost_history.append((global_grasp_step_counter, current_best_for_plot))
            global_grasp_step_counter +=1
            if hc_cost_at_iter < mejor_costo_esta_grasp_iter and es_solucion_factible_2pistas(sol_hc_inicial, numero_total_aviones):
                 mejor_costo_esta_grasp_iter = hc_cost_at_iter # Actualizar el mejor si HC mejora

    if es_solucion_factible_2pistas(sol_hc_inicial, numero_total_aviones, verbose=verbose_hc_aplicado_resumen):
        if verbose_hc_aplicado_resumen: print(f"  GRASP (2 Pistas): Costo tras HC inicial: {costo_hc_inicial:.2f}")
        if costo_hc_inicial < mejor_costo_esta_grasp_iter: # Ya actualizado arriba, pero re-chequeo por si acaso
            mejor_solucion_esta_grasp_iter = copy.deepcopy(sol_hc_inicial)
            mejor_costo_esta_grasp_iter = costo_hc_inicial 
            if verbose_hc_aplicado_resumen: print(f"    HC inicial mejoró. Nuevo mejor costo GRASP iter: {mejor_costo_esta_grasp_iter:.2f}")
    else:
        if verbose_hc_aplicado_resumen: print("  GRASP (2 Pistas): HC inicial produjo solución infactible.")

    if not hist_hc_inicial or hist_hc_inicial[-1][0] == 0 :
         grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp_iter))
    
    # --- 3. Búsqueda Local Iterada (ILS) ---
    if verbose_grasp_general: print(f"\n  GRASP (2 Pistas): Iniciando Fase ILS ({params_grasp.get('num_restarts_ils', 10)} restarts)...")
    solucion_base_ils = copy.deepcopy(mejor_solucion_esta_grasp_iter)

    for i_restart_ils in range(params_grasp.get("num_restarts_ils", 10)):
        if verbose_grasp_general: print(f"    ILS Restart #{i_restart_ils + 1}/{params_grasp.get('num_restarts_ils', 10)}")
        
        current_costo_base_ils = calcular_costo_total_solucion(solucion_base_ils) # type: ignore
        if verbose_perturbacion_detalle: print(f"      ILS: Perturbando solución base (costo: {current_costo_base_ils:.2f})")
        solucion_perturbada = perturbar_solucion_2pistas_con_switches(
            solucion_base_ils, # type: ignore
            params_grasp.get("num_switches_perturbacion", 3),
            numero_total_aviones,
            verbose_perturb=verbose_perturbacion_detalle
        )
        
        if not es_solucion_factible_2pistas(solucion_perturbada, numero_total_aviones, verbose=verbose_perturbacion_detalle):
            if verbose_grasp_general: print("      ILS: Solución perturbada NO es factible. Reajustando base ILS.")
            solucion_base_ils = copy.deepcopy(mejor_solucion_esta_grasp_iter) 
            grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp_iter))
            global_grasp_step_counter += 1
            continue 

        if verbose_perturbacion_detalle: print(f"      ILS: Solución perturbada (factible). Costo: {calcular_costo_total_solucion(solucion_perturbada):.2f}")
        
        grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp_iter)) # Costo antes de HC en ILS
        global_grasp_step_counter += 1

        if verbose_hc_aplicado_resumen: print(f"      ILS: Aplicando HC a solución perturbada...")
        sol_hc_perturb, costo_hc_perturb, hist_hc_perturb = hill_climbing_alguna_mejora_2pistas(
            solucion_perturbada,
            numero_total_aviones,
            paso_tiempo_delta=params_grasp.get("paso_delta_hc", 10),
            max_iter_hc=params_grasp.get("max_iter_hc", 100),
            verbose_hc_detalle=verbose_hc_detallado_interno
        )

        for hc_iter, hc_cost_at_iter in hist_hc_perturb:
            if hc_iter > 0:
                current_best_for_plot = min(mejor_costo_esta_grasp_iter, hc_cost_at_iter if es_solucion_factible_2pistas(sol_hc_perturb, numero_total_aviones) else float('inf'))
                grasp_run_cost_history.append((global_grasp_step_counter, current_best_for_plot))
                global_grasp_step_counter += 1
                if hc_cost_at_iter < mejor_costo_esta_grasp_iter and es_solucion_factible_2pistas(sol_hc_perturb, numero_total_aviones):
                    mejor_costo_esta_grasp_iter = hc_cost_at_iter # Actualizar el mejor si HC mejora

        if es_solucion_factible_2pistas(sol_hc_perturb, numero_total_aviones, verbose=verbose_hc_aplicado_resumen):
            if verbose_hc_aplicado_resumen: print(f"      ILS: Costo tras HC de perturbación: {costo_hc_perturb:.2f}")
            
            # Criterio de aceptación para ILS: si mejora la solución actual de ILS
            if costo_hc_perturb < current_costo_base_ils : # Compara con la base de esta iteración ILS
                 solucion_base_ils = copy.deepcopy(sol_hc_perturb)

            # Actualizar la mejor solución global de esta corrida GRASP
            if costo_hc_perturb < mejor_costo_esta_grasp_iter: # Ya actualizado arriba
                mejor_solucion_esta_grasp_iter = copy.deepcopy(sol_hc_perturb)
                # mejor_costo_esta_grasp_iter = costo_hc_perturb # Ya se hizo en el bucle de hist_hc_perturb
                solucion_base_ils = copy.deepcopy(mejor_solucion_esta_grasp_iter)
                if verbose_grasp_general: print(f"      ILS encontró nueva mejor solución para esta GRASP. Costo: {mejor_costo_esta_grasp_iter:.2f}")
        else:
            if verbose_grasp_general: print("      ILS: HC sobre solución perturbada resultó infactible. Reajustando base ILS.")
            solucion_base_ils = copy.deepcopy(mejor_solucion_esta_grasp_iter) 
        
        if not hist_hc_perturb or hist_hc_perturb[-1][0] == 0 :
            grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp_iter))

    if verbose_grasp_general: print(f"--- Fin Ejecución GRASP con ILS (2 Pistas) (Semilla: {semilla_construccion}). Mejor Costo: {mejor_costo_esta_grasp_iter:.2f} (Inicial: {costo_inicial_construido:.2f}) ---")
    
    if mejor_solucion_esta_grasp_iter is None or not es_solucion_factible_2pistas(mejor_solucion_esta_grasp_iter, numero_total_aviones, verbose=True):
        print(f"  ALERTA FINAL GRASP (2 Pistas) (Semilla {semilla_construccion}): Solución no factible o None. Costo Final: {mejor_costo_esta_grasp_iter}, Inicial: {costo_inicial_construido}")
        if not grasp_run_cost_history or grasp_run_cost_history[-1][0] < global_grasp_step_counter:
             grasp_run_cost_history.append((global_grasp_step_counter, float('inf')))
        return None, float('inf'), costo_inicial_construido, grasp_run_cost_history
    
    if not grasp_run_cost_history or grasp_run_cost_history[-1][0] < global_grasp_step_counter :
         grasp_run_cost_history.append((global_grasp_step_counter, mejor_costo_esta_grasp_iter))
    elif grasp_run_cost_history and grasp_run_cost_history[-1][1] > mejor_costo_esta_grasp_iter:
        grasp_run_cost_history[-1] = (grasp_run_cost_history[-1][0], mejor_costo_esta_grasp_iter)

    return mejor_solucion_esta_grasp_iter, mejor_costo_esta_grasp_iter, costo_inicial_construido, grasp_run_cost_history


# --- Función Principal del Runner GRASP con ILS ---
def main_grasp_ils_runner():
    ruta_archivo_datos = "case1.txt" 
    semillas_grasp_main = [42, 123, 7, 99, 500, 777, 2024, 1, 100, 314] 
    # semillas_grasp_main = [42, 123] # Para pruebas rápidas
    
    parametros_config_grasp = {
        "max_intentos_construccion_por_grasp": 1000, 
        "paso_delta_hc": 50,                   
        "max_iter_hc": 500, # Reducido para que los gráficos no sean demasiado largos por HC                   
        "num_restarts_ils": 5, # Reducido para ILS                 
        "num_switches_perturbacion": 3,        
    }
    
    config_verbose = {
        "grasp_general": True,              
        "greedy_constructor_detalle": False,
        "hc_aplicado_resumen": True,        
        "hc_detallado_interno": False,      
        "perturbacion_detalle": False       
    }

    print(f">>> Iniciando Proceso GRASP con ILS (2 Pistas) <<<")
    print(f"Archivo de datos: {ruta_archivo_datos}")
    print(f"Número de ejecuciones GRASP separadas: {len(semillas_grasp_main)}")
    print(f"Parámetros GRASP/ILS: {parametros_config_grasp}")

    try:
        aviones_base = leer_datos_aviones(ruta_archivo_datos)
        num_aviones_total = len(aviones_base)
        if num_aviones_total == 0:
            print("Error: No se leyeron aviones del archivo.")
            return
        print(f"Leídos {num_aviones_total} aviones.\n")
    except Exception as e:
        print(f"Error crítico leyendo datos: {e}")
        return

    mejor_solucion_global_obj: Optional[List[Avion]] = None
    costo_mejor_solucion_global: float = float('inf')
    costo_inicial_mejor_solucion_global: float = float('inf') 
    semilla_mejor_solucion_global_construccion: Optional[int] = None
    
    resultados_finales_por_semilla = [] 
    historias_de_costos_grasp_2pistas: List[Tuple[int, List[Tuple[int, float]]]] = [] # Para graficar

    for idx_s, semilla_actual_main in enumerate(semillas_grasp_main):
        print(f"\n========================= EJECUCIÓN GRASP GLOBAL (2 Pistas) #{idx_s + 1}/{len(semillas_grasp_main)} (Semilla Constr.: {semilla_actual_main}) =========================")
        
        # MODIFICADO: Recibir historial GRASP
        sol_grasp_iter, costo_final_grasp_iter, costo_inicial_grasp_iter, grasp_iter_history = ejecutar_grasp_una_vez_con_ils(
            semilla_actual_main,
            copy.deepcopy(aviones_base), 
            num_aviones_total,
            parametros_config_grasp,
            config_verbose["grasp_general"],
            config_verbose["greedy_constructor_detalle"],
            config_verbose["hc_aplicado_resumen"],
            config_verbose["hc_detallado_interno"],
            config_verbose["perturbacion_detalle"]
        )
        
        historias_de_costos_grasp_2pistas.append((semilla_actual_main, grasp_iter_history)) # Guardar historial

        factible_esta_iter_grasp = (sol_grasp_iter is not None) and \
                                   es_solucion_factible_2pistas(sol_grasp_iter, num_aviones_total, verbose=False)
        
        resultados_finales_por_semilla.append({
            "semilla_construccion": semilla_actual_main, 
            "costo_inicial_grasp": costo_inicial_grasp_iter, 
            "costo_final_iter_grasp": costo_final_grasp_iter if factible_esta_iter_grasp else float('inf'), 
            "solucion_obj_presente": sol_grasp_iter is not None, 
            "factible_final_iter_grasp": factible_esta_iter_grasp 
        })

        if factible_esta_iter_grasp and costo_final_grasp_iter < costo_mejor_solucion_global:
            costo_mejor_solucion_global = costo_final_grasp_iter
            mejor_solucion_global_obj = copy.deepcopy(sol_grasp_iter)
            semilla_mejor_solucion_global_construccion = semilla_actual_main
            costo_inicial_mejor_solucion_global = costo_inicial_grasp_iter 
            if config_verbose["grasp_general"]:
                print(f"¡NUEVA MEJOR SOLUCIÓN GLOBAL (2 Pistas) encontrada! Costo Inicial: {costo_inicial_grasp_iter:.2f}, Costo Final: {costo_mejor_solucion_global:.2f} (Semilla constr.: {semilla_actual_main})")
        elif not factible_esta_iter_grasp:
             if config_verbose["grasp_general"]: print(f"Ejecución GRASP (2 Pistas) con semilla {semilla_actual_main} no produjo solución final factible (Costo Inicial: {costo_inicial_grasp_iter:.2f}).")
        else: 
             if config_verbose["grasp_general"]: print(f"Ejecución GRASP (2 Pistas) con semilla {semilla_actual_main} finalizada. Costo Inicial: {costo_inicial_grasp_iter:.2f}, Costo Final: {costo_final_grasp_iter:.2f}. (No superó mejor global: {costo_mejor_solucion_global:.2f})")

    print("\n\n========================= RESULTADOS FINALES DEL PROCESO GRASP-ILS (2 Pistas) =========================")
    print("\nResumen de costos por semilla (Inicial -> Final):")
    for res in resultados_finales_por_semilla:
        costo_inicial_str = f"{res['costo_inicial_grasp']:.2f}" if res['costo_inicial_grasp'] != float('inf') else "Inf"
        costo_final_str = f"{res['costo_final_iter_grasp']:.2f}" if res['costo_final_iter_grasp'] != float('inf') else "Inf"
        print(f"  Semilla: {res['semilla_construccion']:<4} -> Costo Inicial: {costo_inicial_str}, Costo Final: {costo_final_str}")

    if mejor_solucion_global_obj:
        print(f"\nMejor solución global GRASP-ILS (2 Pistas) encontrada con costo inicial: {costo_inicial_mejor_solucion_global:.2f} y costo final: {costo_mejor_solucion_global:.2f}")
        if semilla_mejor_solucion_global_construccion is not None:
            print(f"  Originada por la semilla de construcción: {semilla_mejor_solucion_global_construccion}")
        
        print("\nHorario de la mejor solución global GRASP-ILS (2 Pistas):")
        imprimir_matriz_aterrizaje_local_con_cabecera(mejor_solucion_global_obj, f"Mejor Horario GRASP-ILS (2 Pistas) (Semilla Constr. {semilla_mejor_solucion_global_construccion})")
        
        if not es_solucion_factible_2pistas(mejor_solucion_global_obj, num_aviones_total, verbose=True): 
            print("\n¡¡¡ALERTA CRÍTICA!!! La mejor solución global GRASP-ILS (2 Pistas) reportada NO es factible.")
        else:
            print("\nLa mejor solución global GRASP-ILS (2 Pistas) reportada ES factible.")
    else:
        print("\nNo se encontró ninguna solución factible en ninguna de las ejecuciones GRASP-ILS (2 Pistas).")

    # --- AÑADIDO: Sección de Graficación ---
    plt.figure(figsize=(15, 8))
    plot_successful_2pistas = False
    
    for semilla, historia_costos_run in historias_de_costos_grasp_2pistas:
        if historia_costos_run and historia_costos_run[0][1] != float('inf') : 
            pasos = [item[0] for item in historia_costos_run]
            costos = [item[1] for item in historia_costos_run]
            if pasos: 
                 plt.plot(pasos, costos, marker='.', linestyle='-', alpha=0.7, label=f"Semilla {semilla}")
                 plot_successful_2pistas = True
        else:
            print(f"Nota: Historia de costos para GRASP (2 Pistas) con semilla '{semilla}' no se grafica (costo inicial inf o sin datos).")

    if plot_successful_2pistas:
        plt.xlabel("Paso de Evaluación en GRASP-ILS (2 Pistas)")
        plt.ylabel("Mejor Costo Encontrado (hasta el paso)")
        titulo_grafico = (f"Evolución del Costo en GRASP-ILS (2 Pistas) para '{ruta_archivo_datos}'\n"
                          f"({len(semillas_grasp_main)} Semillas)")
        plt.title(titulo_grafico)
        plt.legend(title="Ejecuciones GRASP (2 Pistas)", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.80, 1]) 

        nombre_grafico = f"grasp_ils_2pistas_costo_evolucion_{ruta_archivo_datos.split('.')[0]}.png"
        try:
            plt.savefig(nombre_grafico)
            print(f"\nGráfico de evolución de costos GRASP-ILS (2 Pistas) guardado como: {nombre_grafico}")
        except Exception as e_plot:
            print(f"Error al guardar el gráfico de GRASP-ILS (2 Pistas): {e_plot}")
        plt.show()
    else:
        print("\nNo se generó ningún gráfico de evolución de costos GRASP-ILS (2 Pistas) porque no hubo ejecuciones válidas para graficar.")

if __name__ == "__main__":
    main_grasp_ils_runner()
