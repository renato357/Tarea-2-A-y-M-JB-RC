import sys
import math
import copy # Para deepcopy, crucial para no modificar soluciones accidentalmente
from typing import List, Tuple, Optional, Set
import matplotlib.pyplot as plt # <--- AÑADIDO para graficar

# Constante para el número de pistas
NUMERO_DE_PISTAS = 2

# Intentar importar clases y funciones necesarias del script del greedy determinista para 2 pistas
try:
    # Asumimos que el archivo se llama 'greedydeterminista_2pistas.py'
    # y contiene la clase Avion y las funciones necesarias.
    from greedydeterminista_2pistas import (
        Avion,
        leer_datos_aviones,
        greedy_determinista_2pistas,
        # Ya no importamos la función de imprimir matriz de aquí para evitar conflictos de signatura
    )
except ImportError:
    print("Error: No se pudo importar de 'greedydeterminista_2pistas.py'.")
    print("Asegúrate de que dicho archivo exista en el mismo directorio y contenga:")
    print("  - class Avion (con pista_seleccionada)")
    print("  - def leer_datos_aviones(ruta_archivo: str) -> List[Avion]")
    print("  - def greedy_determinista_2pistas(lista_aviones: List[Avion]) -> Tuple[List[Avion], bool, List[Avion]]")
    sys.exit(1)

# --- Funciones de Cálculo de Costo y Factibilidad (Adaptadas para 2 Pistas) ---

def calcular_costo_individual_avion(avion: Avion, tiempo_aterrizaje_propuesto: int) -> float:
    """
    Calcula el costo de penalización si un avión aterriza en un tiempo específico.
    """
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
    """
    Calcula el costo total de una solución (lista de aviones con t_seleccionado).
    Suma una penalización alta por aviones no programados.
    """
    costo_total = 0.0
    num_aviones_no_programados = 0
    for avion in solucion:
        if avion.t_seleccionado is None or avion.pista_seleccionada is None:
            num_aviones_no_programados +=1
            # Penalización alta por cada avión no programado o sin pista
            costo_total += 1_000_000_000 
        else:
            costo_total += calcular_costo_individual_avion(avion, avion.t_seleccionado)
    
    if num_aviones_no_programados > 0:
        print(f"Advertencia en cálculo de costo: {num_aviones_no_programados} aviones no programados/asignados completamente.")
    return costo_total


def es_solucion_factible_2pistas(solucion: List[Avion], numero_total_aviones_problema: int, verbose: bool = False) -> bool:
    """
    Verifica si una solución dada es factible para el problema de 2 pistas.
    """
    if not solucion:
        if verbose: print("  [Factibilidad HC] Falla: La solución está vacía.")
        return False

    # 1. Verificar que todos los aviones estén programados con tiempo y pista
    aviones_completamente_programados = 0
    for avion_obj in solucion:
        if avion_obj.t_seleccionado is not None and avion_obj.pista_seleccionada is not None:
            if not (0 <= avion_obj.pista_seleccionada < NUMERO_DE_PISTAS):
                if verbose: print(f"  [Factibilidad HC] Falla: Avión ID {avion_obj.id_avion} con pista inválida {avion_obj.pista_seleccionada}.")
                return False
            aviones_completamente_programados += 1
        else: # Si falta tiempo o pista, no está completamente programado
            if verbose: print(f"  [Factibilidad HC] Falla: Avión ID {avion_obj.id_avion} no tiene tiempo ({avion_obj.t_seleccionado}) o pista ({avion_obj.pista_seleccionada}) asignada.")
            return False # Para HC, todos deben estar asignados.

    if aviones_completamente_programados != numero_total_aviones_problema:
        if verbose: print(f"  [Factibilidad HC] Falla: No todos los {numero_total_aviones_problema} aviones están completamente programados. Programados: {aviones_completamente_programados}")
        return False

    # 2. Verificar ventanas de tiempo E_k, L_k para los programados
    for avion in solucion: # Ya sabemos que todos tienen t_seleccionado y pista_seleccionada
        # Se usa type: ignore porque mypy puede no inferir que t_seleccionado no es None aquí
        if not (avion.tiempo_aterrizaje_temprano <= avion.t_seleccionado <= avion.tiempo_aterrizaje_tardio): # type: ignore
            if verbose: print(f"  [Factibilidad HC] Falla: Avión ID {avion.id_avion} t={avion.t_seleccionado} fuera de ventana [{avion.tiempo_aterrizaje_temprano}, {avion.tiempo_aterrizaje_tardio}].")
            return False

    # 3. Verificar separaciones en cada pista
    for pista_idx_verif in range(NUMERO_DE_PISTAS):
        aviones_en_esta_pista = sorted(
            [a for a in solucion if a.pista_seleccionada == pista_idx_verif], 
            key=lambda x: x.t_seleccionado # type: ignore
        )
        for i in range(len(aviones_en_esta_pista) - 1):
            avion_i = aviones_en_esta_pista[i]
            avion_j = aviones_en_esta_pista[i+1]
            
            separacion_requerida_S_ij = avion_i.get_separation_to(avion_j)
            # Se usa type: ignore por la misma razón que antes
            separacion_real = avion_j.t_seleccionado - avion_i.t_seleccionado # type: ignore
            
            if separacion_real < separacion_requerida_S_ij:
                if verbose: print(f"  [Factibilidad HC] Falla: Separación en Pista {pista_idx_verif} entre avión ID {avion_i.id_avion} (t={avion_i.t_seleccionado}) y "
                               f"avión ID {avion_j.id_avion} (t={avion_j.t_seleccionado}). "
                               f"Real: {separacion_real}, Requerida: {separacion_requerida_S_ij}.")
                return False
    return True

def _calcular_tiempo_min_aterrizaje_en_pista_hc(
    avion_k: Avion, 
    pista_idx: int, 
    otros_aviones_en_pista: List[Avion], 
    tiempo_referencia_aterrizaje: Optional[int] = None
    ) -> int:
    """
    Calcula el tiempo más temprano que avion_k puede aterrizar en pista_idx,
    considerando su E_k y la separación con los otros_aviones_en_pista.
    Usado por Hill Climbing para evaluar movimientos.
    """
    t_min_calculado = avion_k.tiempo_aterrizaje_temprano
    if tiempo_referencia_aterrizaje is not None:
        t_min_calculado = max(t_min_calculado, tiempo_referencia_aterrizaje)

    for avion_ya_en_pista in otros_aviones_en_pista: 
        if avion_ya_en_pista.t_seleccionado is None: continue # No debería pasar si la solución es factible
        # avion_k debe aterrizar después de avion_ya_en_pista
        t_min_calculado = max(t_min_calculado, avion_ya_en_pista.t_seleccionado + avion_ya_en_pista.get_separation_to(avion_k))
        
    return t_min_calculado

# NUEVA FUNCION DE IMPRESION LOCAL
def imprimir_matriz_aterrizaje_local_con_cabecera(lista_aviones_procesada: List[Avion], cabecera: str) -> None:
    """
    Imprime una matriz con el horario de aterrizaje de los aviones programados,
    incluyendo la pista asignada y una cabecera personalizada. Se ordena por tiempo de aterrizaje.
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

    # Ordenar por tiempo de aterrizaje, luego por pista, luego por ID para desempate visual
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


# --- Hill Climbing (Alguna-Mejora Determinista para 2 Pistas) ---
def hill_climbing_alguna_mejora_2pistas(
    solucion_inicial: List[Avion],
    numero_total_aviones_problema: int,
    paso_tiempo_delta: int = 10,
    max_iter_hc: int = 100,
    verbose_hc: bool = False
    ) -> Tuple[List[Avion], List[Tuple[int, float]]]: # <--- MODIFICADO: Devuelve historial de costos
    """
    Aplica Hill Climbing con estrategia "alguna-mejora" (first-improvement) de forma determinista.
    Operadores de movimiento:
    1. Cambiar el tiempo de aterrizaje de un solo avión en su pista actual.
    2. Cambiar un avión a la otra pista (ajustando su tiempo).
    Devuelve la mejor solución encontrada y el historial de costos por iteración.
    """
    solucion_actual = copy.deepcopy(solucion_inicial)
    
    if not es_solucion_factible_2pistas(solucion_actual, numero_total_aviones_problema, verbose=True):
        print("Error HC: La solución inicial proporcionada al Hill Climbing no es factible. Abortando HC.")
        costo_infactible = calcular_costo_total_solucion(solucion_inicial)
        return solucion_inicial, [(0, costo_infactible)]

    costo_actual = calcular_costo_total_solucion(solucion_actual)
    
    # --- AÑADIDO: Inicializar historial de costos ---
    cost_history: List[Tuple[int, float]] = []
    cost_history.append((0, costo_actual)) # Iteración 0: costo de la solución inicial

    if verbose_hc:
        print("\n--- Iniciando Hill Climbing (Alguna-Mejora Determinista para 2 Pistas) ---")
        print(f"Costo Inicial (del Greedy): {costo_actual:.2f}")

    iteraciones_hc_global = 0
    mejora_global_realizada = False # Para saber si HC hizo algún cambio real

    while iteraciones_hc_global < max_iter_hc:
        iteraciones_hc_global += 1
        mejora_encontrada_en_esta_iteracion_local = False # Renombrado para claridad
        
        if verbose_hc: 
            print(f"\n--- Iteración Global de Hill Climbing #{iteraciones_hc_global} (Costo actual: {costo_actual:.2f}) ---")
        
        # Ordenar aviones por ID para un procesamiento determinista consistente
        aviones_ordenados_hc = sorted(solucion_actual, key=lambda a: a.id_avion)

        for avion_idx_hc in range(len(aviones_ordenados_hc)):
            # Obtener la referencia al objeto avión DENTRO de solucion_actual
            # avion_a_modificar_original_ref es solo para obtener el ID de forma ordenada
            avion_original_id = aviones_ordenados_hc[avion_idx_hc].id_avion
            avion_a_modificar_en_sol_actual = next(a for a in solucion_actual if a.id_avion == avion_original_id)

            t_original_avion = avion_a_modificar_en_sol_actual.t_seleccionado
            p_original_avion = avion_a_modificar_en_sol_actual.pista_seleccionada

            if t_original_avion is None or p_original_avion is None: 
                # Esto no debería ocurrir si la solución inicial es factible
                if verbose_hc: print(f"Advertencia HC: Avión ID {avion_original_id} con datos incompletos.")
                continue

            # --- Movimiento 1: Cambiar tiempo en la pista actual ---
            tiempos_a_probar_misma_pista: List[int] = []
            # Añadir tiempos clave: preferente, temprano, tardío
            for t_clave in [avion_a_modificar_en_sol_actual.tiempo_aterrizaje_preferente,
                            avion_a_modificar_en_sol_actual.tiempo_aterrizaje_temprano,
                            avion_a_modificar_en_sol_actual.tiempo_aterrizaje_tardio]:
                if t_clave != t_original_avion and \
                   avion_a_modificar_en_sol_actual.tiempo_aterrizaje_temprano <= t_clave <= avion_a_modificar_en_sol_actual.tiempo_aterrizaje_tardio:
                    if t_clave not in tiempos_a_probar_misma_pista: tiempos_a_probar_misma_pista.append(t_clave)
            
            # Añadir tiempos alrededor del actual usando paso_tiempo_delta
            for k_delta in range(1, 3): # Explorar +/- 1*delta y +/- 2*delta
                for sign in [-1, 1]:
                    t_delta = t_original_avion + sign * k_delta * paso_tiempo_delta
                    if t_delta != t_original_avion and \
                       avion_a_modificar_en_sol_actual.tiempo_aterrizaje_temprano <= t_delta <= avion_a_modificar_en_sol_actual.tiempo_aterrizaje_tardio:
                        if t_delta not in tiempos_a_probar_misma_pista: tiempos_a_probar_misma_pista.append(t_delta)
            
            tiempos_a_probar_misma_pista.sort() # Procesar en orden determinista

            for nuevo_t_propuesto in tiempos_a_probar_misma_pista:
                vecino_potencial = copy.deepcopy(solucion_actual)
                avion_modificado_en_vecino = next(a for a in vecino_potencial if a.id_avion == avion_original_id)
                
                avion_modificado_en_vecino.t_seleccionado = nuevo_t_propuesto
                # La pista no cambia en este movimiento

                if es_solucion_factible_2pistas(vecino_potencial, numero_total_aviones_problema, verbose=False):
                    costo_vecino_potencial = calcular_costo_total_solucion(vecino_potencial)
                    if costo_vecino_potencial < costo_actual:
                        if verbose_hc: print(f"  HC Mejora (Misma Pista): Avión ID {avion_modificado_en_vecino.id_avion} de t={t_original_avion} a t={nuevo_t_propuesto} en Pista {p_original_avion}. Costo: {costo_actual:.2f} -> {costo_vecino_potencial:.2f}")
                        solucion_actual = vecino_potencial 
                        costo_actual = costo_vecino_potencial
                        mejora_encontrada_en_esta_iteracion_local = True
                        mejora_global_realizada = True
                        break 
            
            if mejora_encontrada_en_esta_iteracion_local: # Si se encontró mejora con Movimiento 1
                break # Salir del bucle de aviones y pasar a la siguiente iteración global de HC

            # --- Movimiento 2: Cambiar a la otra pista ---
            otra_pista = 1 - p_original_avion # Asumiendo pistas 0 y 1
            
            # Obtener aviones que ya están en la 'otra_pista' EXCLUYENDO el que estamos moviendo
            aviones_fijos_en_otra_pista = [
                a for a in solucion_actual 
                if a.pista_seleccionada == otra_pista and a.id_avion != avion_original_id
            ]
            
            tiempos_a_probar_otra_pista: List[int] = []
            
            # Opción A: Tiempo más temprano posible en la otra pista
            # Usamos el objeto original para calcular esto, ya que sus E_k, L_k, etc., no han cambiado
            t_mas_temprano_valido_otra_pista = _calcular_tiempo_min_aterrizaje_en_pista_hc(
                avion_a_modificar_en_sol_actual, # El avión original con sus propiedades
                otra_pista, 
                aviones_fijos_en_otra_pista
            )

            if t_mas_temprano_valido_otra_pista <= avion_a_modificar_en_sol_actual.tiempo_aterrizaje_tardio:
                if t_mas_temprano_valido_otra_pista not in tiempos_a_probar_otra_pista:
                    tiempos_a_probar_otra_pista.append(t_mas_temprano_valido_otra_pista)

            # Opción B: Tiempo preferente del avión (si es factible en la otra pista)
            pk_avion = avion_a_modificar_en_sol_actual.tiempo_aterrizaje_preferente
            # Verificar si pk_avion es al menos t_mas_temprano_valido_otra_pista y dentro de L_k
            if pk_avion >= t_mas_temprano_valido_otra_pista and \
               pk_avion <= avion_a_modificar_en_sol_actual.tiempo_aterrizaje_tardio:
                if pk_avion not in tiempos_a_probar_otra_pista:
                    tiempos_a_probar_otra_pista.append(pk_avion)
            
            tiempos_a_probar_otra_pista.sort() # Procesar en orden determinista

            for nuevo_t_otra_pista in tiempos_a_probar_otra_pista:
                vecino_potencial = copy.deepcopy(solucion_actual)
                avion_modificado_en_vecino = next(a for a in vecino_potencial if a.id_avion == avion_original_id)

                avion_modificado_en_vecino.t_seleccionado = nuevo_t_otra_pista
                avion_modificado_en_vecino.pista_seleccionada = otra_pista
                
                if es_solucion_factible_2pistas(vecino_potencial, numero_total_aviones_problema, verbose=False):
                    costo_vecino_potencial = calcular_costo_total_solucion(vecino_potencial)
                    if costo_vecino_potencial < costo_actual:
                        if verbose_hc: print(f"  HC Mejora (Cambio Pista): Avión ID {avion_modificado_en_vecino.id_avion} de (t={t_original_avion},p={p_original_avion}) a (t={nuevo_t_otra_pista},p={otra_pista}). Costo: {costo_actual:.2f} -> {costo_vecino_potencial:.2f}")
                        solucion_actual = vecino_potencial 
                        costo_actual = costo_vecino_potencial
                        mejora_encontrada_en_esta_iteracion_local = True
                        mejora_global_realizada = True
                        break 
            
            if mejora_encontrada_en_esta_iteracion_local: # Si se encontró mejora con Movimiento 2
                break # Salir del bucle de aviones y pasar a la siguiente iteración global de HC
        
        # --- AÑADIDO: Registrar costo al final de la iteración global de HC ---
        cost_history.append((iteraciones_hc_global, costo_actual))

        if not mejora_encontrada_en_esta_iteracion_local:
            if verbose_hc: print("  No se encontró ninguna mejora en esta iteración global del vecindario. Óptimo local alcanzado.")
            break # Terminar Hill Climbing si no hay mejora en una iteración completa
    
    # Mensajes de finalización
    if iteraciones_hc_global >= max_iter_hc and mejora_encontrada_en_esta_iteracion_local : # Si alcanzó max_iter pero la última iteración encontró mejora
        if verbose_hc: print(f"\nLímite de iteraciones de Hill Climbing ({max_iter_hc}) alcanzado, pero la última iteración aún encontraba mejora.")
    elif iteraciones_hc_global >= max_iter_hc: # Si alcanzó max_iter y no hubo mejora en la última
         if verbose_hc: print(f"\nLímite de iteraciones de Hill Climbing ({max_iter_hc}) alcanzado.")
    elif not mejora_global_realizada: # Si no hizo ni una sola mejora desde el inicio
        if verbose_hc: print("\nHill Climbing no encontró ninguna mejora sobre la solución inicial.")
    else: # Convergió antes de max_iter
        if verbose_hc: print(f"\nHill Climbing convergió en {iteraciones_hc_global} iteraciones.")


    if verbose_hc: print("\n--- Hill Climbing (Alguna-Mejora Determinista para 2 Pistas) Finalizado ---")
    return solucion_actual, cost_history # <--- MODIFICADO: Devolver historial

# --- Función Principal del Runner ---
def main_runner():
    """
    Función principal para ejecutar el Greedy Determinista para 2 Pistas
    y luego Hill Climbing (Alguna-Mejora Determinista para 2 Pistas).
    También genera un gráfico de convergencia para Hill Climbing.
    """
    ruta_datos_iniciales = "case4.txt" 
    paso_delta_hc_param = 50          
    max_iteraciones_hc_param = 500    
    verbose_mode_hc = True            

    print(">>> Paso 1: Ejecutando Greedy Determinista para 2 Pistas para obtener solución inicial...")
    try:
        lista_aviones_base = leer_datos_aviones(ruta_datos_iniciales)
        if not lista_aviones_base:
            print(f"Error: No se pudieron leer aviones desde '{ruta_datos_iniciales}'.")
            return
        numero_aviones = len(lista_aviones_base)
        print(f"Leídos {numero_aviones} aviones desde '{ruta_datos_iniciales}'.")

        solucion_greedy_lista_copia = copy.deepcopy(lista_aviones_base) 
        
        print("\n--- Salida del Greedy Determinista para 2 Pistas (esperada) ---")
        # El greedy para 2 pistas devuelve: solucion_greedy, factible, aviones_no_programados_greedy
        solucion_obtenida_greedy, factible_greedy, _ = greedy_determinista_2pistas(solucion_greedy_lista_copia)
        print("--- Fin Salida del Greedy Determinista para 2 Pistas ---\n")

        if not factible_greedy:
            print("Error: La solución inicial del Greedy Determinista para 2 Pistas NO es factible (según el propio greedy).")
            print("Hill Climbing no se ejecutará.")
            imprimir_matriz_aterrizaje_local_con_cabecera(solucion_obtenida_greedy, "Matriz Greedy (Infactible por Greedy)")
            costo_greedy_calc = calcular_costo_total_solucion(solucion_obtenida_greedy)
            print(f"Costo (infactible) de la solución Greedy: {costo_greedy_calc:.2f}")
            return

        # Verificación de factibilidad con la función que usará HC
        if not es_solucion_factible_2pistas(solucion_obtenida_greedy, numero_aviones, verbose=True):
            print("Error: La solución del Greedy (2 Pistas), aunque marcada como factible por su script,")
            print("NO pasó la validación de factibilidad de este script de Hill Climbing (es_solucion_factible_2pistas).")
            imprimir_matriz_aterrizaje_local_con_cabecera(solucion_obtenida_greedy, "Matriz Greedy (Infactible según HC)")
            costo_greedy_calc = calcular_costo_total_solucion(solucion_obtenida_greedy)
            print(f"Costo (infactible según este script) de la solución Greedy: {costo_greedy_calc:.2f}")
            return

        costo_greedy = calcular_costo_total_solucion(solucion_obtenida_greedy)
        print(f"\nSolución Inicial del Greedy Determinista para 2 Pistas (Factible):")
        print(f"Costo del Greedy: {costo_greedy:.2f}")
        imprimir_matriz_aterrizaje_local_con_cabecera(solucion_obtenida_greedy, "Matriz Greedy Inicial")

        print("\n>>> Paso 2: Aplicando Hill Climbing (Alguna-Mejora Determinista para 2 Pistas)...")
        # --- MODIFICADO: Recibir historial de costos ---
        solucion_final_hc, hc_cost_history = hill_climbing_alguna_mejora_2pistas( 
            solucion_obtenida_greedy,
            numero_aviones,
            paso_tiempo_delta=paso_delta_hc_param,
            max_iter_hc=max_iteraciones_hc_param,
            verbose_hc=verbose_mode_hc
        )

        costo_final_hc = calcular_costo_total_solucion(solucion_final_hc)
        print("\n\n==================================================================================")
        print("--- Resultados Finales (Greedy Determinista 2 Pistas + Hill Climbing Alguna Mejora) ---")
        print("==================================================================================")
        print(f"Costo Inicial (Greedy Determinista 2 Pistas): {costo_greedy:.2f}")
        print(f"Costo Final (Después de Hill Climbing): {costo_final_hc:.2f}")
        
        if costo_final_hc < costo_greedy:
            mejora = costo_greedy - costo_final_hc
            porcentaje_mejora = (mejora / costo_greedy * 100) if costo_greedy > 0 else 0
            print(f"Mejora obtenida por Hill Climbing: {mejora:.2f} ({porcentaje_mejora:.2f}%)")
        elif costo_final_hc == costo_greedy:
            print("Hill Climbing no encontró una mejora respecto a la solución Greedy.")
        else:
            print("Advertencia: El costo después de Hill Climbing es MAYOR. Revise la lógica si esto es inesperado.")

        print("\nMatriz de Aterrizaje Final (después de Hill Climbing):")
        imprimir_matriz_aterrizaje_local_con_cabecera(solucion_final_hc, "Matriz Final HC")
        
        if not es_solucion_factible_2pistas(solucion_final_hc, numero_aviones, verbose=True):
            print("\n¡¡¡ADVERTENCIA!!! La solución final del Hill Climbing NO es factible según la verificación.")
        else:
            print("\nLa solución final del Hill Climbing es factible.")

        # --- AÑADIDO: Sección de Graficación ---
        if hc_cost_history:
            iteraciones_plot = [item[0] for item in hc_cost_history]
            costos_plot = [item[1] for item in hc_cost_history]

            plt.figure(figsize=(12, 7))
            plt.plot(iteraciones_plot, costos_plot, marker='o', linestyle='-', color='g', label="Costo en Búsqueda Local (HC 2 Pistas)")
            
            plt.xlabel("Iteración de Búsqueda Local (Hill Climbing)")
            plt.ylabel("Costo Total de la Solución")
            # Título adaptado para 2 pistas
            titulo_grafico = (f"GRASP (2 Pistas): Convergencia de Búsqueda Local (HC) para '{ruta_datos_iniciales}'\n"
                              f"Costo Construcción (Greedy 2 Pistas): {costo_greedy:.2f} -> Costo Final (BL): {costo_final_hc:.2f}")
            plt.title(titulo_grafico)
            plt.legend()
            plt.grid(True)
            plt.tight_layout() 

            nombre_grafico = f"grasp_hc_2pistas_costo_evolucion_{ruta_datos_iniciales.split('.')[0]}.png"
            try:
                plt.savefig(nombre_grafico)
                print(f"\nGráfico de convergencia guardado como: {nombre_grafico}")
            except Exception as e_plot:
                print(f"Error al guardar el gráfico: {e_plot}")
            
            plt.show()
        else:
            print("\nNo se generó historial de costos para Hill Climbing, no se puede graficar.")

    except FileNotFoundError:
        print(f"Error CRÍTICO: No se pudo encontrar el archivo de datos '{ruta_datos_iniciales}'.", file=sys.stderr)
    except ImportError as ie:
         print(f"Error CRÍTICO de importación: {ie}. Asegúrate que los archivos necesarios estén presentes.", file=sys.stderr)
    except Exception as e:
        print(f"Ocurrió un error inesperado en main_runner: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_runner()
