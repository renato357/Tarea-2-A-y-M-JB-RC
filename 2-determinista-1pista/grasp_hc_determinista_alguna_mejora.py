import sys
import math
import copy # Para deepcopy, crucial para no modificar soluciones accidentalmente
from typing import List, Tuple, Optional, Set
import matplotlib.pyplot as plt # <--- AÑADIDO para graficar

# Importar clases y funciones necesarias desde el script del greedy determinista
try:
    from greedydeterminista import Avion, leer_datos_aviones, greedy_priorizar_menor_Ek, imprimir_matriz_aterrizaje
except ImportError:
    print("Error: No se pudo importar 'greedydeterminista.py'. Asegúrate de que esté en la misma carpeta.")
    sys.exit(1)

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
    """
    costo_total = 0.0
    num_aviones_no_programados = 0
    for avion in solucion:
        if avion.t_seleccionado is None:
            num_aviones_no_programados +=1
            # Penalización alta por aviones no programados para que la factibilidad lo detecte
            # y para que el costo refleje un estado muy indeseable.
            costo_total += 1_000_000_000 
        else:
            costo_total += calcular_costo_individual_avion(avion, avion.t_seleccionado)
    
    if num_aviones_no_programados > 0:
        # Este print es más una advertencia durante el desarrollo/debug.
        # En producción, la función es_solucion_factible debería prevenir esto.
        print(f"Advertencia en cálculo de costo: {num_aviones_no_programados} aviones no programados.")
    return costo_total


def es_solucion_factible(solucion: List[Avion], numero_total_aviones_problema: int, verbose: bool = False) -> bool:
    """
    Verifica si una solución dada es factible.
    Una solución es factible si:
    1. Todos los aviones del problema están programados (tienen un t_seleccionado).
    2. Cada avión aterriza dentro de su ventana de tiempo [E_k, L_k].
    3. Se respetan los tiempos de separación entre todos los pares de aviones.
    """
    if not solucion:
        if verbose: print("  [Factibilidad] Falla: La solución está vacía.")
        return False

    # Verificar que todos los aviones tengan un tiempo asignado
    aviones_programados_count = 0
    for avion_check in solucion:
        if avion_check.t_seleccionado is None:
            if verbose: print(f"  [Factibilidad] Falla: Avión ID {avion_check.id_avion} no tiene t_seleccionado.")
            return False
        aviones_programados_count +=1
    
    if aviones_programados_count != numero_total_aviones_problema:
        if verbose: print(f"  [Factibilidad] Falla: Número de aviones programados ({aviones_programados_count}) "
                           f"no coincide con el total esperado ({numero_total_aviones_problema}).")
        return False

    # Ordenar por tiempo de aterrizaje para facilitar las comprobaciones
    try:
        # Asegurarse de que todos los t_seleccionado son números antes de ordenar
        for a_sort_check in solucion:
            if not isinstance(a_sort_check.t_seleccionado, (int, float)):
                if verbose: print(f"  [Factibilidad] Falla: Avión ID {a_sort_check.id_avion} tiene t_seleccionado no numérico: {a_sort_check.t_seleccionado}.")
                return False
        secuencia_ordenada = sorted(solucion, key=lambda a: a.t_seleccionado)
    except TypeError as e:
        if verbose: print(f"  [Factibilidad] Error al ordenar aviones: {e}. Verifique tipos de t_seleccionado.")
        return False # Error al ordenar, probablemente por t_seleccionado None que no debería estar aquí

    # Comprobar ventanas de tiempo y separaciones
    for i in range(len(secuencia_ordenada)):
        avion_i = secuencia_ordenada[i]

        # Comprobar ventana de tiempo para avion_i
        if not (avion_i.tiempo_aterrizaje_temprano <= avion_i.t_seleccionado <= avion_i.tiempo_aterrizaje_tardio):
            if verbose: print(f"  [Factibilidad] Falla: Avión ID {avion_i.id_avion} (t={avion_i.t_seleccionado}) "
                               f"fuera de ventana [{avion_i.tiempo_aterrizaje_temprano}, {avion_i.tiempo_aterrizaje_tardio}].")
            return False

        # Comprobar separación con aviones posteriores en la secuencia
        for j in range(i + 1, len(secuencia_ordenada)):
            avion_j = secuencia_ordenada[j]
            
            separacion_requerida = avion_i.get_separation_to(avion_j) # S_ij
            
            # La condición es t_j >= t_i + S_ij
            if avion_j.t_seleccionado < avion_i.t_seleccionado + separacion_requerida:
                if verbose: print(f"  [Factibilidad] Falla: Separación entre avión ID {avion_i.id_avion} (t={avion_i.t_seleccionado}, tipo {avion_i.tipo_avion}) y "
                                   f"avión ID {avion_j.id_avion} (t={avion_j.t_seleccionado}, tipo {avion_j.tipo_avion}). "
                                   f"Real: {avion_j.t_seleccionado - avion_i.t_seleccionado}, Requerida: {separacion_requerida}.")
                return False
            
    return True

def hill_climbing_alguna_mejora_determinista_v2(
    solucion_inicial: List[Avion],
    numero_total_aviones_problema: int,
    paso_tiempo_delta: int = 10,
    max_iter_hc: int = 100
    ) -> Tuple[List[Avion], List[Tuple[int, float]]]: # <--- MODIFICADO: Devuelve también el historial de costos
    """
    Aplica Hill Climbing con estrategia "alguna-mejora" (first-improvement) de forma determinista,
    con un orden de prueba de tiempos específico.
    Operador de movimiento: Cambiar el tiempo de aterrizaje de un solo avión.
    Devuelve la mejor solución encontrada y el historial de costos por iteración.
    """
    solucion_actual = copy.deepcopy(solucion_inicial)
    
    if not es_solucion_factible(solucion_actual, numero_total_aviones_problema, verbose=True):
        print("Error: La solución inicial proporcionada al Hill Climbing no es factible. Abortando HC.")
        # Devuelve la solución inicial y un historial vacío o con el costo infactible
        costo_infactible = calcular_costo_total_solucion(solucion_inicial)
        return solucion_inicial, [(0, costo_infactible)]

    costo_actual = calcular_costo_total_solucion(solucion_actual)
    
    # --- AÑADIDO: Inicializar historial de costos ---
    cost_history: List[Tuple[int, float]] = []
    cost_history.append((0, costo_actual)) # Iteración 0: costo de la solución inicial

    print("\n--- Iniciando Hill Climbing (Alguna-Mejora Determinista V2) ---")
    print(f"Costo Inicial (del Greedy): {costo_actual:.2f}")
    # imprimir_matriz_aterrizaje(solucion_actual) # Opcional: imprimir la matriz inicial

    iteraciones_hc = 0
    mejora_global_encontrada = False # Para saber si HC hizo algún cambio

    while iteraciones_hc < max_iter_hc:
        iteraciones_hc += 1
        # print(f"\n--- Iteración de Hill Climbing #{iteraciones_hc} ---") # Puede ser muy verboso
        mejora_encontrada_en_esta_iteracion_local = False # Renombrado para claridad
        
        # Iterar sobre cada avión en la solución actual para intentar modificarlo
        # Se itera sobre una copia de los índices para evitar problemas si la lista se modifica (aunque aquí no debería)
        for i in range(len(solucion_actual)): 
            avion_a_modificar_original = solucion_actual[i] # Referencia al objeto en solucion_actual
            
            if avion_a_modificar_original.t_seleccionado is None: # No debería ocurrir si la inicial es factible
                print(f"Advertencia: Avión {avion_a_modificar_original.id_avion} con t_seleccionado None en HC iter {iteraciones_hc}")
                continue

            t_original_avion = avion_a_modificar_original.t_seleccionado
            
            # Construir lista de tiempos a probar en un orden específico y determinista
            tiempos_candidatos_ordenados: List[int] = []
            tiempos_ya_anadidos: Set[int] = set()

            def anadir_si_valido_y_nuevo(tiempo: int):
                if tiempo not in tiempos_ya_anadidos and \
                   avion_a_modificar_original.tiempo_aterrizaje_temprano <= tiempo <= avion_a_modificar_original.tiempo_aterrizaje_tardio:
                    tiempos_candidatos_ordenados.append(tiempo)
                    tiempos_ya_anadidos.add(tiempo)

            # 1. Tiempo preferente
            anadir_si_valido_y_nuevo(avion_a_modificar_original.tiempo_aterrizaje_preferente)

            # 2. Deltas pequeños alrededor del original
            # Se añaden primero los más cercanos al original
            for k in range(1, 5): # Explora deltas hasta +/- 4*paso_tiempo_delta/2
                delta_actual = (paso_tiempo_delta * k) // 2 # Usar // para asegurar entero
                if delta_actual == 0 and k > 1 : continue # Evitar delta cero si paso_tiempo_delta es 1

                anadir_si_valido_y_nuevo(t_original_avion - delta_actual)
                if delta_actual != 0: # No añadir dos veces si delta_actual es 0
                    anadir_si_valido_y_nuevo(t_original_avion + delta_actual)
            
            # 3. Extremos de la ventana (si no han sido añadidos ya o son muy lejanos)
            anadir_si_valido_y_nuevo(avion_a_modificar_original.tiempo_aterrizaje_temprano)
            anadir_si_valido_y_nuevo(avion_a_modificar_original.tiempo_aterrizaje_tardio)
            
            # Probar cada tiempo candidato para el avión actual
            for nuevo_tiempo_propuesto in tiempos_candidatos_ordenados:
                if nuevo_tiempo_propuesto == t_original_avion: 
                    continue # No probar el mismo tiempo

                # Crear una copia profunda del estado actual para generar el vecino
                vecino_potencial = copy.deepcopy(solucion_actual)
                
                # Encontrar el avión correspondiente en la copia y modificar su tiempo
                avion_modificado_en_vecino = next((a for a in vecino_potencial if a.id_avion == avion_a_modificar_original.id_avion), None)
                
                if avion_modificado_en_vecino:
                    avion_modificado_en_vecino.t_seleccionado = nuevo_tiempo_propuesto
                else:
                    # Esto no debería ocurrir si la lógica de copia es correcta
                    print(f"Error crítico en HC: No se encontró el avión ID {avion_a_modificar_original.id_avion} en la copia del vecino.")
                    continue 

                # Verificar factibilidad del vecino
                if es_solucion_factible(vecino_potencial, numero_total_aviones_problema, verbose=False): # verbose=False para no inundar la consola
                    costo_vecino_potencial = calcular_costo_total_solucion(vecino_potencial)
                    
                    # Criterio de "alguna mejora" (first improvement)
                    if costo_vecino_potencial < costo_actual: 
                        costo_anterior = costo_actual
                        solucion_actual = copy.deepcopy(vecino_potencial) # Aceptar la mejora
                        costo_actual = costo_vecino_potencial
                        mejora_encontrada_en_esta_iteracion_local = True
                        mejora_global_encontrada = True # Marcar que HC ha hecho al menos una mejora
                        
                        movimiento_realizado_info = (
                            f"Movimiento HC: Avión ID {avion_a_modificar_original.id_avion} "
                            f"de t={t_original_avion} a t={nuevo_tiempo_propuesto}"
                        )
                        print(f"  HC Iter {iteraciones_hc}: ¡Mejora encontrada! Costo: {costo_anterior:.2f} -> {costo_actual:.2f}. {movimiento_realizado_info}")
                        # imprimir_matriz_aterrizaje(solucion_actual) # Opcional: imprimir en cada mejora
                        break # Salir del bucle de tiempos_candidatos (para este avión)
            
            if mejora_encontrada_en_esta_iteracion_local:
                break # Salir del bucle de aviones (para esta iteración_hc)
        
        # --- AÑADIDO: Registrar costo al final de la iteración de HC ---
        cost_history.append((iteraciones_hc, costo_actual))

        if not mejora_encontrada_en_esta_iteracion_local:
            # print("  No se encontró ninguna mejora en esta iteración completa del vecindario. Óptimo local alcanzado.")
            break # Terminar Hill Climbing si no hay mejora en una iteración completa
    
    if iteraciones_hc >= max_iter_hc:
        print(f"\nLímite de iteraciones de Hill Climbing ({max_iter_hc}) alcanzado.")
    elif not mejora_global_encontrada:
         print("\nHill Climbing no encontró ninguna mejora sobre la solución inicial.")
    else:
        print(f"\nHill Climbing convergió en {iteraciones_hc} iteraciones.")


    print("--- Hill Climbing (Alguna-Mejora Determinista V2) Finalizado ---")
    return solucion_actual, cost_history # <--- MODIFICADO: Devolver historial

def main_runner():
    """
    Función principal para ejecutar el Greedy Determinista y luego Hill Climbing.
    También genera un gráfico de convergencia para Hill Climbing.
    """
    ruta_datos_iniciales = "case1.txt" 
    paso_delta_hc = 50 
    max_iteraciones_hc = 500 # Aumentado para permitir más exploración si es necesario

    print(">>> Paso 1: Ejecutando Greedy Determinista para obtener solución inicial...")
    try:
        lista_aviones_base = leer_datos_aviones(ruta_datos_iniciales)
        if not lista_aviones_base:
            print(f"Error: No se pudieron leer aviones desde '{ruta_datos_iniciales}'.")
            return
        numero_aviones = len(lista_aviones_base)
        print(f"Leídos {numero_aviones} aviones desde '{ruta_datos_iniciales}'.")

        solucion_greedy_lista = copy.deepcopy(lista_aviones_base) 
        
        print("\n--- Salida del Greedy Determinista (esperada) ---")
        solucion_obtenida_greedy, factible_greedy, _ = greedy_priorizar_menor_Ek(solucion_greedy_lista)
        print("--- Fin Salida del Greedy Determinista ---\n")

        if not factible_greedy:
            print("Error: La solución inicial del Greedy Determinista NO es factible según el propio greedy.")
            # ... (resto del manejo de error si es necesario, por ahora solo retorna)
            imprimir_matriz_aterrizaje(solucion_obtenida_greedy) # Mostrarla aunque sea infactible
            costo_greedy_infactible = calcular_costo_total_solucion(solucion_obtenida_greedy)
            print(f"Costo (infactible) de la solución Greedy: {costo_greedy_infactible:.2f}")
            return

        # Es crucial verificar la factibilidad con la función que usará HC
        if not es_solucion_factible(solucion_obtenida_greedy, numero_aviones, verbose=True):
            print("Error: La solución del Greedy, aunque marcada como factible por el script original,")
            print("NO pasó la validación de factibilidad de este script (es_solucion_factible).")
            imprimir_matriz_aterrizaje(solucion_obtenida_greedy)
            costo_greedy_infactible_hc = calcular_costo_total_solucion(solucion_obtenida_greedy)
            print(f"Costo (infactible según este script) de la solución Greedy: {costo_greedy_infactible_hc:.2f}")
            return

        costo_greedy = calcular_costo_total_solucion(solucion_obtenida_greedy)
        print(f"\nSolución Inicial del Greedy Determinista (Factible):")
        print(f"Costo del Greedy: {costo_greedy:.2f}")
        imprimir_matriz_aterrizaje(solucion_obtenida_greedy)

        print("\n>>> Paso 2: Aplicando Hill Climbing (Alguna-Mejora Determinista V2) a la solución del Greedy...")
        # --- MODIFICADO: Recibir historial de costos ---
        solucion_final_hc, hc_cost_history = hill_climbing_alguna_mejora_determinista_v2( 
            solucion_obtenida_greedy,
            numero_aviones,
            paso_tiempo_delta=paso_delta_hc,
            max_iter_hc=max_iteraciones_hc
        )

        costo_final_hc = calcular_costo_total_solucion(solucion_final_hc)
        print("\n--- Resultados Finales Después de Hill Climbing (Alguna-Mejora Determinista V2) ---")
        print(f"Costo Inicial (Greedy): {costo_greedy:.2f}")
        print(f"Costo Final (Hill Climbing): {costo_final_hc:.2f}")
        
        if costo_final_hc < costo_greedy:
            print(f"Mejora obtenida por Hill Climbing: {costo_greedy - costo_final_hc:.2f}")
        elif costo_final_hc == costo_greedy:
            print("Hill Climbing no encontró una mejora respecto a la solución Greedy.")
        else:
            # Esto podría pasar si la solución inicial no era realmente factible para HC,
            # o si hay un error en la lógica de costos/factibilidad.
            print("Advertencia: El costo después de Hill Climbing es mayor que el inicial. Revise la lógica.")

        print("\nMatriz de Aterrizaje Final (después de Hill Climbing):")
        imprimir_matriz_aterrizaje(solucion_final_hc)
        
        if not es_solucion_factible(solucion_final_hc, numero_aviones, verbose=True):
            print("¡¡¡ADVERTENCIA!!! La solución final del Hill Climbing NO es factible según la verificación.")
        else:
            print("La solución final del Hill Climbing es factible.")

        # --- AÑADIDO: Sección de Graficación ---
        if hc_cost_history:
            iteraciones_plot = [item[0] for item in hc_cost_history]
            costos_plot = [item[1] for item in hc_cost_history]

            plt.figure(figsize=(12, 7))
            plt.plot(iteraciones_plot, costos_plot, marker='o', linestyle='-', color='b', label="Costo en Búsqueda Local (HC)")
            
            plt.xlabel("Iteración de Búsqueda Local (Hill Climbing)")
            plt.ylabel("Costo Total de la Solución")
            # --- MODIFICADO: Título del gráfico ---
            titulo_grafico = (f"GRASP: Convergencia de Búsqueda Local (Hill Climbing) para '{ruta_datos_iniciales}'\n"
                              f"Costo Construcción (Greedy): {costo_greedy:.2f} -> Costo Final (Búsqueda Local): {costo_final_hc:.2f}")
            plt.title(titulo_grafico)
            plt.legend()
            plt.grid(True)
            plt.tight_layout() # Ajusta el layout para que todo quepa bien

            # Guardar el gráfico
            nombre_grafico = f"grasp_hc_costo_evolucion_{ruta_datos_iniciales.split('.')[0]}.png" # Nombre de archivo modificado
            try:
                plt.savefig(nombre_grafico)
                print(f"\nGráfico de convergencia guardado como: {nombre_grafico}")
            except Exception as e_plot:
                print(f"Error al guardar el gráfico: {e_plot}")
            
            plt.show() # Mostrar el gráfico
        else:
            print("\nNo se generó historial de costos para Hill Climbing, no se puede graficar.")
            
    except FileNotFoundError:
        print(f"Error CRÍTICO: No se pudo encontrar el archivo de datos '{ruta_datos_iniciales}'.", file=sys.stderr)
    except Exception as e:
        print(f"Ocurrió un error inesperado en main_runner: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_runner()
