import sys
import math
import random # Importado para la aleatoriedad
import os 
import copy # Para deepcopy al guardar soluciones
from typing import List, TextIO, Optional, Tuple

# Definición de la clase para representar cada avión
class Avion:
    """
    Representa un avión con sus datos de aterrizaje y separación.
    """
    def __init__(self,
                 id_avion: int,
                 t_temprano: int,
                 t_preferente: int,
                 t_tardio: int,
                 pen_antes: float,
                 pen_despues: float,
                 t_separacion: List[int],
                 t_seleccionado: Optional[int] = None): # Añadido t_seleccionado al constructor
        """Inicializa una instancia de Avion."""
        self.id_avion = id_avion
        self.tiempo_aterrizaje_temprano = t_temprano
        self.tiempo_aterrizaje_preferente = t_preferente
        self.tiempo_aterrizaje_tardio = t_tardio
        self.penalizacion_antes_preferente = pen_antes
        self.penalizacion_despues_preferente = pen_despues
        self.tiempos_separacion = t_separacion
        self.t_seleccionado: int | None = t_seleccionado # Asignar el t_seleccionado pasado

    def __str__(self) -> str:
        """Representación en string del objeto Avion para impresión completa."""
        info = (
            f"--- Avión ID: {self.id_avion} ---\n"
            f"  T. Temprano: {self.tiempo_aterrizaje_temprano}\n"
            f"  T. Preferente: {self.tiempo_aterrizaje_preferente}\n"
            f"  T. Tardío: {self.tiempo_aterrizaje_tardio}\n"
            f"  Penalización Antes: {self.penalizacion_antes_preferente}\n"
            f"  Penalización Después: {self.penalizacion_despues_preferente}\n"
            f"  Tiempo Seleccionado: {self.t_seleccionado if self.t_seleccionado is not None else 'N/A'}\n"
            f"{'-' * (15 + len(str(self.id_avion)))}"
        )
        return info

    def get_separation_to(self, other_avion: 'Avion') -> int:
        """
        Obtiene el tiempo de separación requerido *después* de que este avión (self)
        aterrice para que el otro avión (other_avion) pueda aterrizar.
        """
        other_avion_index = other_avion.id_avion - 1
        if 0 <= other_avion_index < len(self.tiempos_separacion):
            return self.tiempos_separacion[other_avion_index]
        else:
            # Esto no debería ocurrir si los datos son consistentes
            # print(f"Advertencia: ID de avión {other_avion.id_avion} fuera de rango para tiempos de separación del avión {self.id_avion}.", file=sys.stderr)
            return float('inf') # Retornar un valor alto para indicar problema


# Función para leer los datos desde el archivo
def leer_datos_aviones(ruta_archivo: str) -> List[Avion]:
    """
    Lee la información de los aviones desde un archivo de texto formateado.
    """
    aviones: List[Avion] = []
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            try:
                cantidad_aviones = int(f.readline().strip())
                if cantidad_aviones <= 0:
                    raise ValueError("La cantidad de aviones debe ser positiva.")
            except ValueError as e:
                raise ValueError(f"Error leyendo la cantidad de aviones: {e}") from e

            for i in range(1, cantidad_aviones + 1):
                try:
                    partes_tiempos = f.readline().strip().split()
                    if len(partes_tiempos) != 5:
                        raise ValueError(f"Formato incorrecto para tiempos/penalizaciones del avión {i} (se esperan 5 valores).")
                    t_temprano = int(partes_tiempos[0])
                    t_preferente = int(partes_tiempos[1])
                    t_tardio = int(partes_tiempos[2])
                    pen_antes = float(partes_tiempos[3])
                    pen_despues = float(partes_tiempos[4])
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Error en línea de tiempos/penalización para avión {i}: {e}") from e

                tiempos_sep: List[int] = []
                try:
                    valores_leidos = 0
                    while valores_leidos < cantidad_aviones:
                        linea_sep = f.readline().strip().split()
                        if not linea_sep and valores_leidos < cantidad_aviones: 
                            raise ValueError(f"Línea vacía inesperada al leer tiempos de separación para avión {i} después de leer {valores_leidos} valores.")
                        tiempos_sep.extend(int(t) for t in linea_sep)
                        valores_leidos += len(linea_sep)

                    if len(tiempos_sep) != cantidad_aviones:
                        raise ValueError(f"Se esperaban {cantidad_aviones} tiempos de separación T_ij para avión {i}, "
                                         f"se encontraron {len(tiempos_sep)}.")
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Error en líneas de separación para avión {i}: {e}") from e

                aviones.append(Avion(id_avion=i,
                                     t_temprano=t_temprano,
                                     t_preferente=t_preferente,
                                     t_tardio=t_tardio,
                                     pen_antes=pen_antes,
                                     pen_despues=pen_despues,
                                     t_separacion=tiempos_sep))

            if len(aviones) != cantidad_aviones:
                 raise ValueError(f"Se esperaban {cantidad_aviones} bloques de datos de avión, pero solo se procesaron {len(aviones)}.")

    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en '{ruta_archivo}'", file=sys.stderr)
        raise
    except (ValueError, IOError) as e:
        print(f"Error procesando el archivo '{ruta_archivo}': {e}", file=sys.stderr)
        raise
    return aviones

def imprimir_tiempos_principales(lista_aviones: List[Avion]) -> None:
    """Imprime el ID, tiempo temprano, preferente y tardío de cada avión."""
    # Esta función es más para depuración y puede ser opcional en la ejecución final.
    # print("\nTiempos principales de aterrizaje por avión:")
    # print("-" * 60)
    # for avion in lista_aviones:
    #     print(f"Avión ID: {avion.id_avion:2} | "
    #           f"Temprano (E_k): {avion.tiempo_aterrizaje_temprano:4} | "
    #           f"Preferente (P_k): {avion.tiempo_aterrizaje_preferente:4} | "
    #           f"Tardío (L_k): {avion.tiempo_aterrizaje_tardio:4}")
    # print("-" * 60)
    pass


def cumple_separacion(avion_candidato: Avion, tiempo_aterrizaje: int, aviones_aterrizados: List[Avion]) -> bool:
    """Verifica si el avion_candidato puede aterrizar en tiempo_aterrizaje cumpliendo restricciones de separación."""
    if not aviones_aterrizados:
        return True
    for avion_i in aviones_aterrizados:
        if avion_i.t_seleccionado is not None:
            separacion_requerida_S_ik = avion_i.get_separation_to(avion_candidato)
            if tiempo_aterrizaje < avion_i.t_seleccionado + separacion_requerida_S_ik:
                return False
    return True

def calcular_costo_en_tiempo_especifico(avion: Avion, tiempo_aterrizaje_propuesto: int) -> float:
    """Calcula el costo de penalización si el avión aterriza en un tiempo específico."""
    if tiempo_aterrizaje_propuesto is None: return float('inf') # No debería pasar si se llama correctamente
    diferencia = tiempo_aterrizaje_propuesto - avion.tiempo_aterrizaje_preferente
    costo = 0.0
    if diferencia < 0:
        costo = abs(diferencia) * avion.penalizacion_antes_preferente
    elif diferencia > 0:
        costo = diferencia * avion.penalizacion_despues_preferente
    return costo

# ==============================================================================
# ALGORITMO GREEDY ESTOCÁSTICO CON PONDERACIÓN DE COSTO
# ==============================================================================
def greedy_stochastico_Ek_ponderado_costo(lista_aviones_original: List[Avion], verbose_internal: bool = False) -> Tuple[List[Avion], bool, List[Avion]]:
    """
    Algoritmo greedy estocástico que prioriza E_k e introduce aleatoriedad
    en la selección de candidatos ponderada por el costo de aterrizaje.
    Modifica y devuelve la lista_aviones_original con t_seleccionado actualizado.
    """
    for avion_orig in lista_aviones_original: 
        avion_orig.t_seleccionado = None

    aviones_pendientes = [
        Avion(a.id_avion, a.tiempo_aterrizaje_temprano, a.tiempo_aterrizaje_preferente, a.tiempo_aterrizaje_tardio,
              a.penalizacion_antes_preferente, a.penalizacion_despues_preferente, list(a.tiempos_separacion))
        for a in lista_aviones_original 
    ]
    aviones_aterrizados_ordenados: List[Avion] = []


    if not aviones_pendientes:
        return lista_aviones_original, True, [] 

    tiempo_actual: int = min(avion.tiempo_aterrizaje_temprano for avion in aviones_pendientes) if aviones_pendientes else 0
    
    iteracion_actual = 0
    max_tiempo_tardio_global = 0
    if aviones_pendientes:
        finite_L_ks = [a.tiempo_aterrizaje_tardio for a in aviones_pendientes if a.tiempo_aterrizaje_tardio != float('inf')]
        if finite_L_ks: max_tiempo_tardio_global = max(finite_L_ks)
        else: 
            max_Ek_val = [a.tiempo_aterrizaje_temprano for a in aviones_pendientes if a.tiempo_aterrizaje_temprano != float('-inf')]
            max_Ek = max(max_Ek_val) if max_Ek_val else 0
            max_tiempo_tardio_global = max_Ek + 2000 # Estimación si no hay L_k finitos
    
    max_iteraciones_seguridad = len(aviones_pendientes) * (max_tiempo_tardio_global + len(aviones_pendientes) * 200 + 1000) # Ajustado para ser más generoso
    if not aviones_pendientes: max_iteraciones_seguridad = 1

    if verbose_internal: print(f"--- Iniciando Greedy Estocástico Interno. Max iteraciones seguridad: {max_iteraciones_seguridad} ---")

    while aviones_pendientes and iteracion_actual < max_iteraciones_seguridad:
        iteracion_actual += 1
        avion_programado_en_iteracion = False
        if verbose_internal: print(f"\n  Iter. Interna HC {iteracion_actual}, T={tiempo_actual}, Pendientes: {[a.id_avion for a in aviones_pendientes]}")
        
        aviones_pendientes_ordenados_por_Ek = sorted(
            aviones_pendientes,
            key=lambda a: (a.tiempo_aterrizaje_temprano, a.tiempo_aterrizaje_tardio, a.id_avion)
        )
        if not aviones_pendientes_ordenados_por_Ek: break
        avion_mas_temprano_Ek_global = aviones_pendientes_ordenados_por_Ek[0]

        tiempo_min_aterrizaje_para_avion_Ek_global = avion_mas_temprano_Ek_global.tiempo_aterrizaje_temprano
        if aviones_aterrizados_ordenados:
            ultimo_avion = aviones_aterrizados_ordenados[-1]
            if ultimo_avion.t_seleccionado is not None :
                separacion_necesaria = ultimo_avion.get_separation_to(avion_mas_temprano_Ek_global)
                tiempo_min_aterrizaje_para_avion_Ek_global = max(
                    tiempo_min_aterrizaje_para_avion_Ek_global,
                    ultimo_avion.t_seleccionado + separacion_necesaria
                )
        
        if tiempo_actual < tiempo_min_aterrizaje_para_avion_Ek_global:
            if tiempo_min_aterrizaje_para_avion_Ek_global <= avion_mas_temprano_Ek_global.tiempo_aterrizaje_tardio:
                if verbose_internal: print(f"    Avanzando T de {tiempo_actual} a {tiempo_min_aterrizaje_para_avion_Ek_global} para avión ID {avion_mas_temprano_Ek_global.id_avion}")
                tiempo_actual = tiempo_min_aterrizaje_para_avion_Ek_global
        
        candidatos_para_aterrizar_en_T_actual = []
        for avion_k in aviones_pendientes:
            if avion_k.tiempo_aterrizaje_temprano <= tiempo_actual <= avion_k.tiempo_aterrizaje_tardio:
                if cumple_separacion(avion_k, tiempo_actual, aviones_aterrizados_ordenados):
                    candidatos_para_aterrizar_en_T_actual.append(avion_k)
        
        if verbose_internal: print(f"    Candidatos para T={tiempo_actual}: {[c.id_avion for c in candidatos_para_aterrizar_en_T_actual]}")

        if candidatos_para_aterrizar_en_T_actual:
            avion_a_aterrizar: Optional[Avion] = None
            if len(candidatos_para_aterrizar_en_T_actual) == 1:
                avion_a_aterrizar = candidatos_para_aterrizar_en_T_actual[0]
            else:
                # Ponderación por costo inverso (menor costo, mayor peso)
                costs = [calcular_costo_en_tiempo_especifico(avion, tiempo_actual) for avion in candidatos_para_aterrizar_en_T_actual]
                epsilon = 0.001 # Pequeño valor para evitar división por cero si el costo es 0
                weights = [1.0 / (cost + epsilon) for cost in costs]
                
                sum_weights = sum(weights)
                # Si todos los pesos son muy pequeños (costos muy altos) o todos iguales, elegir aleatoriamente
                if sum_weights < epsilon / 100.0 or len(set(weights)) == 1 : 
                    avion_a_aterrizar = random.choice(candidatos_para_aterrizar_en_T_actual)
                    if verbose_internal: print(f"      Selección aleatoria (pesos bajos/iguales) entre {len(candidatos_para_aterrizar_en_T_actual)} candidatos.")
                else:
                    try:
                        avion_a_aterrizar = random.choices(candidatos_para_aterrizar_en_T_actual, weights=weights, k=1)[0]
                        if verbose_internal: print(f"      Selección ponderada por costo entre {len(candidatos_para_aterrizar_en_T_actual)} candidatos.")
                    except ValueError: # Puede ocurrir si la lista de pesos está vacía o contiene valores no válidos
                        avion_a_aterrizar = random.choice(candidatos_para_aterrizar_en_T_actual)
                        if verbose_internal: print(f"      Selección aleatoria (error en ponderación) entre {len(candidatos_para_aterrizar_en_T_actual)} candidatos.")


            if avion_a_aterrizar:
                if verbose_internal: print(f"    Avión seleccionado ID {avion_a_aterrizar.id_avion} para aterrizar en T={tiempo_actual}")
                avion_a_aterrizar.t_seleccionado = tiempo_actual
                aviones_aterrizados_ordenados.append(avion_a_aterrizar)
                aviones_pendientes.remove(avion_a_aterrizar)
                avion_programado_en_iteracion = True

        if not aviones_pendientes:
            if verbose_internal: print("    ¡Todos los aviones programados internamente!")
            break

        if not avion_programado_en_iteracion:
            proximo_tiempo_minimo_posible = float('inf')
            if aviones_pendientes:
                for ap in aviones_pendientes:
                    if ap.tiempo_aterrizaje_temprano >= tiempo_actual:
                        proximo_tiempo_minimo_posible = min(proximo_tiempo_minimo_posible, ap.tiempo_aterrizaje_temprano)

                _aviones_pend_ord_ek_temp_adv = sorted(aviones_pendientes, key=lambda a: (a.tiempo_aterrizaje_temprano, a.id_avion))
                if _aviones_pend_ord_ek_temp_adv:
                    avion_ref_menor_ek_actual = _aviones_pend_ord_ek_temp_adv[0]
                    tiempo_min_aterrizaje_ref = avion_ref_menor_ek_actual.tiempo_aterrizaje_temprano
                    if aviones_aterrizados_ordenados:
                        ultimo_avion = aviones_aterrizados_ordenados[-1]
                        if ultimo_avion.t_seleccionado is not None:
                            sep_necesaria = ultimo_avion.get_separation_to(avion_ref_menor_ek_actual)
                            tiempo_min_aterrizaje_ref = max(tiempo_min_aterrizaje_ref, ultimo_avion.t_seleccionado + sep_necesaria)
                    
                    if tiempo_min_aterrizaje_ref <= avion_ref_menor_ek_actual.tiempo_aterrizaje_tardio:
                        proximo_tiempo_minimo_posible = min(proximo_tiempo_minimo_posible, tiempo_min_aterrizaje_ref)
            
            if proximo_tiempo_minimo_posible == float('inf'):
                if aviones_pendientes: 
                    if verbose_internal: print(f"    No se pudo programar y no hay evento claro. Avanzando T+1 desde {tiempo_actual}")
                    tiempo_actual += 1 
            elif proximo_tiempo_minimo_posible > tiempo_actual:
                if verbose_internal: print(f"    Ningún avión pudo aterrizar. Avanzando T de {tiempo_actual} a {proximo_tiempo_minimo_posible}")
                tiempo_actual = proximo_tiempo_minimo_posible
            else: 
                if verbose_internal: print(f"    Ningún avión pudo aterrizar en T={tiempo_actual}. Avanzando T+1 desde {tiempo_actual}")
                tiempo_actual += 1
        
        if not avion_programado_en_iteracion and aviones_pendientes:
            # Chequeo de seguridad para L_k
            current_max_lk_pendiente = 0
            finite_Lks_pend = [ap.tiempo_aterrizaje_tardio for ap in aviones_pendientes if ap.tiempo_aterrizaje_tardio != float('inf')]
            if finite_Lks_pend: current_max_lk_pendiente = max(finite_Lks_pend)
            else: 
                max_Ek_pend_val = [ap.tiempo_aterrizaje_temprano for ap in aviones_pendientes if ap.tiempo_aterrizaje_temprano != float('-inf')]
                current_max_lk_pendiente = (max(max_Ek_pend_val) if max_Ek_pend_val else tiempo_actual) + 2000 

            if tiempo_actual > current_max_lk_pendiente + (len(aviones_pendientes) * 50) : # Margen de seguridad
                 if verbose_internal: print(f"    ADVERTENCIA INTERNA: T={tiempo_actual} ha superado L_k max pendiente ({current_max_lk_pendiente}). Forzando salida del bucle interno.")
                 break # Salir si T se dispara demasiado

    if iteracion_actual >= max_iteraciones_seguridad and aviones_pendientes:
        if verbose_internal: print(f"\nAdvertencia (interno): Límite de iteraciones ({max_iteraciones_seguridad}) alcanzado. "
              f"Quedan {len(aviones_pendientes)} aviones sin programar: {[a.id_avion for a in aviones_pendientes]}")

    # Actualizar la lista original (lista_aviones_original) con los t_seleccionado de esta ejecución.
    mapa_aviones_aterrizados_internos = {avion.id_avion: avion.t_seleccionado for avion in aviones_aterrizados_ordenados}
    for avion_original_obj in lista_aviones_original:
        if avion_original_obj.id_avion in mapa_aviones_aterrizados_internos:
            avion_original_obj.t_seleccionado = mapa_aviones_aterrizados_internos[avion_original_obj.id_avion]
        # else: t_seleccionado ya fue reseteado a None al inicio de esta función

    # --- Verificación final de factibilidad sobre la lista original actualizada ---
    es_factible_final = True
    aviones_con_problemas_ids = set()
    
    aviones_no_programados_final = [avion for avion in lista_aviones_original if avion.t_seleccionado is None]
    if aviones_no_programados_final:
        es_factible_final = False
        for avion in aviones_no_programados_final:
            aviones_con_problemas_ids.add(avion.id_avion)

    aviones_si_programados = [avion for avion in lista_aviones_original if avion.t_seleccionado is not None]
    for avion in aviones_si_programados:
        if avion.t_seleccionado is None: continue 
        if not (avion.tiempo_aterrizaje_temprano <= avion.t_seleccionado <= avion.tiempo_aterrizaje_tardio):
            es_factible_final = False
            aviones_con_problemas_ids.add(avion.id_avion)

    aviones_aterrizados_para_verificacion = sorted(
        [a for a in lista_aviones_original if a.t_seleccionado is not None],
        key=lambda avion: avion.t_seleccionado if avion.t_seleccionado is not None else float('inf')
    )

    for i in range(len(aviones_aterrizados_para_verificacion)):
        for j in range(i + 1, len(aviones_aterrizados_para_verificacion)):
            avion_i = aviones_aterrizados_para_verificacion[i]
            avion_j = aviones_aterrizados_para_verificacion[j]
            separacion_requerida_S_ij = avion_i.get_separation_to(avion_j)
            if avion_i.t_seleccionado is not None and avion_j.t_seleccionado is not None:
                 separacion_real = avion_j.t_seleccionado - avion_i.t_seleccionado
                 if separacion_real < separacion_requerida_S_ij:
                    es_factible_final = False
                    aviones_con_problemas_ids.add(avion_i.id_avion)
                    aviones_con_problemas_ids.add(avion_j.id_avion)
            else: 
                es_factible_final = False 
                aviones_con_problemas_ids.add(avion_i.id_avion)
                aviones_con_problemas_ids.add(avion_j.id_avion)

    mapa_aviones_original_por_id = {avion.id_avion: avion for avion in lista_aviones_original}
    aviones_con_problemas_obj = [mapa_aviones_original_por_id[id_avion] for id_avion in sorted(list(aviones_con_problemas_ids))]
    
    return lista_aviones_original, es_factible_final and not aviones_no_programados_final, aviones_con_problemas_obj
# ==============================================================================

def calcular_costo_total(lista_aviones: List[Avion]) -> float:
    """Calcula el costo total de la programación de aterrizajes."""
    costo_total = 0.0
    aviones_efectivamente_programados = [avion for avion in lista_aviones if avion.t_seleccionado is not None]
    if not aviones_efectivamente_programados:
        return 0.0
    for avion in aviones_efectivamente_programados:
        if avion.t_seleccionado is not None: 
            costo_total += calcular_costo_en_tiempo_especifico(avion, avion.t_seleccionado)
    return costo_total

def imprimir_matriz_aterrizaje(lista_aviones_procesada: List[Avion], cabecera: str = "Matriz de Horario de Aterrizaje") -> None:
    """Imprime una matriz con el horario de aterrizaje de los aviones programados."""
    print(f"\n--- {cabecera} ---")
    print("---------------------------------------")
    print("| Tiempo Aterrizaje | ID Avión      |")
    print("|-------------------|---------------|")
    aviones_aterrizados = [avion for avion in lista_aviones_procesada if avion.t_seleccionado is not None]
    if not aviones_aterrizados:
        print("| Sin aterrizajes   | N/A           |")
    else:
        aviones_aterrizados.sort(key=lambda avion: avion.t_seleccionado if avion.t_seleccionado is not None else float('inf'))
        for avion in aviones_aterrizados:
            tiempo_str = f"{avion.t_seleccionado:<17}" if avion.t_seleccionado is not None else "Error: T None   "
            avion_id_str = f"{avion.id_avion:<13}"
            print(f"| {tiempo_str} | {avion_id_str} |")
    print("---------------------------------------")

def buscar_solucion_factible_estocastica(
        lista_aviones_original_raw: List[Avion], 
        max_intentos_greedy: int, # Renombrado para claridad
        verbose_greedy_internal: bool = False 
    ) -> Tuple[Optional[List[Avion]], float, int, bool]:
    """
    Ejecuta el algoritmo greedy estocástico hasta encontrar una solución factible
    o alcanzar el número máximo de intentos.
    Devuelve la primera solución factible encontrada.
    """
    solucion_encontrada_aviones: Optional[List[Avion]] = None
    costo_solucion_encontrada = float('inf')
    iteracion_encontrada = -1
    factible_hallada = False

    # No imprimir esto aquí, se imprimirá por cada semilla en main
    # print(f"\n--- Buscando Solución Factible (Máx. Intentos: {max_intentos_greedy}) ---")

    for intento_actual in range(max_intentos_greedy):
        aviones_para_intento_actual = [
            Avion(a.id_avion, a.tiempo_aterrizaje_temprano, a.tiempo_aterrizaje_preferente, a.tiempo_aterrizaje_tardio,
                  a.penalizacion_antes_preferente, a.penalizacion_despues_preferente, list(a.tiempos_separacion))
            for a in lista_aviones_original_raw 
        ]
        
        # Pasar verbose_greedy_internal a la función greedy
        lista_procesada_intento, es_completamente_factible, _ = greedy_stochastico_Ek_ponderado_costo(aviones_para_intento_actual, verbose_internal=verbose_greedy_internal)
        
        if es_completamente_factible: 
            if not verbose_greedy_internal : # Solo imprimir si no estamos en modo verbose interno
                print(f"  ¡Solución factible encontrada en el intento {intento_actual + 1}/{max_intentos_greedy} del greedy estocástico!")
            factible_hallada = True
            costo_solucion_encontrada = calcular_costo_total(lista_procesada_intento)
            solucion_encontrada_aviones = [ 
                Avion(a.id_avion, a.tiempo_aterrizaje_temprano, a.tiempo_aterrizaje_preferente, a.tiempo_aterrizaje_tardio,
                      a.penalizacion_antes_preferente, a.penalizacion_despues_preferente, list(a.tiempos_separacion),
                      t_seleccionado=a.t_seleccionado) 
                for a in lista_procesada_intento 
            ]
            iteracion_encontrada = intento_actual + 1
            break 
        else:
            # Imprimir progreso solo si no estamos en modo verbose interno y es un hito
            if not verbose_greedy_internal and (intento_actual + 1) % 50 == 0 and max_intentos_greedy > 50 :
                 print(f"    Intento {intento_actual + 1}/{max_intentos_greedy} del greedy: No se encontró solución factible aún.")

    # No imprimir resumen aquí, se hará en main para cada semilla
    return solucion_encontrada_aviones, costo_solucion_encontrada, iteracion_encontrada, factible_hallada


def main() -> Tuple[List[Optional[List[Avion]]], List[float]]: # Modificado para devolver lista de soluciones y costos
    """
    Función principal: Lee datos, ejecuta el greedy estocástico 10 veces con semillas predefinidas
    y devuelve las soluciones factibles encontradas.
    """
    ruta_datos = "case3.txt" # Asegúrate que este archivo exista o ajústalo
    # ruta_datos = "case3.txt" 
    
    # Lista de 10 semillas predefinidas
    semillas_predefinidas = [42, 123, 7, 99, 500, 777, 2024, 1, 100, 314]
    num_soluciones_a_generar = len(semillas_predefinidas)

    soluciones_factibles_generadas: List[Optional[List[Avion]]] = []
    costos_soluciones_generadas: List[float] = []

    print(f"Iniciando generación de {num_soluciones_a_generar} soluciones factibles con semillas predefinidas...")

    try:
        lista_de_aviones_leidos_base = leer_datos_aviones(ruta_datos)
        print(f"Datos leídos correctamente para {len(lista_de_aviones_leidos_base)} aviones desde '{ruta_datos}'.\n")
        # imprimir_tiempos_principales(lista_de_aviones_leidos_base) # Opcional

        max_intentos_por_semilla = 1000 # Intentos para que el greedy estocástico encuentre una solución factible por semilla

        for i in range(num_soluciones_a_generar):
            semilla_actual = semillas_predefinidas[i]
            print(f"\n--- Ejecución Estocástica #{i+1}/{num_soluciones_a_generar} (Semilla: {semilla_actual}) ---")
            random.seed(semilla_actual) # Establecer la semilla

            # Crear una copia fresca de los datos base para esta ejecución con semilla
            # Esto es importante si leer_datos_aviones devuelve objetos que podrían ser modificados
            # por buscar_solucion_factible_estocastica en ejecuciones anteriores (aunque no debería si se hacen copias internas).
            # Para mayor seguridad, se puede hacer una copia profunda aquí también si es necesario,
            # pero buscar_solucion_factible_estocastica ya hace copias.
            lista_aviones_para_semilla = copy.deepcopy(lista_de_aviones_leidos_base)


            # El verbose_greedy_internal puede ser útil para depurar una semilla específica si no encuentra solución
            solucion, costo, intentos, fue_factible = buscar_solucion_factible_estocastica(
                lista_aviones_para_semilla, # Pasar la copia fresca
                max_intentos_por_semilla,
                verbose_greedy_internal=False # Poner True para depurar una semilla específica
            )

            if fue_factible and solucion:
                print(f"  Solución factible encontrada para semilla {semilla_actual} en {intentos} intento(s) del greedy.")
                print(f"  Costo de esta solución: {costo:.2f}")
                # imprimir_matriz_aterrizaje(solucion, f"Horario para Semilla {semilla_actual}") # Opcional
                soluciones_factibles_generadas.append(copy.deepcopy(solucion)) # Guardar una copia
                costos_soluciones_generadas.append(costo)
            else:
                print(f"  No se encontró solución factible para semilla {semilla_actual} después de {max_intentos_por_semilla} intentos del greedy.")
                soluciones_factibles_generadas.append(None) # Marcar que no se encontró para esta semilla
                costos_soluciones_generadas.append(float('inf'))
        
        print(f"\n--- Resumen de Generación de Soluciones Estocásticas ---")
        num_exitos = sum(1 for s in soluciones_factibles_generadas if s is not None)
        print(f"Se generaron {num_exitos} soluciones factibles de {num_soluciones_a_generar} intentos con semillas predefinidas.")
        
        return soluciones_factibles_generadas, costos_soluciones_generadas

    except FileNotFoundError:
        print(f"Error CRÍTICO: No se pudo encontrar el archivo de datos en '{ruta_datos}'. Verifica la ruta.", file=sys.stderr)
        return [None] * num_soluciones_a_generar, [float('inf')] * num_soluciones_a_generar # Devolver Nones en caso de error mayor
    except (ValueError, IOError) as e:
        print(f"Error CRÍTICO procesando el archivo '{ruta_datos}': {e}", file=sys.stderr)
        return [None] * num_soluciones_a_generar, [float('inf')] * num_soluciones_a_generar
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la ejecución: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return [None] * num_soluciones_a_generar, [float('inf')] * num_soluciones_a_generar

if __name__ == "__main__":
    # La función main ahora devuelve las soluciones, así que si ejecutas este script directamente,
    # podrías querer hacer algo con el resultado aquí, o simplemente dejar que imprima.
    lista_soluciones_generadas, lista_costos_generados = main()
    
    print("\n\n--- Costos de las Soluciones Estocásticas Generadas ---")
    for idx, costo_sol in enumerate(lista_costos_generados):
        if lista_soluciones_generadas[idx] is not None:
            print(f"Solución con Semilla #{idx+1} ({[42, 123, 7, 99, 500, 777, 2024, 1, 100, 314][idx]}): Costo = {costo_sol:.2f}")
        else:
            print(f"Solución con Semilla #{idx+1} ({[42, 123, 7, 99, 500, 777, 2024, 1, 100, 314][idx]}): No se encontró solución factible.")

    # sys.exit(0) # Puedes decidir el código de salida basado en si se generaron todas las soluciones
