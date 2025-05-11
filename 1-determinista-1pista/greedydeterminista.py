import sys
import math
from typing import List, TextIO, Optional, Tuple

# Definición de la clase para representar cada avión
class Avion:
    """
    Representa un avión con sus datos de aterrizaje y separación.

    Atributos:
        id_avion (int): Identificador único (1 a D).
        tiempo_aterrizaje_temprano (int): T. más temprano de aterrizaje (E_k).
        tiempo_aterrizaje_preferente (int): T. preferente de aterrizaje.
        tiempo_aterrizaje_tardio (int): T. más tarde de aterrizaje (L_k).
        penalizacion_antes_preferente (float): Costo por aterrizar antes (C_k).
        penalizacion_despues_preferente (float): Costo por aterrizar después (C'_k).
        tiempos_separacion (List[int]): Tiempos T_ij requeridos después de
                                          que este avión (i) aterrice para
                                          que el avión j pueda aterrizar.
        t_seleccionado (int | None): El tiempo de aterrizaje asignado por el
                                     algoritmo greedy, o None si aún no ha
                                     sido programado.
    """
    def __init__(self,
                 id_avion: int,
                 t_temprano: int,
                 t_preferente: int,
                 t_tardio: int,
                 pen_antes: float,
                 pen_despues: float,
                 t_separacion: List[int]):
        """Inicializa una instancia de Avion."""
        self.id_avion = id_avion
        self.tiempo_aterrizaje_temprano = t_temprano
        self.tiempo_aterrizaje_preferente = t_preferente
        self.tiempo_aterrizaje_tardio = t_tardio
        self.penalizacion_antes_preferente = pen_antes
        self.penalizacion_despues_preferente = pen_despues
        self.tiempos_separacion = t_separacion
        self.t_seleccionado: int | None = None

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
            print(f"Advertencia: ID de avión {other_avion.id_avion} fuera de rango para tiempos de separación del avión {self.id_avion}.", file=sys.stderr)
            return float('inf')


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
    """
    Imprime el ID, tiempo temprano, preferente y tardío de cada avión.
    """
    print("\nTiempos principales de aterrizaje por avión:")
    print("-" * 60)
    for avion in lista_aviones:
        print(f"Avión ID: {avion.id_avion:2} | "
              f"Temprano (E_k): {avion.tiempo_aterrizaje_temprano:4} | "
              f"Preferente (P_k): {avion.tiempo_aterrizaje_preferente:4} | "
              f"Tardío (L_k): {avion.tiempo_aterrizaje_tardio:4}")
    print("-" * 60)

def cumple_separacion(avion_candidato: Avion, tiempo_aterrizaje: int, aviones_aterrizados: List[Avion]) -> bool:
    """
    Verifica si el avion_candidato puede aterrizar en tiempo_aterrizaje
    cumpliendo las restricciones de separación con los aviones_aterrizados.
    """
    if not aviones_aterrizados: # Si no hay aviones aterrizados, siempre se cumple
        return True
    for avion_i in aviones_aterrizados:
        if avion_i.t_seleccionado is not None:
            separacion_requerida_S_ij = avion_i.get_separation_to(avion_candidato)
            if tiempo_aterrizaje < avion_i.t_seleccionado + separacion_requerida_S_ij:
                return False
    return True

def calcular_costo_en_tiempo_especifico(avion: Avion, tiempo_aterrizaje_propuesto: int) -> float:
    """
    Calcula el costo de penalización si el avión aterriza en un tiempo específico.
    """
    diferencia = tiempo_aterrizaje_propuesto - avion.tiempo_aterrizaje_preferente
    costo = 0.0
    if diferencia < 0:
        costo = abs(diferencia) * avion.penalizacion_antes_preferente
    elif diferencia > 0:
        costo = diferencia * avion.penalizacion_despues_preferente
    return costo

# ==============================================================================
#               NUEVO ALGORITMO GREEDY: PRIORIZAR MENOR E_k
# ==============================================================================
def greedy_priorizar_menor_Ek(lista_aviones_original: List[Avion]) -> Tuple[List[Avion], bool, List[Avion]]:
    """
    Algoritmo greedy que prioriza aterrizar aviones con el menor tiempo temprano (E_k).
    Si un avión con E_k bajo no puede aterrizar por separación, se adelanta el tiempo T
    lo mínimo necesario para que pueda hacerlo, siempre que no exceda su L_k.
    El objetivo principal es que todos los aviones aterricen.
    """
    # Reseteamos t_seleccionado por si se reutiliza la lista
    for avion in lista_aviones_original:
        avion.t_seleccionado = None

    # Copiamos la lista para no modificar la original directamente en términos de orden
    # aunque los objetos Avion sí se modificarán (t_seleccionado)
    lista_aviones = lista_aviones_original[:]
    aviones_aterrizados_ordenados: List[Avion] = []
    aviones_pendientes = lista_aviones[:]

    if not aviones_pendientes:
        print("No hay aviones para programar.")
        return [], True, []

    # Inicializar el tiempo actual T. Podría ser el E_k mínimo de todos los aviones.
    tiempo_actual: int = min(avion.tiempo_aterrizaje_temprano for avion in aviones_pendientes)
    print(f"\n--- Ejecutando Algoritmo Greedy: Priorizar Menor E_k ---")
    print(f"Tiempo inicial T: {tiempo_actual}")

    iteracion_actual = 0
    # Un límite generoso para evitar bucles infinitos en casos inesperados.
    # Considera el número de aviones y el rango máximo de tiempos de aterrizaje.
    max_tiempo_tardio_global = 0
    if lista_aviones: # Asegurar que la lista no está vacía
        max_tiempo_tardio_global = max(a.tiempo_aterrizaje_tardio for a in lista_aviones)
    
    # Incrementamos el factor para dar más margen, especialmente si hay muchos aviones o tiempos de separación grandes
    max_iteraciones_seguridad = len(lista_aviones) * (max_tiempo_tardio_global + len(lista_aviones) * 100)
    if not lista_aviones: # Si no hay aviones, no hay iteraciones.
        max_iteraciones_seguridad = 1


    while aviones_pendientes and iteracion_actual < max_iteraciones_seguridad:
        iteracion_actual += 1
        avion_programado_en_iteracion = False
        print(f"\n--- Iteración {iteracion_actual}, Tiempo Actual (T): {tiempo_actual} ---")
        print(f"Aviones pendientes: {[a.id_avion for a in sorted(aviones_pendientes, key=lambda x: x.id_avion)]}")
        if aviones_aterrizados_ordenados:
            ultimo_aterrizado = aviones_aterrizados_ordenados[-1]
            print(f"Último avión aterrizado: ID {ultimo_aterrizado.id_avion} en t={ultimo_aterrizado.t_seleccionado}")
        else:
            print("Ningún avión ha aterrizado aún.")

        # 1. Identificar el avión con el E_k más temprano entre los pendientes
        aviones_pendientes_ordenados_por_Ek = sorted(
            aviones_pendientes,
            key=lambda a: (a.tiempo_aterrizaje_temprano, a.tiempo_aterrizaje_tardio, a.id_avion)
        )
        avion_mas_temprano_Ek_global = aviones_pendientes_ordenados_por_Ek[0]
        print(f"  Avión global con menor E_k pendiente: ID {avion_mas_temprano_Ek_global.id_avion} (E_k: {avion_mas_temprano_Ek_global.tiempo_aterrizaje_temprano})")

        # 2. Calcular el tiempo mínimo en que este avión (avion_mas_temprano_Ek_global) podría aterrizar
        #    Considerando su E_k y la separación con el último avión aterrizado.
        tiempo_min_aterrizaje_para_avion_Ek_global = avion_mas_temprano_Ek_global.tiempo_aterrizaje_temprano
        if aviones_aterrizados_ordenados:
            ultimo_avion = aviones_aterrizados_ordenados[-1]
            separacion_necesaria = ultimo_avion.get_separation_to(avion_mas_temprano_Ek_global)
            if ultimo_avion.t_seleccionado is not None : # Asegurarse que el tiempo de aterrizaje no es None
                tiempo_min_aterrizaje_para_avion_Ek_global = max(
                    tiempo_min_aterrizaje_para_avion_Ek_global,
                    ultimo_avion.t_seleccionado + separacion_necesaria
                )

        # 3. Si el tiempo_actual es menor que el tiempo mínimo en que el avión más temprano (Ek global) podría aterrizar,
        #    y ese tiempo mínimo está dentro de su ventana [Ek, Lk], entonces adelantamos T.
        if tiempo_actual < tiempo_min_aterrizaje_para_avion_Ek_global:
            if tiempo_min_aterrizaje_para_avion_Ek_global <= avion_mas_temprano_Ek_global.tiempo_aterrizaje_tardio:
                print(f"  Adelantando T de {tiempo_actual} a {tiempo_min_aterrizaje_para_avion_Ek_global} para permitir aterrizar al avión ID {avion_mas_temprano_Ek_global.id_avion} (menor E_k global).")
                tiempo_actual = tiempo_min_aterrizaje_para_avion_Ek_global
            else:
                print(f"  ADVERTENCIA: Adelantar T a {tiempo_min_aterrizaje_para_avion_Ek_global} para el avión ID {avion_mas_temprano_Ek_global.id_avion} "
                      f"excedería su L_k ({avion_mas_temprano_Ek_global.tiempo_aterrizaje_tardio}). Se mantiene T={tiempo_actual} por ahora.")


        # 4. Encontrar candidatos que puedan aterrizar en el `tiempo_actual`
        candidatos_para_aterrizar_en_T_actual = []
        for avion_k in aviones_pendientes:
            if avion_k.tiempo_aterrizaje_temprano <= tiempo_actual <= avion_k.tiempo_aterrizaje_tardio:
                if cumple_separacion(avion_k, tiempo_actual, aviones_aterrizados_ordenados):
                    candidatos_para_aterrizar_en_T_actual.append(avion_k)

        print(f"  Candidatos para aterrizar en T={tiempo_actual} (ventana y separación OK): {[c.id_avion for c in candidatos_para_aterrizar_en_T_actual]}")

        if candidatos_para_aterrizar_en_T_actual:
            # 5. Seleccionar el mejor candidato de los que pueden aterrizar AHORA
            candidatos_ordenados = sorted(
                candidatos_para_aterrizar_en_T_actual,
                key=lambda a: (a.tiempo_aterrizaje_temprano, a.tiempo_aterrizaje_tardio, a.tiempo_aterrizaje_preferente, a.id_avion)
            )
            avion_a_aterrizar = candidatos_ordenados[0]

            print(f"  Avión seleccionado para aterrizar en T={tiempo_actual}: ID {avion_a_aterrizar.id_avion} "
                  f"(E_k: {avion_a_aterrizar.tiempo_aterrizaje_temprano}, L_k: {avion_a_aterrizar.tiempo_aterrizaje_tardio})")

            avion_a_aterrizar.t_seleccionado = tiempo_actual
            aviones_aterrizados_ordenados.append(avion_a_aterrizar)
            aviones_pendientes.remove(avion_a_aterrizar)
            avion_programado_en_iteracion = True

        if not aviones_pendientes:
            print("¡Todos los aviones han sido programados!")
            break

        if not avion_programado_en_iteracion:
            proximo_tiempo_minimo_posible = float('inf')
            if aviones_pendientes: # Solo si quedan aviones pendientes
                # Opción A: El E_k más cercano de un avión pendiente que sea >= T actual.
                for ap in aviones_pendientes:
                    if ap.tiempo_aterrizaje_temprano >= tiempo_actual: # Considerar solo E_k futuros o el actual
                        proximo_tiempo_minimo_posible = min(proximo_tiempo_minimo_posible, ap.tiempo_aterrizaje_temprano)

                # Opción B: El tiempo en que el avión con menor E_k global (recalculado) podría aterrizar
                _aviones_pend_ord_ek_temp = sorted(aviones_pendientes, key=lambda a: (a.tiempo_aterrizaje_temprano, a.id_avion))
                avion_ref_menor_ek_actual = _aviones_pend_ord_ek_temp[0]
                
                tiempo_min_aterrizaje_ref = avion_ref_menor_ek_actual.tiempo_aterrizaje_temprano
                if aviones_aterrizados_ordenados:
                    ultimo_avion = aviones_aterrizados_ordenados[-1]
                    if ultimo_avion.t_seleccionado is not None: # Chequeo de None
                        sep_necesaria = ultimo_avion.get_separation_to(avion_ref_menor_ek_actual)
                        tiempo_min_aterrizaje_ref = max(tiempo_min_aterrizaje_ref, ultimo_avion.t_seleccionado + sep_necesaria)

                if tiempo_min_aterrizaje_ref <= avion_ref_menor_ek_actual.tiempo_aterrizaje_tardio:
                     proximo_tiempo_minimo_posible = min(proximo_tiempo_minimo_posible, tiempo_min_aterrizaje_ref)
            
            if proximo_tiempo_minimo_posible == float('inf'):
                if aviones_pendientes:
                    print(f"  ADVERTENCIA: No se pudo programar avión y no se encontró un próximo tiempo de evento claro. Aviones pendientes: {[a.id_avion for a in aviones_pendientes]}. Avanzando T+1.")
                    # Intentar avanzar T basado en el L_k más restrictivo si T ya lo superó
                    avion_mas_temprano_L_k_pendiente = min(ap.tiempo_aterrizaje_tardio for ap in aviones_pendientes)
                    if tiempo_actual > avion_mas_temprano_L_k_pendiente:
                         print(f"  CRÍTICO: El tiempo actual T={tiempo_actual} ha superado el L_k ({avion_mas_temprano_L_k_pendiente}) "
                               f"del avión más restrictivo pendiente. Es improbable que todos aterricen.")
                         # Aquí se podría romper el bucle si se considera que ya no es factible.
                         # Por ahora, se permite que el chequeo de aviones_pendientes al final lo maneje.
                    tiempo_actual += 1 # Avance mínimo para evitar estancamiento
                # else: No hay aviones pendientes, el bucle principal debería terminar.
            elif proximo_tiempo_minimo_posible > tiempo_actual:
                print(f"  Ningún avión pudo aterrizar en T={tiempo_actual}. Avanzando T a {proximo_tiempo_minimo_posible}.")
                tiempo_actual = proximo_tiempo_minimo_posible
            else: # proximo_tiempo_minimo_posible <= tiempo_actual
                print(f"  Ningún avión pudo aterrizar en T={tiempo_actual} (que ya era un tiempo candidato). Avanzando T en +1.")
                tiempo_actual += 1
        
        if not avion_programado_en_iteracion and aviones_pendientes:
            # Medida de seguridad adicional: si T supera el L_k máximo de todos los aviones pendientes, algo va mal.
            max_lk_pendiente = max(ap.tiempo_aterrizaje_tardio for ap in aviones_pendientes)
            if tiempo_actual > max_lk_pendiente + 10: # Un margen
                print(f"ADVERTENCIA: Tiempo actual T={tiempo_actual} ha superado significativamente el L_k máximo de los aviones pendientes ({max_lk_pendiente}). Forzando salida.")
                break


    if iteracion_actual >= max_iteraciones_seguridad and aviones_pendientes:
        print(f"\nAdvertencia: Límite de iteraciones de seguridad ({max_iteraciones_seguridad}) alcanzado. "
              f"Quedan {len(aviones_pendientes)} aviones sin programar: {[a.id_avion for a in aviones_pendientes]}")

    print("\n--- Algoritmo Greedy (Priorizar Menor E_k) Finalizado ---")

    es_factible = True
    aviones_con_problemas_ids = set()

    aviones_no_programados_final = [avion for avion in lista_aviones_original if avion.t_seleccionado is None]
    if aviones_no_programados_final:
        es_factible = False
        print(f"  ❌ No todos los aviones ({len(aviones_no_programados_final)}) pudieron ser programados.")
        for avion in aviones_no_programados_final:
            print(f"    Avión ID {avion.id_avion}: No fue programado.")
            aviones_con_problemas_ids.add(avion.id_avion)

    aviones_si_programados = [avion for avion in lista_aviones_original if avion.t_seleccionado is not None]
    for avion in aviones_si_programados:
        if avion.t_seleccionado is None: continue
        if not (avion.tiempo_aterrizaje_temprano <= avion.t_seleccionado <= avion.tiempo_aterrizaje_tardio):
            print(f"    ❌ Avión ID {avion.id_avion}: Tiempo asignado {avion.t_seleccionado} está FUERA de su ventana [{avion.tiempo_aterrizaje_temprano}, {avion.tiempo_aterrizaje_tardio}].")
            es_factible = False
            aviones_con_problemas_ids.add(avion.id_avion)

    aviones_aterrizados_para_verificacion = sorted(aviones_si_programados, key=lambda avion: avion.t_seleccionado if avion.t_seleccionado is not None else float('inf'))
    for i in range(len(aviones_aterrizados_para_verificacion)):
        for j in range(i + 1, len(aviones_aterrizados_para_verificacion)):
            avion_i = aviones_aterrizados_para_verificacion[i]
            avion_j = aviones_aterrizados_para_verificacion[j]
            if avion_i.t_seleccionado is not None and avion_j.t_seleccionado is not None:
                separacion_requerida_S_ij = avion_i.get_separation_to(avion_j)
                separacion_real = avion_j.t_seleccionado - avion_i.t_seleccionado
                if separacion_real < separacion_requerida_S_ij:
                    print(f"    ❌ VIOLACIÓN DE SEPARACIÓN S_{avion_i.id_avion},{avion_j.id_avion}: "
                          f"Avión {avion_j.id_avion} (en {avion_j.t_seleccionado}) aterrizó muy pronto después del Avión {avion_i.id_avion} (en {avion_i.t_seleccionado}). "
                          f"Separación real: {separacion_real}, Requerida: {separacion_requerida_S_ij}.")
                    es_factible = False
                    aviones_con_problemas_ids.add(avion_i.id_avion)
                    aviones_con_problemas_ids.add(avion_j.id_avion)

    mapa_aviones_por_id = {avion.id_avion: avion for avion in lista_aviones_original}
    aviones_con_problemas_obj = [mapa_aviones_por_id[id_avion] for id_avion in sorted(list(aviones_con_problemas_ids))]

    print("\n--- Resumen de Factibilidad Final (Greedy Priorizar Menor E_k) ---")
    if es_factible:
        print("✅ La solución encontrada es factible.")
    else:
        print("❌ La solución encontrada NO es factible.")
        if aviones_con_problemas_obj:
            print("  Aviones involucrados en problemas:")
            for avion_p in aviones_con_problemas_obj:
                 estado = f"aterrizó en {avion_p.t_seleccionado}" if avion_p.t_seleccionado is not None else "NO FUE PROGRAMADO"
                 detalle_ventana = ""
                 if avion_p.t_seleccionado is not None and \
                    not (avion_p.tiempo_aterrizaje_temprano <= avion_p.t_seleccionado <= avion_p.tiempo_aterrizaje_tardio):
                     detalle_ventana = f" (Ventana: [{avion_p.tiempo_aterrizaje_temprano}, {avion_p.tiempo_aterrizaje_tardio}]) ¡FALLO VENTANA!"
                 print(f"    - Avión ID {avion_p.id_avion}: {estado}{detalle_ventana}")

    print("\nHorario de Aterrizaje Detallado (Greedy Priorizar Menor E_k):")
    horario_final_ordenado = sorted([a for a in lista_aviones_original if a.t_seleccionado is not None], key=lambda avion: avion.t_seleccionado if avion.t_seleccionado is not None else float('inf'))
    if horario_final_ordenado:
        for avion in horario_final_ordenado:
            print(f"  Avión ID {avion.id_avion}: aterriza en t = {avion.t_seleccionado}")
    else:
        print("  No se pudo programar ningún aterrizaje.")

    return lista_aviones_original, es_factible, aviones_con_problemas_obj
# ==============================================================================


def calcular_costo_total(lista_aviones: List[Avion]) -> float:
    """
    Calcula el costo total de la programación de aterrizajes
    basado en los tiempos seleccionados y las penalizaciones.
    Solo considera aviones que fueron efectivamente programados.
    """
    costo_total = 0.0
    print("\n--- Cálculo de Costo Total de la Programación ---")
    print("-" * 50)

    aviones_efectivamente_programados = [avion for avion in lista_aviones if avion.t_seleccionado is not None]

    if not aviones_efectivamente_programados:
        print("No hay aviones programados para calcular el costo.")
        print("-" * 50)
        print(f"Costo Total de la Programación: {costo_total:.2f}")
        return 0.0

    aviones_efectivamente_programados.sort(key=lambda avion: avion.id_avion)

    for avion in aviones_efectivamente_programados:
        if avion.t_seleccionado is not None:
            costo_avion_actual = calcular_costo_en_tiempo_especifico(avion, avion.t_seleccionado)

            tiempo_diferencia = avion.t_seleccionado - avion.tiempo_aterrizaje_preferente
            detalle_calculo = ""
            if tiempo_diferencia > 0:
                detalle_calculo = f"+{tiempo_diferencia} unidades tarde * {avion.penalizacion_despues_preferente}/unidad"
            elif tiempo_diferencia < 0:
                detalle_calculo = f"{abs(tiempo_diferencia)} unidades temprano * {avion.penalizacion_antes_preferente}/unidad"
            else:
                detalle_calculo = "Aterrizaje en tiempo preferente"

            print(f"  Avión ID {avion.id_avion:2} (Aterrizó: {avion.t_seleccionado:4}, Preferente: {avion.tiempo_aterrizaje_preferente:4}, "
                  f"Ventana: [{avion.tiempo_aterrizaje_temprano:4},{avion.tiempo_aterrizaje_tardio:4}]): "
                  f"Costo: {costo_avion_actual:8.2f} ({detalle_calculo})")
            costo_total += costo_avion_actual

    print("-" * 50)
    print(f"Costo Total de la Programación (solo aviones aterrizados): {costo_total:.2f}")
    print("--- Fin Cálculo de Costo ---")

    return costo_total

def imprimir_matriz_aterrizaje(lista_aviones_procesada: List[Avion]) -> None:
    """
    Imprime una matriz con el horario de aterrizaje de los aviones programados.
    La matriz tiene dos columnas: Tiempo de Aterrizaje y ID del Avión.
    Se ordena por tiempo de aterrizaje.
    """
    print("\n--- Matriz de Horario de Aterrizaje ---")
    print("---------------------------------------")
    print("| Tiempo Aterrizaje | ID Avión      |")
    print("|-------------------|---------------|")

    aviones_aterrizados = [avion for avion in lista_aviones_procesada if avion.t_seleccionado is not None]
    
    if not aviones_aterrizados:
        print("| Sin aterrizajes   | N/A           |")
        print("---------------------------------------")
        return

    # Ordenar por tiempo de aterrizaje
    aviones_aterrizados.sort(key=lambda avion: avion.t_seleccionado if avion.t_seleccionado is not None else float('inf'))

    for avion in aviones_aterrizados:
        # Asegurarse de que t_seleccionado no es None antes de formatear
        tiempo_str = f"{avion.t_seleccionado:<17}" if avion.t_seleccionado is not None else "Error: T None   "
        avion_id_str = f"{avion.id_avion:<13}"
        print(f"| {tiempo_str} | {avion_id_str} |")
    
    print("---------------------------------------")


def main() -> int:
    """
    Punto de entrada principal del script.
    """
    # ¡¡IMPORTANTE!! Ajusta esta ruta a la ubicación real de tu archivo.
    # Ejemplo: ruta_datos = r"C:/Users/TuUsuario/Desktop/caso3.txt"
    # O si el archivo está en el mismo directorio que el script:
    ruta_datos = r"case1.txt" 

    try:
        lista_de_aviones_original = leer_datos_aviones(ruta_datos)
        print(f"Datos leídos correctamente para {len(lista_de_aviones_original)} aviones desde '{ruta_datos}'.\n")

        imprimir_tiempos_principales(lista_de_aviones_original)

        lista_aviones_procesada, es_solucion_factible, aviones_con_problemas_final = greedy_priorizar_menor_Ek(lista_de_aviones_original)

        costo_total_obtenido = calcular_costo_total(lista_aviones_procesada)

        if not es_solucion_factible:
            print("\n*********************************************************************")
            print("RECORDATORIO: La solución generada NO fue completamente factible.")
            print(f"Número de aviones con problemas (no programados o violaciones): {len(aviones_con_problemas_final)}")
            if aviones_con_problemas_final:
                 print(f"IDs de aviones con problemas: {[a.id_avion for a in aviones_con_problemas_final]}")
            print("*********************************************************************")

        aviones_no_programados_ids = [avion.id_avion for avion in lista_aviones_procesada if avion.t_seleccionado is None]
        if not aviones_no_programados_ids:
            print("\n¡¡¡FELICIDADES!!! ¡Todos los aviones fueron programados con esta lógica!")
        else:
            print(f"\nDesafortunadamente, los siguientes aviones NO pudieron ser programados: {sorted(aviones_no_programados_ids)}")




        # --- NUEVA SECCIÓN: Imprimir la matriz de aterrizaje ---
        imprimir_matriz_aterrizaje(lista_aviones_procesada)
        # --- FIN NUEVA SECCIÓN ---

        return 0

    except FileNotFoundError:
        print(f"Error CRÍTICO: No se pudo encontrar el archivo de datos en '{ruta_datos}'. Verifica la ruta.", file=sys.stderr)
        return 1
    except ValueError as ve:
        print(f"Error CRÍTICO en los datos o formato del archivo: {ve}", file=sys.stderr)
        return 1
    except IOError as ioe:
        print(f"Error CRÍTICO de entrada/salida: {ioe}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la ejecución: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    codigo_salida_script = main()
    sys.exit(codigo_salida_script)
