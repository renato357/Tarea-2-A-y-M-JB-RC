import sys
import math
from typing import List, TextIO, Optional, Tuple

# Constante para el número de pistas
NUMERO_DE_PISTAS = 2

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
        t_seleccionado (Optional[int]): El tiempo de aterrizaje asignado.
        pista_seleccionada (Optional[int]): La pista asignada (0 o 1).
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
        self.t_seleccionado: Optional[int] = None
        self.pista_seleccionada: Optional[int] = None

    def __str__(self) -> str:
        """Representación en string del objeto Avion para impresión completa."""
        pista_str = f"Pista {self.pista_seleccionada}" if self.pista_seleccionada is not None else "N/A"
        info = (
            f"--- Avión ID: {self.id_avion} ---\n"
            f"  T. Temprano: {self.tiempo_aterrizaje_temprano}\n"
            f"  T. Preferente: {self.tiempo_aterrizaje_preferente}\n"
            f"  T. Tardío: {self.tiempo_aterrizaje_tardio}\n"
            f"  Penalización Antes: {self.penalizacion_antes_preferente}\n"
            f"  Penalización Después: {self.penalizacion_despues_preferente}\n"
            f"  Tiempo Seleccionado: {self.t_seleccionado if self.t_seleccionado is not None else 'N/A'}\n"
            f"  Pista Seleccionada: {pista_str}\n"
            f"{'-' * (20 + len(str(self.id_avion)))}"
        )
        return info

    def get_separation_to(self, other_avion: 'Avion') -> int:
        """
        Obtiene el tiempo de separación requerido *después* de que este avión (self)
        aterrice para que el otro avión (other_avion) pueda aterrizar.
        """
        # El ID del avión es 1-based, el índice de la lista es 0-based.
        other_avion_index = other_avion.id_avion - 1
        if 0 <= other_avion_index < len(self.tiempos_separacion):
            return self.tiempos_separacion[other_avion_index]
        else:
            # Esto no debería ocurrir si los datos de entrada son consistentes.
            print(f"Advertencia: ID de avión {other_avion.id_avion} fuera de rango "
                  f"para tiempos de separación del avión {self.id_avion}.", file=sys.stderr)
            return float('inf') # Retornar un valor alto para indicar problema.

# Función para leer los datos desde el archivo
def leer_datos_aviones(ruta_archivo: str) -> List[Avion]:
    """
    Lee la información de los aviones desde un archivo de texto formateado.
    No cambia respecto a la versión de 1 pista, ya que la asignación de pista
    es parte del algoritmo, no de la entrada.
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
                    # Los tiempos de separación pueden estar en múltiples líneas.
                    while valores_leidos < cantidad_aviones:
                        linea_sep = f.readline().strip().split()
                        if not linea_sep and valores_leidos < cantidad_aviones: # Línea vacía inesperada
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
        raise # Re-lanza la excepción para que sea manejada por el main.
    except (ValueError, IOError) as e: # Captura errores de formato y de E/S.
        print(f"Error procesando el archivo '{ruta_archivo}': {e}", file=sys.stderr)
        raise # Re-lanza.
    return aviones

def imprimir_tiempos_principales(lista_aviones: List[Avion]) -> None:
    """
    Imprime el ID, tiempo temprano, preferente y tardío de cada avión.
    """
    print("\nTiempos principales de aterrizaje por avión:")
    print("-" * 70)
    print(f"{'ID':<5} | {'Temprano (E_k)':<15} | {'Preferente (P_k)':<18} | {'Tardío (L_k)':<15}")
    print("-" * 70)
    for avion in lista_aviones:
        print(f"{avion.id_avion:<5} | {avion.tiempo_aterrizaje_temprano:<15} | "
              f"{avion.tiempo_aterrizaje_preferente:<18} | {avion.tiempo_aterrizaje_tardio:<15}")
    print("-" * 70)

def calcular_costo_en_tiempo_especifico(avion: Avion, tiempo_aterrizaje_propuesto: int) -> float:
    """
    Calcula el costo de penalización si el avión aterriza en un tiempo específico.
    """
    if tiempo_aterrizaje_propuesto is None: # Debería tener un tiempo asignado si se llama esta función.
        return float('inf')

    diferencia = tiempo_aterrizaje_propuesto - avion.tiempo_aterrizaje_preferente
    costo = 0.0
    if diferencia < 0: # Aterrizaje temprano
        costo = abs(diferencia) * avion.penalizacion_antes_preferente
    elif diferencia > 0: # Aterrizaje tardío
        costo = diferencia * avion.penalizacion_despues_preferente
    return costo

# ==============================================================================
#               ALGORITMO GREEDY DETERMINISTA PARA 2 PISTAS
# ==============================================================================
def greedy_determinista_2pistas(lista_aviones_original: List[Avion]) -> Tuple[List[Avion], bool, List[Avion]]:
    """
    Algoritmo greedy determinista para programar aterrizajes en dos pistas.
    En cada paso, selecciona la combinación (avión, pista, tiempo) que permite
    el aterrizaje más temprano posible, respetando todas las restricciones.
    """
    # Reseteamos t_seleccionado y pista_seleccionada por si se reutiliza la lista
    for avion_obj in lista_aviones_original:
        avion_obj.t_seleccionado = None
        avion_obj.pista_seleccionada = None

    # Copiamos la lista para trabajar sobre ella (los objetos Avion sí se modificarán)
    aviones_pendientes = lista_aviones_original[:]
    
    # Listas para llevar el registro de aviones aterrizados en cada pista
    aviones_aterrizados_por_pista: List[List[Avion]] = [[] for _ in range(NUMERO_DE_PISTAS)]
    
    # Lista para guardar todos los aviones aterrizados en orden global (para costo y verificación final)
    aviones_aterrizados_globalmente_ordenados: List[Avion] = []

    print(f"\n--- Ejecutando Algoritmo Greedy Determinista para {NUMERO_DE_PISTAS} Pistas ---")

    iteraciones_seguridad = 0
    max_iteraciones_posibles = len(lista_aviones_original) * len(lista_aviones_original) * 2 # Un límite generoso
    if not lista_aviones_original: max_iteraciones_posibles = 1


    while aviones_pendientes:
        iteraciones_seguridad += 1
        if iteraciones_seguridad > max_iteraciones_posibles:
            print("ADVERTENCIA: Límite de iteraciones de seguridad alcanzado en el bucle principal del greedy.")
            break

        mejor_opcion_actual: Optional[Tuple[Avion, int, int]] = None # (avion, pista_idx, tiempo_aterrizaje)
        
        # Criterios de ordenamiento para la tupla de evaluación:
        # (tiempo_aterrizaje_efectivo, Ek, Lk, Pk, id_avion, pista_idx)
        # Esto asegura que elegimos el aterrizaje más temprano, y luego desglosamos por criterios del avión.
        mejor_opcion_tupla_eval = (float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'))

        for avion_k in aviones_pendientes:
            for pista_idx in range(NUMERO_DE_PISTAS):
                # Calcular el tiempo más temprano posible para avion_k en pista_idx
                tiempo_aterrizaje_efectivo_en_pista = avion_k.tiempo_aterrizaje_temprano
                
                if aviones_aterrizados_por_pista[pista_idx]: # Si hay aviones ya en esta pista
                    ultimo_avion_en_pista = aviones_aterrizados_por_pista[pista_idx][-1] # El último es el más reciente
                    if ultimo_avion_en_pista.t_seleccionado is not None:
                        tiempo_aterrizaje_efectivo_en_pista = max(
                            tiempo_aterrizaje_efectivo_en_pista,
                            ultimo_avion_en_pista.t_seleccionado + ultimo_avion_en_pista.get_separation_to(avion_k)
                        )
                    else: # No debería pasar si la lógica es correcta
                        print(f"ERROR INTERNO: Último avión en pista {pista_idx} no tiene t_seleccionado.", file=sys.stderr)
                        continue


                # Verificar si este tiempo de aterrizaje es válido (dentro de la ventana Lk del avión)
                if tiempo_aterrizaje_efectivo_en_pista <= avion_k.tiempo_aterrizaje_tardio:
                    # Esta es una opción válida. Comparar con la mejor opción encontrada hasta ahora.
                    opcion_actual_tupla_eval = (
                        tiempo_aterrizaje_efectivo_en_pista,
                        avion_k.tiempo_aterrizaje_temprano,
                        avion_k.tiempo_aterrizaje_tardio,
                        avion_k.tiempo_aterrizaje_preferente,
                        avion_k.id_avion,
                        pista_idx # Preferir pista 0 en caso de empate absoluto
                    )

                    if opcion_actual_tupla_eval < mejor_opcion_tupla_eval:
                        mejor_opcion_tupla_eval = opcion_actual_tupla_eval
                        mejor_opcion_actual = (avion_k, pista_idx, tiempo_aterrizaje_efectivo_en_pista)
        
        # Si encontramos una opción para aterrizar un avión
        if mejor_opcion_actual:
            avion_elegido, pista_elegida, tiempo_elegido = mejor_opcion_actual
            
            avion_elegido.t_seleccionado = tiempo_elegido
            avion_elegido.pista_seleccionada = pista_elegida
            
            aviones_aterrizados_por_pista[pista_elegida].append(avion_elegido)
            # Es importante que la lista por pista se mantenga ordenada por tiempo si se añaden fuera de orden,
            # pero este greedy debería añadirlos en orden para cada pista. Re-ordenar por seguridad:
            aviones_aterrizados_por_pista[pista_elegida].sort(key=lambda av: av.t_seleccionado if av.t_seleccionado is not None else float('inf'))

            aviones_pendientes.remove(avion_elegido)
            aviones_aterrizados_globalmente_ordenados.append(avion_elegido)
            
            print(f"  Aterrizaje programado: Avión ID {avion_elegido.id_avion} en Pista {pista_elegida} en t = {tiempo_elegido}")
        else:
            # No se pudo encontrar ninguna opción válida para ningún avión pendiente.
            # Esto significa que los aviones restantes no pueden aterrizar o hay un error.
            if aviones_pendientes: # Solo si aún quedan aviones
                print(f"ADVERTENCIA: No se pudo programar ningún avión de los {len(aviones_pendientes)} pendientes. "
                      "Esto puede indicar una solución infactible o un problema en la lógica.")
            break # Salir del bucle principal

    print("\n--- Algoritmo Greedy Determinista para 2 Pistas Finalizado ---")

    # Ordenar la lista global de aterrizados por tiempo para la verificación final y costo
    aviones_aterrizados_globalmente_ordenados.sort(key=lambda av: av.t_seleccionado if av.t_seleccionado is not None else float('inf'))

    # --- Verificación de Factibilidad ---
    es_factible = True
    aviones_con_problemas_ids = set()

    # 1. Verificar si todos los aviones fueron programados
    aviones_no_programados_final = [avion for avion in lista_aviones_original if avion.t_seleccionado is None or avion.pista_seleccionada is None]
    if aviones_no_programados_final:
        es_factible = False
        print(f"  ❌ No todos los aviones ({len(aviones_no_programados_final)}) pudieron ser programados completamente.")
        for avion in aviones_no_programados_final:
            print(f"    Avión ID {avion.id_avion}: No fue programado o no se le asignó pista/tiempo.")
            aviones_con_problemas_ids.add(avion.id_avion)

    # 2. Verificar ventanas de tiempo E_k, L_k para los programados
    for avion in aviones_aterrizados_globalmente_ordenados: # Usar la lista global que tiene los programados
        if avion.t_seleccionado is None: continue # Ya cubierto arriba, pero por si acaso
        if not (avion.tiempo_aterrizaje_temprano <= avion.t_seleccionado <= avion.tiempo_aterrizaje_tardio):
            print(f"    ❌ Avión ID {avion.id_avion}: Tiempo asignado {avion.t_seleccionado} está FUERA de su ventana [{avion.tiempo_aterrizaje_temprano}, {avion.tiempo_aterrizaje_tardio}].")
            es_factible = False
            aviones_con_problemas_ids.add(avion.id_avion)

    # 3. Verificar separaciones en cada pista
    for pista_idx in range(NUMERO_DE_PISTAS):
        # La lista aviones_aterrizados_por_pista[pista_idx] ya debería estar ordenada por t_seleccionado
        # gracias al sort dentro del bucle o por la naturaleza del greedy.
        # Si no, se debe ordenar aquí:
        # aviones_en_esta_pista = sorted([a for a in lista_aviones_original if a.pista_seleccionada == pista_idx and a.t_seleccionado is not None], key=lambda x: x.t_seleccionado)
        aviones_en_esta_pista = aviones_aterrizados_por_pista[pista_idx]

        for i in range(len(aviones_en_esta_pista) - 1):
            avion_i = aviones_en_esta_pista[i]
            avion_j = aviones_en_esta_pista[i+1]
            
            # Asegurarse que ambos aviones tienen tiempos válidos (ya deberían por chequeos previos)
            if avion_i.t_seleccionado is None or avion_j.t_seleccionado is None:
                print(f"ERROR INTERNO: Aviones en pista {pista_idx} con t_seleccionado None durante chequeo de separación.", file=sys.stderr)
                es_factible = False
                aviones_con_problemas_ids.add(avion_i.id_avion)
                aviones_con_problemas_ids.add(avion_j.id_avion)
                continue

            separacion_requerida_S_ij = avion_i.get_separation_to(avion_j)
            separacion_real = avion_j.t_seleccionado - avion_i.t_seleccionado
            
            if separacion_real < separacion_requerida_S_ij:
                print(f"    ❌ VIOLACIÓN DE SEPARACIÓN en Pista {pista_idx} entre S_{avion_i.id_avion},{avion_j.id_avion}: "
                      f"Avión {avion_j.id_avion} (en {avion_j.t_seleccionado}) aterrizó muy pronto después del Avión {avion_i.id_avion} (en {avion_i.t_seleccionado}). "
                      f"Separación real: {separacion_real}, Requerida: {separacion_requerida_S_ij}.")
                es_factible = False
                aviones_con_problemas_ids.add(avion_i.id_avion)
                aviones_con_problemas_ids.add(avion_j.id_avion)

    mapa_aviones_por_id = {avion.id_avion: avion for avion in lista_aviones_original}
    aviones_con_problemas_obj = [mapa_aviones_por_id[id_avion] for id_avion in sorted(list(aviones_con_problemas_ids))]

    print("\n--- Resumen de Factibilidad Final (Greedy 2 Pistas) ---")
    if es_factible:
        print("✅ La solución encontrada es factible.")
    else:
        print("❌ La solución encontrada NO es factible.")
        if aviones_con_problemas_obj:
            print("  Aviones involucrados en problemas:")
            for avion_p in aviones_con_problemas_obj:
                 estado = f"aterrizó en t={avion_p.t_seleccionado} en Pista {avion_p.pista_seleccionada}" \
                          if avion_p.t_seleccionado is not None and avion_p.pista_seleccionada is not None \
                          else "NO FUE PROGRAMADO CORRECTAMENTE"
                 detalle_ventana = ""
                 if avion_p.t_seleccionado is not None and \
                    not (avion_p.tiempo_aterrizaje_temprano <= avion_p.t_seleccionado <= avion_p.tiempo_aterrizaje_tardio):
                     detalle_ventana = f" (Ventana: [{avion_p.tiempo_aterrizaje_temprano}, {avion_p.tiempo_aterrizaje_tardio}]) ¡FALLO VENTANA!"
                 print(f"    - Avión ID {avion_p.id_avion}: {estado}{detalle_ventana}")

    # Devolvemos la lista original de aviones (que ha sido modificada con t_seleccionado y pista_seleccionada),
    # el estado de factibilidad y la lista de objetos avión con problemas.
    return lista_aviones_original, es_factible, aviones_con_problemas_obj
# ==============================================================================


def calcular_costo_total(lista_aviones: List[Avion]) -> float:
    """
    Calcula el costo total de la programación de aterrizajes.
    Solo considera aviones que fueron efectivamente programados con un tiempo.
    La pista no afecta el cálculo del costo según la definición del problema.
    """
    costo_total = 0.0
    print("\n--- Cálculo de Costo Total de la Programación (2 Pistas) ---")
    print("-" * 70)
    print(f"{'ID':<5} | {'Pista':<7} | {'T.Aterrizaje':<12} | {'T.Preferente':<12} | {'Costo Avión':<12} | {'Detalle'}")
    print("-" * 70)

    # Ordenar por ID para una presentación consistente del desglose de costos
    aviones_para_costo = sorted([avion for avion in lista_aviones if avion.t_seleccionado is not None], key=lambda a: a.id_avion)

    if not aviones_para_costo:
        print("No hay aviones programados para calcular el costo.")
        print("-" * 70)
        print(f"Costo Total de la Programación: {costo_total:.2f}")
        return 0.0

    for avion in aviones_para_costo:
        # t_seleccionado ya no debería ser None aquí debido al filtro anterior
        costo_avion_actual = calcular_costo_en_tiempo_especifico(avion, avion.t_seleccionado) # type: ignore

        tiempo_diferencia = avion.t_seleccionado - avion.tiempo_aterrizaje_preferente # type: ignore
        detalle_calculo = ""
        if tiempo_diferencia > 0:
            detalle_calculo = f"+{tiempo_diferencia} tarde * {avion.penalizacion_despues_preferente}/u"
        elif tiempo_diferencia < 0:
            detalle_calculo = f"{abs(tiempo_diferencia)} temp. * {avion.penalizacion_antes_preferente}/u"
        else:
            detalle_calculo = "En tiempo preferente"
        
        pista_str = str(avion.pista_seleccionada) if avion.pista_seleccionada is not None else "N/A"
        print(f"{avion.id_avion:<5} | {pista_str:<7} | {avion.t_seleccionado:<12} | "
              f"{avion.tiempo_aterrizaje_preferente:<12} | {costo_avion_actual:<12.2f} | {detalle_calculo}")
        costo_total += costo_avion_actual

    print("-" * 70)
    print(f"Costo Total de la Programación (solo aviones aterrizados): {costo_total:.2f}")
    print("--- Fin Cálculo de Costo ---")
    return costo_total

def imprimir_matriz_aterrizaje(lista_aviones_procesada: List[Avion]) -> None:
    """
    Imprime una matriz con el horario de aterrizaje de los aviones programados,
    incluyendo la pista asignada. Se ordena por tiempo de aterrizaje.
    """
    print("\n--- Matriz de Horario de Aterrizaje (2 Pistas) ---")
    print("----------------------------------------------------")
    print(f"| {'Tiempo Aterrizaje':<18} | {'ID Avión':<10} | {'Pista':<7} |")
    print("|--------------------|------------|---------|")

    # Filtrar solo aviones que realmente aterrizaron y tienen pista asignada
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


def main() -> int:
    """
    Punto de entrada principal del script.
    """
    # ¡¡IMPORTANTE!! Ajusta esta ruta a la ubicación real de tu archivo.
    # Ejemplo: ruta_datos = r"C:/Users/TuUsuario/Desktop/caso1.txt"
    # O si el archivo está en el mismo directorio que el script:
    ruta_datos = r"case1.txt" # Puedes cambiar esto a case1.txt, case1.txt, case1.txt

    try:
        lista_de_aviones_original = leer_datos_aviones(ruta_datos)
        print(f"Datos leídos correctamente para {len(lista_de_aviones_original)} aviones desde '{ruta_datos}'.\n")

        imprimir_tiempos_principales(lista_de_aviones_original)

        # Ejecutar el algoritmo greedy para 2 pistas
        lista_aviones_procesada, es_solucion_factible, aviones_con_problemas_final = \
            greedy_determinista_2pistas(lista_de_aviones_original)

        costo_total_obtenido = calcular_costo_total(lista_aviones_procesada)

        if not es_solucion_factible:
            print("\n*********************************************************************")
            print("RECORDATORIO: La solución generada NO fue completamente factible.")
            print(f"Número de aviones con problemas (no programados o violaciones): {len(aviones_con_problemas_final)}")
            if aviones_con_problemas_final:
                 print(f"IDs de aviones con problemas: {[a.id_avion for a in aviones_con_problemas_final]}")
            print("*********************************************************************")

        aviones_no_programados_ids = [
            avion.id_avion for avion in lista_aviones_procesada 
            if avion.t_seleccionado is None or avion.pista_seleccionada is None
        ]
        if not aviones_no_programados_ids:
            print("\n¡¡¡FELICIDADES!!! ¡Todos los aviones fueron programados con esta lógica de 2 pistas!")
        else:
            print(f"\nDesafortunadamente, los siguientes aviones NO pudieron ser programados completamente: {sorted(aviones_no_programados_ids)}")

        imprimir_matriz_aterrizaje(lista_aviones_procesada)
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
