import sys
import math
import random
import copy # Para deepcopy
from typing import List, TextIO, Optional, Tuple

# Constante para el número de pistas
NUMERO_DE_PISTAS = 2
EPSILON = 0.001 # Pequeño valor para evitar división por cero en ponderación de costos
K_MAX_SLACK_TIEMPO = 5 # NUEVA CONSTANTE: Máximo slack aleatorio para el tiempo de aterrizaje

# Definición de la clase para representar cada avión
class Avion:
    """
    Representa un avión con sus datos de aterrizaje y separación.
    Incluye pista_seleccionada para el escenario de múltiples pistas.
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
        self.pista_seleccionada: Optional[int] = None # 0 o 1 para dos pistas

    def __str__(self) -> str:
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
        other_avion_index = other_avion.id_avion - 1
        if 0 <= other_avion_index < len(self.tiempos_separacion):
            return self.tiempos_separacion[other_avion_index]
        print(f"Advertencia: ID de avión {other_avion.id_avion} fuera de rango para T_sep del avión {self.id_avion}.", file=sys.stderr)
        return float('inf')

# Funciones de utilidad (lectura, costos, impresión) adaptadas para 2 pistas

def leer_datos_aviones(ruta_archivo: str) -> List[Avion]:
    """Lee la información de los aviones desde un archivo."""
    aviones: List[Avion] = []
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            cantidad_aviones = int(f.readline().strip())
            if cantidad_aviones <= 0:
                raise ValueError("La cantidad de aviones debe ser positiva.")
            for i in range(1, cantidad_aviones + 1):
                partes_tiempos = f.readline().strip().split()
                if len(partes_tiempos) != 5:
                    raise ValueError(f"Formato incorrecto para avión {i} (tiempos/penalizaciones).")
                t_temprano, t_preferente, t_tardio = map(int, partes_tiempos[:3])
                pen_antes, pen_despues = map(float, partes_tiempos[3:])
                
                tiempos_sep: List[int] = []
                valores_leidos = 0
                while valores_leidos < cantidad_aviones:
                    linea_sep = f.readline().strip().split()
                    if not linea_sep and valores_leidos < cantidad_aviones:
                        raise ValueError(f"Línea vacía inesperada para T_sep avión {i}.")
                    tiempos_sep.extend(int(t) for t in linea_sep)
                    valores_leidos += len(linea_sep)
                if len(tiempos_sep) != cantidad_aviones:
                    raise ValueError(f"Número incorrecto de T_sep para avión {i}.")
                
                aviones.append(Avion(i, t_temprano, t_preferente, t_tardio, pen_antes, pen_despues, tiempos_sep))
            if len(aviones) != cantidad_aviones:
                raise ValueError("No se procesaron todos los aviones esperados.")
    except FileNotFoundError:
        print(f"Error: Archivo '{ruta_archivo}' no encontrado.", file=sys.stderr)
        raise
    except (ValueError, IOError) as e:
        print(f"Error procesando archivo '{ruta_archivo}': {e}", file=sys.stderr)
        raise
    return aviones

def calcular_costo_en_tiempo_especifico(avion: Avion, tiempo_aterrizaje_propuesto: int) -> float:
    """Calcula el costo de penalización para un avión en un tiempo dado."""
    if tiempo_aterrizaje_propuesto is None: return float('inf')
    diferencia = tiempo_aterrizaje_propuesto - avion.tiempo_aterrizaje_preferente
    costo = 0.0
    if diferencia < 0:
        costo = abs(diferencia) * avion.penalizacion_antes_preferente
    elif diferencia > 0:
        costo = diferencia * avion.penalizacion_despues_preferente
    return costo

def _cumple_separacion_en_pista(avion_candidato: Avion, tiempo_aterrizaje: int, pista_idx: int, aviones_aterrizados_por_pista: List[List[Avion]]) -> bool:
    """Verifica si se cumple la separación para aterrizar en una pista específica."""
    aviones_en_esta_pista = aviones_aterrizados_por_pista[pista_idx]
    if not aviones_en_esta_pista:
        return True # Primera vez en esta pista
    ultimo_avion_en_pista = aviones_en_esta_pista[-1] # El último es el más reciente
    if ultimo_avion_en_pista.t_seleccionado is None: # No debería pasar
        print("Error interno: Ultimo avión en pista sin t_seleccionado.", file=sys.stderr)
        return False 
    
    separacion_requerida = ultimo_avion_en_pista.get_separation_to(avion_candidato)
    return tiempo_aterrizaje >= ultimo_avion_en_pista.t_seleccionado + separacion_requerida

def _calcular_tiempo_min_aterrizaje_en_pista_base(avion_k: Avion, pista_idx: int, aviones_aterrizados_por_pista: List[List[Avion]]) -> int:
    """
    Calcula el tiempo más temprano que avion_k puede aterrizar en pista_idx,
    considerando su E_k y la separación con el último avión en esa pista.
    Este es el tiempo base ANTES de aplicar cualquier slack aleatorio.
    """
    tiempo_efectivo = avion_k.tiempo_aterrizaje_temprano
    aviones_en_esta_pista = aviones_aterrizados_por_pista[pista_idx]
    if aviones_en_esta_pista:
        ultimo_avion = aviones_en_esta_pista[-1]
        if ultimo_avion.t_seleccionado is not None:
             tiempo_efectivo = max(tiempo_efectivo, ultimo_avion.t_seleccionado + ultimo_avion.get_separation_to(avion_k))
    return tiempo_efectivo

# ==============================================================================
#               ALGORITMO GREEDY ESTOCÁSTICO PARA 2 PISTAS (MÁS RANDOM)
# ==============================================================================
def greedy_estocastico_2pistas_mas_random(
    lista_aviones_para_ejecucion: List[Avion], 
    verbose_internal: bool = False
    ) -> Tuple[List[Avion], bool, List[Avion]]:
    """
    Algoritmo greedy estocástico para programar aterrizajes en dos pistas,
    con aleatoriedad adicional en el orden de evaluación y en el tiempo de aterrizaje.
    Modifica y devuelve la lista_aviones_para_ejecucion.
    """
    for avion_obj in lista_aviones_para_ejecucion:
        avion_obj.t_seleccionado = None
        avion_obj.pista_seleccionada = None

    aviones_pendientes = lista_aviones_para_ejecucion[:]
    aviones_aterrizados_por_pista: List[List[Avion]] = [[] for _ in range(NUMERO_DE_PISTAS)]
    
    if not aviones_pendientes:
        return lista_aviones_para_ejecucion, True, []

    tiempo_actual: int = min(avion.tiempo_aterrizaje_temprano for avion in aviones_pendientes)
    
    iteracion_actual_seguridad = 0
    max_tiempo_tardio_global = max(a.tiempo_aterrizaje_tardio for a in aviones_pendientes) if aviones_pendientes else 0
    max_iteraciones_seguridad = len(aviones_pendientes) * (max_tiempo_tardio_global + len(aviones_pendientes) * 200 + 1000) # Aumentado margen
    if not aviones_pendientes: max_iteraciones_seguridad = 1

    if verbose_internal: print(f"--- Iniciando Greedy Estocástico 2 Pistas (Más Random). T inicial: {tiempo_actual}, Max iter seg: {max_iteraciones_seguridad}, K_MAX_SLACK: {K_MAX_SLACK_TIEMPO} ---")

    while aviones_pendientes and iteracion_actual_seguridad < max_iteraciones_seguridad:
        iteracion_actual_seguridad += 1
        avion_programado_en_iteracion = False
        
        # 1. Barajar aviones pendientes para cambiar el orden de evaluación
        random.shuffle(aviones_pendientes) # NUEVO

        if verbose_internal: 
            print(f"\n  Iter. Estoc. {iteracion_actual_seguridad}, T_base={tiempo_actual}, Pendientes (barajados): {[a.id_avion for a in aviones_pendientes]}")
            for p_idx in range(NUMERO_DE_PISTAS):
                ids_pista = [a.id_avion for a in aviones_aterrizados_por_pista[p_idx]]
                if ids_pista: print(f"    Pista {p_idx}: Aterrizados IDs {ids_pista}, Último t={aviones_aterrizados_por_pista[p_idx][-1].t_seleccionado}")

        # 2. Construir lista de candidatos con tiempo flexible
        #    Un candidato es (avion, pista_idx, tiempo_propuesto_aterrizaje)
        candidatos_flexibles: List[Tuple[Avion, int, int]] = []
        
        for avion_k in aviones_pendientes: # Ahora iteramos sobre la lista barajada
            for pista_idx in range(NUMERO_DE_PISTAS):
                # Considerar si avion_k puede aterrizar en pista_idx A PARTIR de tiempo_actual
                # Primero, el avión debe estar listo (Ek <= tiempo_actual)
                if avion_k.tiempo_aterrizaje_temprano > tiempo_actual:
                    continue

                # Segundo, debe cumplir separación si aterriza en tiempo_actual (como base)
                if not _cumple_separacion_en_pista(avion_k, tiempo_actual, pista_idx, aviones_aterrizados_por_pista):
                    continue
                
                # Si puede aterrizar en tiempo_actual (base), consideramos un slack
                # El tiempo_base para esta opción es tiempo_actual.
                # El slack se suma a este tiempo_base.
                
                # Calcular slack máximo permitido sin exceder Lk y K_MAX_SLACK_TIEMPO
                max_slack_lk = avion_k.tiempo_aterrizaje_tardio - tiempo_actual
                if max_slack_lk < 0: # No debería ocurrir si tiempo_actual <= Lk (que no se chequea explícitamente aquí, pero Ek sí)
                                   # Se asume que si Ek <= tiempo_actual, y _cumple_separacion, entonces tiempo_actual es un inicio viable.
                                   # El chequeo de Lk se hace después de añadir el slack.
                    max_slack_lk = 0 
                                
                max_slack_permitido_final = min(max_slack_lk, K_MAX_SLACK_TIEMPO)
                if max_slack_permitido_final < 0 : max_slack_permitido_final = 0 # Asegurar no negativo
                                
                slack_aleatorio = random.randint(0, max_slack_permitido_final)
                tiempo_propuesto_final = tiempo_actual + slack_aleatorio
                
                # Asegurarse de que el tiempo propuesto final sigue siendo válido (no excede Lk)
                if tiempo_propuesto_final <= avion_k.tiempo_aterrizaje_tardio:
                     candidatos_flexibles.append((avion_k, pista_idx, tiempo_propuesto_final))

        if verbose_internal:
            if candidatos_flexibles:
                print(f"    Candidatos flexibles (av,p,t_prop) para T_base={tiempo_actual}: "
                      f"{[(c[0].id_avion, c[1], c[2]) for c in candidatos_flexibles]}")
            else:
                print(f"    No hay candidatos flexibles directos para T_base={tiempo_actual}.")

        # 3. Selección estocástica de los candidatos flexibles
        if candidatos_flexibles:
            avion_elegido_final: Optional[Avion] = None
            pista_elegida_final: Optional[int] = None
            tiempo_elegido_final: Optional[int] = None

            if len(candidatos_flexibles) == 1:
                avion_elegido_final, pista_elegida_final, tiempo_elegido_final = candidatos_flexibles[0]
                if verbose_internal: print(f"      Selección única de candidato flexible: Avión ID {avion_elegido_final.id_avion} en Pista {pista_elegida_final} en t={tiempo_elegido_final}.")
            else:
                costos_candidatos = [
                    calcular_costo_en_tiempo_especifico(cand[0], cand[2]) 
                    for cand in candidatos_flexibles
                ]
                pesos = [1.0 / (cost + EPSILON) for cost in costos_candidatos]
                
                sum_pesos = sum(pesos)
                if sum_pesos < EPSILON / 100.0 or len(set(pesos)) == 1:
                    if verbose_internal: print(f"      Selección aleatoria uniforme (pesos bajos/iguales) entre {len(candidatos_flexibles)} candidatos flexibles.")
                    idx_elegido = random.randrange(len(candidatos_flexibles))
                    avion_elegido_final, pista_elegida_final, tiempo_elegido_final = candidatos_flexibles[idx_elegido]
                else:
                    try:
                        if verbose_internal: print(f"      Selección ponderada por costo entre {len(candidatos_flexibles)} candidatos flexibles. Pesos: {['{:.3f}'.format(p) for p in pesos]}")
                        elegido_completo = random.choices(candidatos_flexibles, weights=pesos, k=1)[0]
                        avion_elegido_final, pista_elegida_final, tiempo_elegido_final = elegido_completo
                    except ValueError: 
                        if verbose_internal: print(f"      Error en ponderación de candidatos flexibles, usando selección aleatoria uniforme.")
                        idx_elegido = random.randrange(len(candidatos_flexibles))
                        avion_elegido_final, pista_elegida_final, tiempo_elegido_final = candidatos_flexibles[idx_elegido]
            
            if avion_elegido_final and pista_elegida_final is not None and tiempo_elegido_final is not None:
                # La condición de separación ya fue verificada para tiempo_actual.
                # Como tiempo_elegido_final >= tiempo_actual, la separación se mantiene.
                # La condición de Lk también se verificó al añadir a candidatos_flexibles.
                avion_elegido_final.t_seleccionado = tiempo_elegido_final
                avion_elegido_final.pista_seleccionada = pista_elegida_final
                
                aviones_aterrizados_por_pista[pista_elegida_final].append(avion_elegido_final)
                aviones_aterrizados_por_pista[pista_elegida_final].sort(key=lambda av: av.t_seleccionado if av.t_seleccionado is not None else float('inf'))
                
                aviones_pendientes.remove(avion_elegido_final)
                avion_programado_en_iteracion = True
                if verbose_internal: print(f"    PROGRAMADO: Avión ID {avion_elegido_final.id_avion} en Pista {pista_elegida_final} en t={tiempo_elegido_final}")

        if not aviones_pendientes:
            if verbose_internal: print("    ¡Todos los aviones han sido programados!")
            break

        # 4. Avance de tiempo_actual (si no se programó nada en esta iteración)
        if not avion_programado_en_iteracion:
            proximo_tiempo_minimo_posible_aterrizaje_global = float('inf')
            if aviones_pendientes:
                # Para el avance de tiempo, consideramos el tiempo base más temprano (sin slack)
                for avion_k_pendiente in aviones_pendientes: # Usar la lista actual barajada está bien aquí, o una ordenada por E_k para ser más "greedy" en el avance.
                    for pista_idx_pendiente in range(NUMERO_DE_PISTAS):
                        t_min_avion_pista_base = _calcular_tiempo_min_aterrizaje_en_pista_base(
                            avion_k_pendiente, pista_idx_pendiente, aviones_aterrizados_por_pista
                        )
                        if t_min_avion_pista_base <= avion_k_pendiente.tiempo_aterrizaje_tardio:
                            proximo_tiempo_minimo_posible_aterrizaje_global = min(
                                proximo_tiempo_minimo_posible_aterrizaje_global, 
                                t_min_avion_pista_base
                            )
            
            if proximo_tiempo_minimo_posible_aterrizaje_global == float('inf'):
                if aviones_pendientes and verbose_internal:
                    print(f"    ADVERTENCIA: No se pudo programar y no se encontró un próximo tiempo de evento claro (base). "
                          f"Aviones pendientes: {[a.id_avion for a in aviones_pendientes]}. Deteniendo.")
                break 
            
            if proximo_tiempo_minimo_posible_aterrizaje_global > tiempo_actual:
                if verbose_internal: print(f"    Ningún avión pudo aterrizar. Avanzando T_base de {tiempo_actual} a {proximo_tiempo_minimo_posible_aterrizaje_global}.")
                tiempo_actual = proximo_tiempo_minimo_posible_aterrizaje_global
            else: 
                if verbose_internal: print(f"    Ningún avión pudo aterrizar en T_base={tiempo_actual}. Avanzando T_base en +1.")
                tiempo_actual += 1
        
        if not avion_programado_en_iteracion and aviones_pendientes:
            max_lk_pendiente_actual = max(ap.tiempo_aterrizaje_tardio for ap in aviones_pendientes) if aviones_pendientes else tiempo_actual
            if tiempo_actual > max_lk_pendiente_actual + (len(aviones_pendientes) * 75) : 
                if verbose_internal: print(f"ADVERTENCIA INTERNA: T_base={tiempo_actual} ha superado L_k max pendiente ({max_lk_pendiente_actual}). Forzando salida.")
                break

    if iteracion_actual_seguridad >= max_iteraciones_seguridad and aviones_pendientes:
        if verbose_internal: print(f"\nAdvertencia (interno): Límite de iteraciones ({max_iteraciones_seguridad}) alcanzado. "
              f"Quedan {len(aviones_pendientes)} aviones sin programar: {[a.id_avion for a in aviones_pendientes]}")

    # --- Verificación de Factibilidad Final para esta ejecución ---
    es_factible_final = True
    aviones_con_problemas_ids = set()
    
    aviones_no_programados_final = [avion for avion in lista_aviones_para_ejecucion if avion.t_seleccionado is None or avion.pista_seleccionada is None]
    if aviones_no_programados_final:
        es_factible_final = False
        for avion in aviones_no_programados_final: aviones_con_problemas_ids.add(avion.id_avion)

    aviones_si_programados_global = [avion for avion in lista_aviones_para_ejecucion if avion.t_seleccionado is not None and avion.pista_seleccionada is not None]
    for avion in aviones_si_programados_global:
        if not (avion.tiempo_aterrizaje_temprano <= avion.t_seleccionado <= avion.tiempo_aterrizaje_tardio): # type: ignore
            es_factible_final = False
            aviones_con_problemas_ids.add(avion.id_avion)

    for pista_idx_verif in range(NUMERO_DE_PISTAS):
        aviones_en_pista_verif = sorted(
            [a for a in lista_aviones_para_ejecucion if a.pista_seleccionada == pista_idx_verif and a.t_seleccionado is not None],
            key=lambda x: x.t_seleccionado # type: ignore
        )
        for i in range(len(aviones_en_pista_verif) - 1):
            avion_i = aviones_en_pista_verif[i]
            avion_j = aviones_en_pista_verif[i+1]
            separacion_requerida_S_ij = avion_i.get_separation_to(avion_j)
            separacion_real = avion_j.t_seleccionado - avion_i.t_seleccionado # type: ignore
            if separacion_real < separacion_requerida_S_ij:
                es_factible_final = False
                aviones_con_problemas_ids.add(avion_i.id_avion)
                aviones_con_problemas_ids.add(avion_j.id_avion)

    mapa_aviones_original_por_id = {avion.id_avion: avion for avion in lista_aviones_para_ejecucion}
    aviones_con_problemas_obj = [mapa_aviones_original_por_id[id_val] for id_val in sorted(list(aviones_con_problemas_ids))]
    
    if verbose_internal:
        print("\n--- Resumen de Factibilidad Interna (Greedy Estocástico 2 Pistas Más Random) ---")
        if es_factible_final: print("✅ Solución interna parece factible.")
        else: 
            print("❌ Solución interna NO es factible.")
            if aviones_con_problemas_obj: print(f"  Aviones con problemas: {[a.id_avion for a in aviones_con_problemas_obj]}")

    return lista_aviones_para_ejecucion, es_factible_final, aviones_con_problemas_obj
# ==============================================================================

def calcular_costo_total_2pistas(lista_aviones: List[Avion]) -> float:
    """Calcula el costo total de la programación para 2 pistas."""
    costo_total = 0.0
    for avion in lista_aviones:
        if avion.t_seleccionado is not None: 
            costo_total += calcular_costo_en_tiempo_especifico(avion, avion.t_seleccionado)
    return costo_total

def imprimir_matriz_aterrizaje_2pistas(lista_aviones_procesada: List[Avion], cabecera: str) -> None:
    """Imprime la matriz de aterrizaje para 2 pistas."""
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
        aviones_aterrizados.sort(key=lambda avion: (avion.t_seleccionado, avion.pista_seleccionada, avion.id_avion)) # type: ignore
        for avion in aviones_aterrizados:
            tiempo_str = f"{avion.t_seleccionado:<18}"
            avion_id_str = f"{avion.id_avion:<10}"
            pista_str = f"{avion.pista_seleccionada:<7}"
            print(f"| {tiempo_str} | {avion_id_str} | {pista_str} |")
    print("----------------------------------------------------")

def main() -> None:
    """
    Función principal: Lee datos, ejecuta el greedy estocástico para 2 pistas
    10 veces con semillas predefinidas y reporta resultados.
    """
    ruta_datos = "case3.txt" # Cambiar según el caso de prueba
    semillas_predefinidas = [42, 123, 7, 99, 500, 777, 2024, 1, 100, 314]
    num_ejecuciones = len(semillas_predefinidas)

    print(f"Iniciando {num_ejecuciones} ejecuciones de Greedy Estocástico para 2 Pistas (Más Random)...")
    print(f"Archivo de datos: {ruta_datos}, K_MAX_SLACK_TIEMPO: {K_MAX_SLACK_TIEMPO}\n")

    try:
        lista_de_aviones_base = leer_datos_aviones(ruta_datos)
        print(f"Datos leídos para {len(lista_de_aviones_base)} aviones.\n")

        resultados_globales = [] 

        for i in range(num_ejecuciones):
            semilla_actual = semillas_predefinidas[i]
            print(f"\n--- Ejecución Estocástica 2 Pistas (Más Random) #{i+1}/{num_ejecuciones} (Semilla: {semilla_actual}) ---")
            random.seed(semilla_actual)
            
            aviones_para_esta_semilla = copy.deepcopy(lista_de_aviones_base)

            solucion_semilla, factible_semilla, problemas_semilla = greedy_estocastico_2pistas_mas_random(
                aviones_para_esta_semilla, 
                verbose_internal=False # Poner True para depuración detallada
            )
            
            costo_semilla = float('inf')
            if factible_semilla:
                costo_semilla = calcular_costo_total_2pistas(solucion_semilla)
                print(f"  Solución factible encontrada para semilla {semilla_actual}.")
                print(f"  Costo de esta solución: {costo_semilla:.2f}")
            else:
                print(f"  No se encontró solución completamente factible para semilla {semilla_actual}.")
                if problemas_semilla:
                    print(f"    Aviones con problemas: {[p.id_avion for p in problemas_semilla]}")
                costo_parcial = calcular_costo_total_2pistas(solucion_semilla) # Costo de lo que se pudo programar
                print(f"    Costo (parcial/infactible) de lo programado: {costo_parcial:.2f}")

            imprimir_matriz_aterrizaje_2pistas(solucion_semilla, f"Horario para Semilla {semilla_actual} (Más Random)")
            resultados_globales.append((semilla_actual, costo_semilla if factible_semilla else float('inf'), factible_semilla, solucion_semilla))
        
        print(f"\n\n--- Resumen de {num_ejecuciones} Ejecuciones Estocásticas para 2 Pistas (Más Random) ---")
        num_exitos_factibles = 0
        costos_factibles = []
        
        print("\nDetalle por Semilla:")
        print("-" * 60)
        print(f"{'Semilla':<10} | {'Factible?':<10} | {'Costo Total':<15}")
        print("-" * 60)
        for semilla, costo, factible, _ in resultados_globales:
            factible_str = "Sí" if factible else "No"
            costo_str = f"{costo:.2f}" if factible and costo != float('inf') else "Inf (o N/A)"
            print(f"{semilla:<10} | {factible_str:<10} | {costo_str:<15}")
            if factible:
                num_exitos_factibles += 1
                costos_factibles.append(costo)
        print("-" * 60)

        print(f"\nNúmero de soluciones factibles encontradas: {num_exitos_factibles} de {num_ejecuciones}")
        if costos_factibles:
            print(f"Costo mínimo de las soluciones factibles: {min(costos_factibles):.2f}")
            print(f"Costo promedio de las soluciones factibles: {sum(costos_factibles)/len(costos_factibles):.2f}")
            print(f"Costo máximo de las soluciones factibles: {max(costos_factibles):.2f}")
            # Contar costos distintos
            costos_distintos = sorted(list(set(costos_factibles)))
            print(f"Número de costos distintos encontrados: {len(costos_distintos)}")
            if len(costos_distintos) <= 20: # Imprimir si no son demasiados
                 print(f"Costos distintos: {[f'{c:.2f}' for c in costos_distintos]}")
        else:
            print("No se encontraron soluciones factibles para calcular estadísticas de costos.")

    except FileNotFoundError:
        print(f"Error CRÍTICO: No se pudo encontrar el archivo de datos en '{ruta_datos}'.", file=sys.stderr)
    except (ValueError, IOError) as e:
        print(f"Error CRÍTICO procesando datos: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Ocurrió un error inesperado en la ejecución principal: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
