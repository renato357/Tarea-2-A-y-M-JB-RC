# Tarea 2: Algoritmos Exactos y Metaheurísticas - Secuenciación de Aterrizajes (ALP)

Este repositorio contiene las implementaciones para la Tarea 2 del curso CIT3352, enfocada en resolver el Problema de Secuenciación de Aterrizajes de Aeronaves (ALP) mediante diversas estrategias algorítmicas. Se consideran escenarios con 1 y 2 pistas de aterrizaje y se utilizan 4 casos de prueba (`case1.txt` a `case4.txt`).

## Algoritmos Implementados

Se desarrollaron y evaluaron los siguientes algoritmos y sus variantes:

1.  **Greedy:**
    * Determinista
    * Estocástico (10 ejecuciones por semilla)
2.  **GRASP + Hill Climbing (Alguna Mejora):**
    * Base Greedy Determinista
    * Base Greedy Estocástico (con componente de Restart ILS)
3.  **Simulated Annealing (SA):**
    * Partida Greedy Determinista (5 configuraciones de parámetros)
    * Partida Greedy Estocástico (5 configs. sobre 10 soluciones estocásticas)

Cada algoritmo se aplicó a escenarios de 1 y 2 pistas.

## Estructura del Repositorio

El repositorio se organiza en carpetas principales según la actividad (1: Greedy, 2: GRASP, 3: SA), la variante del algoritmo (determinista, estocástico) y el número de pistas (1pista, 2pistas).

* **Carpetas de Algoritmos (`[actividad]-[variante]-[pistas]/`):**
    * Contienen los scripts de Python (`.py`) de cada algoritmo.
    * **Salidas Deterministas/Simples:** Generalmente incluyen archivos de texto (`.txt`) con los resultados y, para GRASP/SA determinista, gráficos (`.png`) de convergencia por caso.
    * **Salidas Estocásticas/Múltiples Ejecuciones:**
        * Para *Greedy Estocástico*: Múltiples archivos de salida (`.txt`) por caso (uno por semilla).
        * Para *GRASP Estocástico* y *SA Estocástico*: Suelen tener subcarpetas por caso (`case1/`, `case2/`, etc.). Cada una contiene múltiples gráficos (`.png`) y archivos de salida (`.txt`) correspondientes a las diferentes semillas y/o configuraciones.

* **`datos/`**: Contiene los archivos de entrada (`case1.txt` a `case4.txt`).
* **`informe/`**: Contiene el informe del proyecto en formato PDF (`Tarea_2_A_y_M.pdf`) y, opcionalmente, los fuentes LaTeX (`.tex`) e imágenes (`img/`).
* **`README.md`**: Este archivo.

## Resultados y Análisis

Los resultados detallados, análisis de rendimiento y comparaciones se encuentran en el informe `Tarea_2_A_y_M.pdf` dentro de la carpeta `informe/`.

## Autores

* **Renato Óscar Benjamín Contreras Carvajal** - `renato.contreras@mail.udp.cl`
* **José Martín Berríos Piña** - `jose.berrios1@mail.udp.cl`
