import pygame
import math

# Configuraciones iniciales
ANCHO_VENTANA = 600
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = col * ancho
        self.y = fila * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []
        self.g = float("inf")
        self.h = float("inf")
        self.f = float("inf")
        self.padre = None

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        self.color = VERDE

    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_camino(self):
        self.color = AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for d_fila, d_col in direcciones:
            nueva_fila = self.fila + d_fila
            nueva_col = self.col + d_col
            if 0 <= nueva_fila < self.total_filas and 0 <= nueva_col < self.total_filas:
                vecino = grid[nueva_fila][nueva_col]
                if not vecino.es_pared():
                    self.vecinos.append(vecino)

def heuristica(a, b):
    dx = abs(a.fila - b.fila)
    dy = abs(a.col - b.col)
    return 10 * (dx + dy) + (14 - 2 * 10) * min(dx, dy)

def reconstruir_camino(nodo_final, dibujar_func):
    actual = nodo_final
    while actual.padre:
        actual = actual.padre
        if not actual.es_inicio():
            actual.hacer_camino()
            dibujar_func()

def algoritmo_estrella(grid, inicio, fin, dibujar_func):
    abierta = [inicio]
    cerrada = []

    inicio.g = 0
    inicio.h = heuristica(inicio, fin)
    inicio.f = inicio.h

    while len(abierta) > 0:
        pygame.event.pump()

        # Imprimir solo las coordenadas de los nodos en las listas abierta y cerrada
        print("------------- Iteración nueva -------------")
        print("Lista Abierta:")
        for nodo in abierta:
            print(f"  {nodo.get_pos()}")

        print("\nLista Cerrada:")
        for nodo in cerrada:
            print(f"  {nodo.get_pos()}")
        print("-------------------------------------------\n")

        # Desempate en 'f', 'h', 'col' y 'fila' para situaciones simétricas
        actual = min(abierta, key=lambda nodo: (nodo.f, nodo.h, nodo.col, nodo.fila))

        if actual == fin:
            reconstruir_camino(fin, dibujar_func)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return True

        abierta.remove(actual)
        cerrada.append(actual)
        if not actual.es_inicio():
            actual.hacer_cerrado()

        for vecino in actual.vecinos:
            if vecino in cerrada:
                continue

            dx = abs(actual.fila - vecino.fila)
            dy = abs(actual.col - vecino.col)
            costo = 14 if dx != 0 and dy != 0 else 10

            temp_g = actual.g + costo

            if temp_g < vecino.g:
                vecino.padre = actual
                vecino.g = temp_g
                vecino.h = heuristica(vecino, fin)
                vecino.f = vecino.g + vecino.h

                if vecino not in abierta:
                    abierta.append(vecino)
                    if not vecino.es_fin():
                        vecino.hacer_abierto()

        dibujar_func()

    return False
# Tabla
def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    x, y = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def main(ventana, ancho):
    FILAS = 11
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None
# Eventos
    corriendo = True
    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and inicio and fin:
                    for fila_grid in grid:
                        for nodo_grid in fila_grid:
                            nodo_grid.actualizar_vecinos(grid)
                    algoritmo_estrella(grid, inicio, fin, lambda: dibujar(ventana, grid, FILAS, ancho))

                if evento.key == pygame.K_c:  # Limpiar todo
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)