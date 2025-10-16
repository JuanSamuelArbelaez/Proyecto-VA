from utils import mostrar_imagen
import anturios as ant
import manzanas as mnz
import pruebas as pb

def anturios():
    rutas = [
        "src/data/P1/anturio01.jpg",
        "src/data/P1/anturio02.jpg",
        "src/data/P1/anturio03.jpg",
        "src/data/P1/anturio04.jpg",
        "src/data/P1/anturio05.jpg",
        "src/data/P1/anturio06.jpg",
        "src/data/P1/anturio07.jpg",
    ]

    ant.encontrar_petalos(rutas)

def manzanas():
    rutas = [
        "src/data/P2/Manzana_01_Matlab.png",
        "src/data/P2/Manzana_02_Matlab.png",
        "src/data/P2/Manzana_03_Matlab.png",
        "src/data/P2/Manzana_04_Matlab.png",
        "src/data/P2/Manzana_05_Matlab.png",
        "src/data/P2/Manzana_06_Matlab.png",
        "src/data/P2/Manzana_07_Matlab.png",
        "src/data/P2/Manzana_08_Matlab.png",
        "src/data/P2/Manzana_09_Matlab.png",
        "src/data/P2/Manzana_10_Matlab.png",
    ]

    mnz.analizar_manzanas(rutas)
    #mnz.test(rutas)

if __name__ == "__main__":
    manzanas()