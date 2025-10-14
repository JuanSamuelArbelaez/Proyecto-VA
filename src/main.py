from utils import mostrar_imagen
import anturios as ant
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
        "src/data/P2/Manzana_01.png",
        "src/data/P2/Manzana_02.png",
        "src/data/P2/Manzana_03.png",
        "src/data/P2/Manzana_04.png",
        "src/data/P2/Manzana_05.png",
        "src/data/P2/Manzana_06.jpg",
        "src/data/P2/Manzana_07.jpg",
        "src/data/P2/Manzana_08.png",
        "src/data/P2/Manzana_09.png",
        "src/data/P2/Manzana_10.png",
    ]

    ruta1 = rutas[3]
    ruta2 = rutas[4]

    pb.pruebas_hough(ruta1)
    pb.pruebas_descriptores(ruta1, ruta2)
    pb.pruebas_hog(ruta1)
    pb.pruebas_texturas(ruta1)
    pb.pruebas_momentos_hu(ruta1)

if __name__ == "__main__":
    manzanas()