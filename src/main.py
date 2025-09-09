from utils import mostrar_imagen
import anturios as ant
import pruebas as pb

if __name__ == "__main__":
    ruta1 = "src/data/anturio01.jpg"

    """
    pb.pruebas_color_ops(ruta1)
    pb.pruebas_arithm_ops(ruta1, ruta1)
    pb.pruebas_geometric_ops(ruta1)
    pb.pruebas_logic_ops(ruta1, ruta1)
    pb.pruebas_morph_ops(ruta1)
    pb.pruebas_morph_segm_ops(ruta1)
    pb.pruebas_filters(ruta1)
    """

    rutas = ["src/data/anturio01.jpg",
             "src/data/anturio02.jpg",
             "src/data/anturio03.jpg",
             "src/data/anturio04.jpg",
             "src/data/anturio05.jpg",
             "src/data/anturio06.jpg",]
    
    for ruta in rutas:
        resultado, length = ant.detectar_anturiors(ruta)
        print("Número de anturios detectados:", length)
        mostrar_imagen(resultado,"Bordes: "+str(length))

