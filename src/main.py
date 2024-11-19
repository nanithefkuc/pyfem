from src.shapes import Rectangle, Circle, SemiCircle
from src.regions import Region
from src.materials import LinearMaterial
from src.mesher import Mesher

IRON = LinearMaterial("Pure Iron", 5000, 5000, 0, 0)
N52 = LinearMaterial("N52", 1.04, 1.04, 0, 0)
COPPER = LinearMaterial("Copper", 2, 2, 0, 0)
AIR = LinearMaterial("Air", 1, 1, 0, 0)

def run():
    iron_rect = Region(Rectangle(0, 0, 5, 10), IRON)
    # iron_rect2 = Region(Rectangle(5, 0, 2, 2), IRON)

    mesher = Mesher([iron_rect])
    mesher.CVT(iterations=100)
    mesher.visualize_points()
    mesh = mesher.create_mesh()
    mesher.visualize_mesh(mesh)

if __name__ == "__main__":
    run()