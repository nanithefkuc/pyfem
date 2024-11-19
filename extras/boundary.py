import math

PERMEABILITY_TABLE = [
    [10.0634],
    [0.188706, 39.9975],
    [4.37479, 0.075118, 85.8557],
    [0.285096, 12.4933, 0.051874, 147.652],  	  	  	  	 	 
    [3.06077, 0.112378, 22.0265, 0.0423167, 225.43],
    [0.374816, 8.17242, 0.0740027, 33.0192, 0.0371234, 319.234],
    [2.40892, 0.149299, 13.8176, 0.0582172, 45.656, 0.0338673, 429.046],
    [0.457719, 6.15238, 0.0951954, 19.795, 0.0495821, 60.002, 0.0316392, 554.889],
    [2.01871, 0.187242, 10.323, 0.0732287, 26.2901, 0.0441395, 76.0845, 0.0300195, 696.75],
    [0.533241, 4.9312, 0.116425, 14.5221, 0.0612364, 33.375, 0.0403993, 93.9198, 0.028790, 854.64],
]

class ABCBoundaryCondition:
    """The Asymptotic Boundary Condition (ABC)."""
    def __init__(self, shells, center, radius):
        self.shells = shells
        self.center = center
        self.radius = radius
        self.permeability = PERMEABILITY_TABLE[shells]

        self.shell_radii = [radius * (i + 1) for i in range(self.shells)]

    def __repr__(self):
        return f"ABC(shells={self.shells}, center={self.center}, radius={self.radius})"
    
    def apply(self):
        """Apply the ABC."""
        for i, shell_radius in enumerate(self.shell_radii):
            ...