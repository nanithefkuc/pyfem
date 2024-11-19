class Material:
    def __init__(self, name: str, *, coercivity: float=0, conductivity: float=0, current_density: float=0):
        self.name = name
        self.coercivity = coercivity
        self.conductivity = conductivity
        self.current_density = current_density

    def __repr__(self) -> str:
        return f"Material({self.name}, {self.coercivity} A/m, {self.conductivity} MS/m, {self.current_density} MA/m^2)"
    
class LinearMaterial(Material):
    def __init__(self, name: str, Ux: float, Uy: float, Hx: float, Hy: float, *, coercivity: float=0, conductivity: float=0, current_density: float=0):
        super().__init__(name, coercivity=coercivity, conductivity=conductivity, current_density=current_density)

        self.Ux = Ux
        self.Uy = Uy
        self.Hx = Hx
        self.Hy = Hy
    
    def __repr__(self) -> str:
        return f"LinearMaterial({self.Ux}, {self.Uy}, {self.Hx}, {self.Hy})"