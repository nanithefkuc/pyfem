import numpy as np

POINT = tuple[float, float]

class Point:
    def __init__(self, x: float, y: float, is_boundary: bool=False):
        self.x = x
        self.y = y
        self.is_boundary = is_boundary

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"
    
    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
class Line:
    def __init__(self, a: Point | POINT, b: Point | POINT):
        self.a = a
        self.b = b

        self.length = self.a.distance_to(self.b)

    def __repr__(self) -> str:
        return f"Line({self.a} -> {self.b})"
    
class Arc:
    def __init__(self, center: Point | POINT, radius: float, angle: float):
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])
        self.radius = radius
        self.angle = angle

        self.arc_length = radius * np.radians(self.angle)

    def __repr__(self) -> str:
        return f"Arc({self.center}, {self.radius}, {self.angle})"
    
class Rectangle:
    """
    Rectangle class.
    :param x: X coordinate of the starting point.
    :param y: Y coordinate of the starting point.
    :param length: Length of the rectangle.
    :param width: Width of the rectangle.
    :param rotation: The rotation of the rectangle about it's center (in degrees).
    """
    def __init__(self, x: float, y: float, length: float, width: float, *, rotation: float=0):
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.rotation = rotation
        self.center = Point((self.x + self.length) / 2, (self.y + self.width) / 2)

        # Rectangular vertices
        v1 = Point(self.x, self.y)
        v2 = Point(self.x + self.length, self.y)
        v3 = Point(self.x + self.length, self.y + self.width)
        v4 = Point(self.x, self.y + self.width)

        # Rectangle's side's properties
        self.left = Line(v1, v4)
        self.right = Line(v2, v3)
        self.top = Line(v3, v4)
        self.bottom = Line(v1, v2)

        # Geometric properties
        self.area = self.length * self.width
        vertex_matrix = np.array([
            [v1.x, v1.y],
            [v2.x, v2.y],
            [v3.x, v3.y],
            [v4.x, v4.y]
        ])

        if rotation != 0:
            vertex_matrix = vertex_matrix - [self.center.x, self.center.y]

            rotation_matrix = np.array([
                [np.cos(np.radians(rotation)), -np.sin(np.radians(rotation))],
                [np.sin(np.radians(rotation)),  np.cos(np.radians(rotation))]
            ])
            vertex_matrix = np.dot(vertex_matrix, rotation_matrix) + [self.center.x, self.center.y]

            v1 = vertex_matrix[0]
            v2 = vertex_matrix[1]
            v3 = vertex_matrix[2]
            v4 = vertex_matrix[3]

        self.vertices = [Point(p[0], p[1]) for p in vertex_matrix]

        min_x, min_y = np.min(vertex_matrix, axis=0)
        max_x, max_y = np.max(vertex_matrix, axis=0)

        self.bounds = (min_x, min_y, max_x, max_y)

    def __repr__(self) -> str:
        return f"Rectangle(({self.x}, {self.y}), {self.length}, {self.width})"
    
class Circle:
    def __init__(self, center: Point | POINT, radius: float):
        _circle_ = Arc(center, radius, 2 * np.pi)

        # Circle properties
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])
        self.radius = radius
        self.circumference = _circle_.arc_length
        self.diameter = self.radius * 2

        # Geometric properties
        self.area = np.pi * radius ** 2
        self.bounds = (self.center.x - self.radius, self.center.y - self.radius, self.center.x + self.radius, self.center.y + self.radius)

    def __repr__(self) -> str:
        return f"Circle({self.center}, {self.radius})"
    
class SemiCircle:
    def __init__(self, center: Point | POINT, radius: float, *, rotation: float=0):
        _semicircle_ = Arc(center, radius, np.pi)

        # Semicircle properties
        self.center = center if isinstance(center, Point) else Point(center[0], center[1])
        self.radius = radius
        self.rotation = rotation
        self.arc_length = _semicircle_.arc_length
        self.diameter = self.radius * 2
        self.area = 0.5 * np.pi * self.radius ** 2

        # Points on the flat edge
        p1_x = self.center.x
        p1_y = self.center.y + self.radius
        p2_x = self.center.x
        p2_y = self.center.y - self.radius

        # Furthest point on the arc
        p3_x = self.center.x + self.radius
        p3_y = self.center.y

        # Apply rotation if needed
        if rotation != 0:
            points = np.array([
                [p1_x, p1_y],
                [p2_x, p2_y],
                [p3_x, p3_y]
            ])

            rotation_matrix = np.array([
                [np.cos(np.radians(rotation)), -np.sin(np.radians(rotation))],
                [np.sin(np.radians(rotation)),  np.cos(np.radians(rotation))]
            ])

            # Rotate around center
            points = points - [self.center.x, self.center.y]
            points = np.dot(points, rotation_matrix)
            points = points + [self.center.x, self.center.y]

            # Extract rotated points
            p1_x, p1_y = points[0]
            p2_x, p2_y = points[1]
            p3_x, p3_y = points[2]

        # Calculate the bounding box from the points
        points = np.array([
            [p1_x, p1_y],
            [p2_x, p2_y],
            [p3_x, p3_y]
        ])

        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        self.bounds = (min_x, min_y, max_x, max_y)