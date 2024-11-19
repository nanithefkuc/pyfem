import numpy as np
from src.shapes import Point, Rectangle, Circle, SemiCircle
from src.materials import Material

class Region:
    def __init__(self, shape: Rectangle | Circle | SemiCircle, material: Material | None=None):
        self.shape = shape
        self.material = material

    def __repr__(self) -> str:
        return f"Region({self.shape}, material={self.material})"
    
    def generate_boundary_points(self, num_points: int=100):
        boundary_points: list[Point] = []
        if isinstance(self.shape, Rectangle):
            rectangle = self.shape
            cx, cy = rectangle.center.x, rectangle.center.y
            length, width = rectangle.length, rectangle.width
            angle = rectangle.rotation

            corners = [
                np.array([cx - length / 2, cy - width / 2]),
                np.array([cx + length / 2, cy - width / 2]),
                np.array([cx + length / 2, cy + width / 2]),
                np.array([cx - length / 2, cy + width / 2])
            ]
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            rotated_corners = [rotation_matrix @ (corner - np.array([cx, cy])) + np.array([cx, cy]) for corner in corners]

            perimeter = 2 * (length + width)
            points_per_side = [int(num_points * (length / perimeter)), int(num_points * (width / perimeter))]

            for i in range(4):
                start_corner = rotated_corners[i]
                end_corner = rotated_corners[(i + 1) % 4]
                num_side_points = points_per_side[i % 2]

                for j in range(num_side_points):
                    t = j / (num_side_points - 1) if num_side_points > 1 else 0
                    x = (1 - t) * start_corner[0] + t * end_corner[0]
                    y = (1 - t) * start_corner[1] + t * end_corner[1]
                    boundary_points.append(Point(x, y, is_boundary=True))
        elif isinstance(self.shape, Circle):
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = self.shape.center.x + self.shape.radius * np.cos(angle)
                y = self.shape.center.y + self.shape.radius * np.sin(angle)
                boundary_points.append(Point(x, y, is_boundary=True))
        elif isinstance(self.shape, SemiCircle):
            semi_circle = self.shape
            cx, cy = semi_circle.center.x, semi_circle.center.y
            radius = semi_circle.radius
            rotation = semi_circle.rotation

            # Split points between the flat side and the curved side
            num_points_flat = num_points // 2
            num_points_curved = num_points - num_points_flat

            # Generate points along the curved side
            start_angle = rotation - np.pi / 2
            end_angle = rotation + np.pi / 2

            angles = np.linspace(start_angle, end_angle, num_points_curved)
            for angle in angles:
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                boundary_points.append(Point(x, y, is_boundary=True))

            # Generate points along the flat side
            flat_start_x = cx + radius * np.cos(rotation + np.pi / 2)
            flat_start_y = cy + radius * np.sin(rotation + np.pi / 2)
            flat_end_x = cx + radius * np.cos(rotation - np.pi / 2)
            flat_end_y = cy + radius * np.sin(rotation - np.pi / 2)

            for i in range(num_points_flat):
                t = i / (num_points_flat - 1) if num_points_flat > 1 else 0
                x = (1 - t) * flat_start_x + t * flat_end_x
                y = (1 - t) * flat_start_y + t * flat_end_y
                boundary_points.append(Point(x, y, is_boundary=True))
        
        return boundary_points
    
    def in_bounds(self, point: Point | tuple[float, float]) -> bool:
        if not isinstance(point, Point):
            point = Point(point[0], point[1])

        min_x, min_y, max_x, max_y = self.shape.bounds
        if (min_x <= point.x <= max_x) and (min_y <= point.y <= max_y):
            if isinstance(self.shape, Circle):
                return self.shape.center.distance_to(point) < self.shape.radius
            elif isinstance(self.shape, SemiCircle):
                return self._point_in_semicircle_(point)
            else:
                return self._point_in_polygon_(point)
        else:
            return False
        
    def _point_in_polygon_(self, point: Point | tuple[float, float]) -> bool:
        inside = False

        for i in range(len(self.shape.vertices)):
            v1_x, v1_y = self.shape.vertices[i].x, self.shape.vertices[i].y
            v2_x, v2_y = self.shape.vertices[(i + 1) % len(self.shape.vertices)].x, self.shape.vertices[(i + 1) % len(self.shape.vertices)].y

            # Check if the ray from the point intersects the edge
            if ((v1_y > point.y) != (v2_y > point.y)) and (point.x < (v2_x - v1_x) * (point.y - v1_y) / (v2_y - v1_y) + v1_x):
                inside = not inside
        
        return inside
    
    def _point_in_semicircle_(self, point: Point | tuple[float, float]) -> bool:
        if not isinstance(point, Point):
            point = Point(point[0], point[1])

        distance = np.sqrt((point.x - self.shape.center.x) ** 2 + (point.y - self.shape.center.y) ** 2)
        if distance > self.shape.radius:
            return False
        
        # Angle Check
        point_angle = np.arctan2(point.y - self.shape.center.y, point.x - self.shape.center.x)

        # Normalize point angle relative to orientation
        angle_diff = np.arctan2(np.sin(point_angle - self.shape.rotation),
                                np.cos(point_angle - self.shape.rotation))
        
        # Check if point lies on the curved sid
        if -np.pi / 2 <= angle_diff <= np.pi / 2:
            return True
        
        return False