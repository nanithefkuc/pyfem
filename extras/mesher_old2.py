import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial import Voronoi
from src.shapes import Point, Rectangle, Circle, SemiCircle, POINT
from src.regions import Region
from src.materials import Material

class Mesher:
    def __init__(self, regions: list[Region]):
        self.regions = regions
        self.grid_points: list[Point] = []

    def apply_cvt(self, max_iterations: int=100, tolerance: float=1e-6, relaxation: float=1.0):
        for iteration in range(max_iterations):
            # Convert points to numpy array for Voronoi computation
            points = np.array([[p.x, p.y] for p in self.grid_points])
            vor = Voronoi(points)

            # Track maximum movement
            max_movement = 0

            # Process each point
            for i, point in enumerate(self.grid_points):
                if point in self.edge_points:
                    print(f"Edge point before: ({point.x}, {point.y})")
                    new_position = self._move_edge_point_(point, vor.regions[vor.point_region[i]], vor.vertices)
                    print(f"Edge point after: {new_position}")
                else:
                    # Handle internal point movement
                    new_position = self._compute_centroid_(vor.regions[vor.point_region[i]], vor.vertices)

                if new_position is None:
                    continue

                # Apply relaxation
                dx = (new_position[0] - point.x) * relaxation
                dy = (new_position[1] - point.y) * relaxation

                # Update point position
                point.x += dx
                point.y += dy

                # Track maximum movement
                max_movement = max(max_movement, np.sqrt(dx * dx + dy * dy))

            # Check for convergence
            if max_movement < tolerance:
                break

        print(f"Converged after {iteration + 1} iterations")

    def visualize(self):
        fig, ax = plt.subplots()

        # Plot each shape in regions
        for region in self.regions:
            shape = region.shape
            if isinstance(shape, SemiCircle):
                wedge = patches.Wedge((shape.center.x, shape.center.y), shape.radius,
                                      np.degrees(shape.rotation - np.pi / 2),
                                      np.degrees(shape.rotation + np.pi / 2),
                                      color='green', alpha=0.5)
                ax.add_patch(wedge)
                # Plot flat edge of the semicircle
                flat_x1 = shape.center.x + shape.radius * np.cos(shape.rotation + np.pi / 2)
                flat_y1 = shape.center.y + shape.radius * np.sin(shape.rotation + np.pi / 2)
                flat_x2 = shape.center.x + shape.radius * np.cos(shape.rotation - np.pi / 2)
                flat_y2 = shape.center.y + shape.radius * np.sin(shape.rotation - np.pi / 2)
                ax.plot([flat_x1, flat_x2], [flat_y1, flat_y2], 'k-')
            elif isinstance(shape, Circle):
                circle = patches.Circle((shape.center.x, shape.center.y), shape.radius,
                                        color='purple', alpha=0.5, edgecolor='black', label='Circle')
                ax.add_patch(circle)
            else:
                x, y = [p.x for p in shape.vertices], [p.y for p in shape.vertices]
                ax.fill(x, y, alpha=0.5, fc='blue', ec='black', label='Rectangle')

        if self.grid_points:
            grid_x = [p.x for p in self.grid_points]
            grid_y = [p.y for p in self.grid_points]
            ax.scatter(grid_x, grid_y, s=10, c='red', marker='o', label='Grid Points')

        ax.set_aspect('equal', 'box')
        ax.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Geometry Visualization with Grid Points')
        plt.show()

    def _create_grid_points_(self, step_size: float=1.0) -> None:
        if not self.regions:
            raise ValueError("No regions provided!")
        
        min_x, min_y, max_x, max_y = self.regions[0].shape.bounds

        for region in self.regions:
            min_x = min(min_x, region.shape.bounds[0])
            min_y = min(min_y, region.shape.bounds[1])
            max_x = max(max_x, region.shape.bounds[2])
            max_y = max(max_y, region.shape.bounds[3])

        # Create edge points
        edge_points = set()
        for region in self.regions:
            shape = region.shape
            if isinstance(shape, Rectangle):
                # Add points along each edge of the rectangle
                for i in range(len(shape.vertices)):
                    p1 = shape.vertices[i]
                    p2 = shape.vertices[(i + 1) % len(shape.vertices)]

                    # Caculate number of poitns needed for this edge
                    edge_length = p2.distance_to(p1)
                    num_points = int(edge_length / step_size)

                    for j in range(1, num_points):
                        t = j / num_points
                        x = p1.x + t * (p2.x - p1.x)
                        y = p1.y + t * (p2.y - p1.y)
                        edge_points.add(Point(x, y))

                    edge_points.add(p1)
            elif isinstance(shape, Circle):
                # Add points along the circle's circumference
                num_points = int(2 * np.pi * shape.radius / step_size)
                for i in range(num_points):
                    theta = 2 * np.pi * i / num_points
                    x = shape.center.x + shape.radius * np.cos(theta)
                    y = shape.center.y + shape.radius * np.sin(theta)
                    edge_points.add(Point(x, y))
            elif isinstance(shape, SemiCircle):
                # Add points along the semicircle's arc
                num_points = int(np.pi * shape.radius / step_size)
                for i in range(num_points):
                    theta = shape.rotation - np.pi / 2 + np.pi * i / (num_points - 1)
                    x = shape.center.x + shape.radius * np.cos(theta)
                    y = shape.center.y + shape.radius * np.sin(theta)
                    edge_points.add(Point(x, y))

                # Add points along the flat edge
                flat_x1 = shape.center.x + shape.radius * np.cos(np.radians(shape.rotation) + np.pi / 2)
                flat_y1 = shape.center.y + shape.radius * np.sin(np.radians(shape.rotation) + np.pi / 2)
                flat_x2 = shape.center.x + shape.radius * np.cos(np.radians(shape.rotation) - np.pi / 2)
                flat_y2 = shape.center.y + shape.radius * np.sin(np.radians(shape.rotation) - np.pi / 2)

                edge_length = np.sqrt((flat_x2 - flat_x1)**2 + (flat_y2 - flat_y1)**2)
                num_points = int(edge_length / step_size)
                for i in range(num_points + 1):
                    t = i / num_points
                    x = flat_x1 + t * (flat_x2 - flat_x1)
                    y = flat_y1 + t * (flat_y2 - flat_y1)
                    edge_points.add(Point(x, y))
        
        self.edge_points = edge_points

        x_points = np.arange(min_x + step_size, max_x, step_size)
        y_points = np.arange(min_y + step_size, max_y, step_size)

        print(f"Y points range: {y_points[0]} to {y_points[-1]}")

        grid_points = [Point(x, y) for x in x_points for y in y_points]

        filtered_points = []
        for point in grid_points:
            for region in self.regions:
                if region.in_bounds(point):
                    filtered_points.append(point)
                    break

        print(f"Grid points: {len(grid_points)}\nFiltered points: {len(filtered_points)}")

        self.grid_points = filtered_points + list(edge_points)

    def _compute_centroid_(self, region_indices: list, vor_vertices: np.ndarray) -> tuple[float, float]:
        if not region_indices: # Handle unbounded regions
            return None
        
        # Filter out -1 indices (unbounded regions)
        valid_indices = [idx for idx in region_indices if idx != -1]
        if not valid_indices:
            return None
        
        # Get vertices of the Voronoi cell
        vertices = vor_vertices[region_indices]

        # Close the polygon by repeating the first vertex
        vertices = np.vstack((vertices, vertices[0]))

        # Compute area and centroid using the shoelace formula
        x = vertices[:, 0]
        y = vertices[:, 1]

        # Shifted arrays for computation
        x_shift = np.roll(x, -1)
        y_shift = np.roll(y, -1)

        # Area using shoelace formula
        A = 0.5 * np.sum(x * y_shift - x_shift * y)

        if abs(A) < 1e-10:
            return (np.mean(x[:-1]), np.mean(y[:-1]))
        
        # Centroid formula for polygon
        cx = np.sum((x + x_shift) * (x * y_shift - x_shift * y)) / (6.0 * A)
        cy = np.sum((y + y_shift) * (x * y_shift - x_shift * y)) / (6.0 * A)

        return (cx, cy)

    def _move_edge_point_(self, point: Point, region_indices: list, vor_vertices: np.ndarray) -> tuple[float, float]:
        original_pos = (point.x, point.y)
        
        # Compute unrestrained centeroid
        centroid = self._compute_centroid_(region_indices, vor_vertices)
        if centroid is None:
            return original_pos

        # Find which shape and boundary this point belongs to
        for region in self.regions:
            shape = region.shape

            if isinstance(shape, Rectangle):
                # For rectangles, project on to closest edge
                for i in range(len(shape.vertices)):
                    p1 = shape.vertices[i]
                    p2 = shape.vertices[(i + 1) % len(shape.vertices)]

                    # Check if point is on this edge (within some tolerance)
                    if self._point_on_line_segment_(point, p1, p2, 1e-6):
                        new_pos = self._project_point_to_line_segment_(centroid, p1, p2)
                        # Verify the project is actually on the edge
                        if not self._point_on_line_segment_(Point(new_pos[0], new_pos[1]), p1, p2, 1e-6):
                            return original_pos
                        return new_pos
            elif isinstance(shape, Circle):
                # For circles, project onto the circumference
                if abs(np.sqrt((point.x - shape.center.x) ** 2 +
                               (point.y - shape.center.y) ** 2) - shape.radius) < 1e-6:
                    # Vector from center to centroid
                    dx = centroid[0] - shape.center.x
                    dy = centroid[1] - shape.center.y
                    dist = np.sqrt(dx * dx + dy * dy)

                    # Scale to radius
                    new_pos = (
                        shape.center.x + (dx / dist) * shape.radius,
                        shape.center.y + (dy / dist) * shape.radius
                    )

                    # Verify new position is still on circle
                    if abs(np.sqrt((new_pos[0] - shape.center.x) ** 2 +
                                   (new_pos[1] - shape.center.y) ** 2) - shape.radius) > 1e-6:
                        return original_pos
                    return new_pos
            elif isinstance(shape, SemiCircle):
                # Check if point is on the flat edge
                flat_x1 = shape.center.x + shape.radius * np.cos(np.radians(shape.rotation) + np.pi / 2)
                flat_y1 = shape.center.y + shape.radius * np.sin(np.radians(shape.rotation) + np.pi / 2)
                flat_x2 = shape.center.x + shape.radius * np.cos(np.radians(shape.rotation) - np.pi / 2)
                flat_y2 = shape.center.y + shape.radius * np.sin(np.radians(shape.rotation) - np.pi / 2)

                if self._point_on_line_segment_(point, Point(flat_x1, flat_y1), Point(flat_x2, flat_y2), 1e-6):
                    new_pos = self._project_point_to_line_segment_(
                        centroid,
                        Point(flat_x1, flat_y1),
                        Point(flat_x2, flat_y2)
                    )
                    # Verify new position is on flat edge
                    if not self._point_on_line_segment_(Point(new_pos[0], new_pos[1]),
                                                        Point(flat_x1, flat_y1),
                                                        Point(flat_x2, flat_y2), 1e-6):
                        return original_pos
                    return new_pos
                
                # Check if point is on the arc
                dist_to_center = np.sqrt((point.x - shape.center.x) ** 2 +
                                         (point.y - shape.center.y) ** 2)
                if abs(dist_to_center - shape.radius) < 1e-6:
                    # Project onto arc, ensuring we stay within semicircle bounds
                    dx = centroid[0] - shape.center.x
                    dy = centroid[1] - shape.center.y
                    angle = np.arctan2(dy, dx)

                    # Constrain angle to semicircle range
                    min_angle = shape.rotation - np.pi / 2
                    max_angle = shape.rotation + np.pi / 2
                    angle = np.clip(angle, min_angle, max_angle)

                    new_pos = (
                        shape.center.x + shape.radius * np.cos(np.radians(angle)),
                        shape.center.y + shape.radius * np.sin(np.radians(angle))
                    )

                    # Verify new position is on arc and within angle bounds
                    new_angle = np.arctan2(new_pos[1] - shape.center.y,
                                           new_pos[0] - shape.center.x)
                    if (abs(np.sqrt((new_pos[0] - shape.center.x) ** 2 +
                                    (new_pos[1] - shape.center.y) ** 2) - shape.radius) > 1e-6 or
                                    not min_angle <= new_angle <= max_angle):
                        return original_pos
                    return new_pos
                
        return original_pos
    
    def _point_on_line_segment_(self, point: Point, p1: Point, p2: Point, tolerance: float) -> bool:
        # Vector from p1 to p2
        v = (p1.x - p2.x, p1.y - p2.y)
        # Vector from p1 to point
        w = (p1.x - point.x, p1.y - point.y)

        # Length of line segment squared
        c1 = v[0] * v[0] + v[1] * v[1]

        if c1 == 0:
            return np.sqrt(w[0] * w[0] + w[1] * w[1]) <= tolerance
        
        # Projection ratio
        t = max(0, min(1, (w[0] * v[0] + w[1] * v[1]) / c1))

        # Project point onto line
        projection = (
            p1.x + t * v[0],
            p1.y + t * v[1]
        )

        # Check distance from point to projection
        dx = point.x - projection[0]
        dy = point.y - projection[1]

        return np.sqrt(dx * dx + dy * dy) <= tolerance
    
    def _project_point_to_line_segment_(self, point: POINT, p1: Point, p2: Point) -> POINT:
        # Vector from p1 to p2
        v = (p1.x - p2.x, p1.y - p2.y)
        # Vector from p1 to point
        w = (point[0] - p1.x, point[1] - p1.y)

        # Length of line segment squared
        c1 = v[0] * v[0] + v[1] * v[1]

        if c1 == 0:
            return (p1.x, p1.y)
        
        t = max(0, min(1, (w[0] * v[0] + w[1] * v[1]) / c1))

        return (
            p1.x + t * v[0],
            p1.y + t * v[1]
        )