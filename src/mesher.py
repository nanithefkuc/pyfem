import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial import Voronoi, Delaunay

from src.shapes import Point, Rectangle, Circle, SemiCircle
from src.regions import Region

class Mesher:
    def __init__(self, regions: list[Region]):
        self.regions = regions
        self.points: list[Point] = []
        self._initialize_points()

    def CVT(self, iterations: int=100) -> None:
        start_time = time.time()
        for iter in range(iterations):
            point_coords = np.array([[p.x, p.y] for p in self.points])
            vor = Voronoi(point_coords)

            new_points = []
            for region_index in vor.regions:
                if not region_index or -1 in region_index:
                    continue

                vertices = vor.vertices[region_index]

                if len(vertices) > 0:
                    cx, cy = self._calculate_centroid(vertices)
                    new_points.append(Point(cx, cy))

            self.points = self._update_points(new_points)

            if iter % 10 == 0:
                print(f"Iteration {iter} complete. Took {time.time() - start_time}s since started.")
        
        print("CVT iterations complete!")
    
    def create_mesh(self):
        point_coords = np.array([[p.x, p.y] for p in self.points])

        triangulation = Delaunay(point_coords)

        clipped_triangles = self._clip_triangles_to_shape(triangulation)

        return clipped_triangles

    def visualize_points(self) -> None:
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

        if self.points:
            grid_x = [p.x for p in self.points]
            grid_y = [p.y for p in self.points]
            ax.scatter(grid_x, grid_y, s=10, c='red', marker='o', label='Grid Points')

        ax.set_aspect('equal', 'box')
        ax.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Geometry Visualization with Grid Points')
        plt.show()

    def visualize_mesh(self, triangles):
        fig, ax = plt.subplots()

        # Plot the shape boundary (circle, rectangle, etc.)
        for region in self.regions:
            shape = region.shape
            if isinstance(shape, Circle):
                circle = patches.Circle((shape.center.x, shape.center.y), shape.radius,
                                        color='purple', alpha=0.5, edgecolor='black')
                ax.add_patch(circle)
            elif isinstance(shape, Rectangle):
                rectangle = patches.Rectangle((shape.center.x - shape.length / 2, shape.center.y - shape.width / 2),
                                            shape.length, shape.width, color='green', alpha=0.5)
                ax.add_patch(rectangle)
            elif isinstance(shape, SemiCircle):
                wedge = patches.Wedge((shape.center.x, shape.center.y), shape.radius,
                                      np.degrees(shape.rotation - np.pi / 2),
                                      np.degrees(shape.rotation + np.pi / 2),
                                      color='green', alpha=0.5, label='Semi-Circle')
                ax.add_patch(wedge)

                # Plot the flat edge of the semi-circle (along the x-axis)
                flat_x1 = shape.center.x + shape.radius * np.cos(shape.rotation + np.pi / 2)
                flat_y1 = shape.center.y + shape.radius * np.sin(shape.rotation + np.pi / 2)
                flat_x2 = shape.center.x + shape.radius * np.cos(shape.rotation - np.pi / 2)
                flat_y2 = shape.center.y + shape.radius * np.sin(shape.rotation - np.pi / 2)
                ax.plot([flat_x1, flat_x2], [flat_y1, flat_y2], 'k-', lw=2)

        # Plot the triangular mesh
        for triangle in triangles:
            triangle_patch = patches.Polygon(triangle, closed=True, fill=True, color='blue', edgecolor='yellow', alpha=0.6)
            ax.add_patch(triangle_patch)

        # Set the axis limits based on the points' bounding box
        all_points = np.array([p for region in self.regions for p in triangles])
        x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)

        ax.set_aspect('equal', 'box')
        plt.show()
    
    def _initialize_points(self) -> None:
        for region in self.regions:
            # Generate and add boundary points
            boundary_points = region.generate_boundary_points()
            self.points.extend(boundary_points)

            # Generate random internal points
            num_internal_points = 1000
            internal_points = []

            if isinstance(region.shape, Circle):
                while len(internal_points) < num_internal_points:
                    x = random.uniform(region.shape.center.x - region.shape.radius,
                                    region.shape.center.x + region.shape.radius)
                    y = random.uniform(region.shape.center.y - region.shape.radius,
                                    region.shape.center.y + region.shape.radius)
                    if region.in_bounds((x, y)):
                        internal_points.append(Point(x, y))
            elif isinstance(region.shape, Rectangle):
                while len(internal_points) < num_internal_points:
                    x = random.uniform(region.shape.center.x - region.shape.length / 2,
                                       region.shape.center.x + region.shape.length / 2)
                    y = random.uniform(region.shape.center.y - region.shape.width / 2,
                                       region.shape.center.y + region.shape.width / 2)
                    if region.in_bounds((x, y)):
                        internal_points.append(Point(x, y))
            elif isinstance(region.shape, SemiCircle):
                while len(internal_points) < num_internal_points:
                    angle = random.uniform(region.shape.rotation - np.pi / 2, region.shape.rotation + np.pi / 2)
                    radius = region.shape.radius * np.sqrt(random.uniform(0, 1))
                    x = region.shape.center.x + radius * np.cos(angle)
                    y = region.shape.center.y + radius * np.sin(angle)

                    # Flat side bounds
                    if random.random() < 0.5:
                        x = random.uniform(
                            region.shape.center.x + region.shape.radius * np.cos(region.shape.rotation + np.pi / 2),
                            region.shape.center.x + region.shape.radius * np.cos(region.shape.rotation - np.pi / 2)
                        )
                        y = region.shape.center.y + region.shape.radius * np.sin(region.shape.rotation)

                    if region.in_bounds((x, y)):
                        internal_points.append(Point(x, y))

        self.points.extend(internal_points)

    def _calculate_centroid(self, vertices: np.ndarray) -> tuple[float, float]:
        vertices = np.array(vertices)
        if not np.array_equal(vertices[0], vertices[-1]):
            vertices = np.vstack([vertices, vertices[0]])

        x = vertices[:, 0]
        y = vertices[:, 1]
        area = 0.5 * np.sum(x[:-1] * y[1:] - y[:-1] * x[1:])

        cx = (1 / (6 * area)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - y[:-1] * x[1:]))
        cy = (1 / (6 * area)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - y[:-1] * x[1:]))

        return (cx, cy)

    def _update_points(self, new_points) -> list[Point]:
        updated_points = []
        boundary_index = 0
        internal_index = 0
        for point in self.points:
            if point.is_boundary:
                self._move_boundary_point(point, new_points[boundary_index] if boundary_index < len(new_points) else None)
                updated_points.append(point)
                boundary_index += 1
            else:
                if internal_index < len(new_points):
                    new_point = new_points[internal_index]
                    is_valid = any(region.in_bounds((new_point.x, new_point.y)) for region in self.regions)
                    if is_valid:
                        updated_points.append(new_point)
                    else:
                        updated_points.append(point)
                    internal_index += 1
                else:
                    updated_points.append(point)
        
        return updated_points
    
    def _move_boundary_point(self, point: Point, new_position: list) -> None:
        for region in self.regions:
            if isinstance(region.shape, Rectangle):
                self._move_boundary_point_rectangle(region.shape, point)
            elif isinstance(region.shape, Circle):
                self._move_boundary_point_circle(region.shape, point)
            elif isinstance(region.shape, SemiCircle):
                self._move_boundary_point_semicircle(region.shape, point)

    def _move_boundary_point_rectangle(self, shape: Rectangle, point: Point) -> None:
        cx, cy = shape.center.x, shape.center.y
        angle = shape.rotation
        length, width = shape.length, shape.width

        cos_theta = np.cos(-angle)
        sin_theta = np.sin(-angle)
        local_x = cos_theta * (point.x - cx) - sin_theta * (point.y - cy)
        local_y = sin_theta * (point.x - cx) - cos_theta * (point.y - cy)

        if abs(local_x) > length / 2 or abs(local_y) > width / 2:
            clamped_x = max(-length / 2, min(length / 2, local_x))
            clamped_y = max(-width / 2, min(width / 2, local_y))
        else:
            if length / 2 - abs(local_x) < width / 2 - abs(local_y):
                clamped_x = np.sign(local_x) * length / 2
                clamped_y = local_y
            else:
                clamped_x = local_x
                clamped_y = np.sign(local_y) * width / 2

        global_x = cos_theta * clamped_x + sin_theta * clamped_y + cx
        global_y = -sin_theta * clamped_x + cos_theta * clamped_y + cy

        point.x, point.y = global_x, global_y

    def _move_boundary_point_circle(self, shape: Circle, point: Point) -> None:
        vector_to_new_point = np.array([point.x - shape.center.x, point.y - shape.center.y])

        norm = np.linalg.norm(vector_to_new_point)
        if norm == 0:
            vector_to_new_point = np.array([shape.radius, 0])
        else:
            vector_to_new_point = (vector_to_new_point / norm) * shape.radius

        point.x = shape.center.x + vector_to_new_point[0]
        point.y = shape.center.y + vector_to_new_point[1]

    def _move_boundary_point_semicircle(self, shape: SemiCircle, point: Point):
        cx, cy = shape.center.x, shape.center.y
        radius = shape.radius
        rotation = shape.rotation

        vector_to_point = np.array([point.x - cx, point.y - cy])
        distance_to_point = np.linalg.norm(vector_to_point)

        angle_to_point = np.arctan2(vector_to_point[1], vector_to_point[0])
        relative_angle = angle_to_point - rotation

        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

        if -np.pi / 2 <= relative_angle <= np.pi / 2:
            # Curved side
            if distance_to_point != 0:
                vector_to_point_normalized = vector_to_point / distance_to_point
                point.x = cx + radius * vector_to_point_normalized[0]
                point.y = cy + radius * vector_to_point_normalized[1]
        else:
            flat_x1 = cx + radius * np.cos(rotation + np.pi / 2)
            flat_y1 = cy + radius * np.sin(rotation + np.pi / 2)
            flat_x2 = cx + radius * np.cos(rotation - np.pi / 2)
            flat_y2 = cy + radius * np.sin(rotation - np.pi / 2)

            line_vector = np.array([flat_x2 - flat_x1, flat_y2 - flat_y1])
            point_vector = np.array([point.x - flat_x1, point.y - flat_y1])
            t = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
            t = max(0, min(1, t))

            closest_point = np.array([flat_x1, flat_y1]) + t * line_vector
            point.x, point.y = closest_point[0], closest_point[1]

    def _clip_triangles_to_shape(self, triangulation: Delaunay):
        clipped_triangles = []

        for simplex in triangulation.simplices:
            for region in self.regions:
                triangles = [triangulation.points[simplex[i]] for i in range(3)]
                clipped_triangle = self._clip_triangle_to_region(triangles, region)

                if clipped_triangles is not None:
                    clipped_triangles.append(clipped_triangle)
                break

        return clipped_triangles
    
    def _clip_triangle_to_region(self, triangle, region: Region):
        if isinstance(region.shape, Rectangle):
            return self._clip_triangle_to_rectangle(triangle, region)
        elif isinstance(region.shape, Circle):
            return self._clip_triangle_to_circle(triangle, region)
        elif isinstance(region.shape, SemiCircle):
            return self._clip_triangle_to_semicircle(triangle, region)      
        
    def _clip_triangle_to_circle(self, triangle, region: Region):
        circle = region.shape if isinstance(region.shape, Circle) else None
        cx, cy = circle.center.x, circle.center.y
        radius = circle.radius

        clipped_triangle = []
        for vertex in triangle:
            distance_to_center = np.linalg.norm([vertex[0] - cx, vertex[1] - cy])
            if distance_to_center <= radius:
                clipped_triangle.append(vertex)
            else:
                direction = np.array([vertex[0] - cx, vertex[1] - cy])
                direction /= np.linalg.norm(direction)
                clipped_vertex = np.array([cx, cy]) + direction * radius
                clipped_triangle.append(clipped_vertex)

        if len(clipped_triangle) == 3:
            return np.array(clipped_triangle)
        else:
            print(f"Triangle discarded: {triangle}")
            return None
        
    def _clip_triangle_to_rectangle(self, triangle, region: Region):
        min_x, min_y, max_x, max_y = region.shape.bounds

        clipped_triangle = []

        def clip_to_boundary(p):
            clipped_x = np.clip(p[0], min_x, max_x)
            clipped_y = np.clip(p[1], min_y, max_y)
            return np.array([clipped_x, clipped_y])
        
        for vertex in triangle:
            if region.in_bounds(Point(vertex[0], vertex[1])):
                clipped_triangle.append(vertex)
            else:
                clipped_vertex = clip_to_boundary(vertex)
                clipped_triangle.append(clipped_vertex)

        if len(clipped_triangle) == 3:
            return np.array(clipped_triangle)
        
        return None
    
    def _clip_triangle_to_semicircle(self, triangle, region: Region):
        semicircle = region.shape
        cx, cy = semicircle.center.x, semicircle.center.y
        radius = semicircle.radius
        rotation = semicircle.rotation

        clipped_triangle = []

        def clip_to_semicircle_boundary(p):
            vector = np.array([p.x - cx, p.y - cy])
            norm = np.linalg.norm(vector)
            if norm == 0:
                return np.array([cx, cy])
            if norm > radius:
                vector = vector / norm * radius
            return np.array([cx, cy]) + vector
        
        for vertex in triangle:
            if region.in_bounds(vertex):
                clipped_triangle.append(vertex)
            else:
                clipped_vertex = clip_to_semicircle_boundary(vertex)
                clipped_triangle.append(clipped_vertex)
        
        if len(clipped_triangle) == 3:
            return np.array(clipped_triangle)
        
        return None