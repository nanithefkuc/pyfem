import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from shapely.errors import GEOSException

from geometry import Geometry, Line
from material import Material

class Node:
    """A mesh node (vertex)."""
    def __init__(self, x: float, y: float, boundary: bool=False, id:int | None=None):
        """
        Initializes a node with its coordinates and boundary flag.

        :param x: x-coordinate of the node.
        :param y: y-coordinate of the node.
        :param boundary: Boolean flag to mark if the node is on the boundary.
        :param id: The unique identifier of the element.
        """
        self.x = x
        self.y = y
        self.boundary = boundary
        self.id = id
        self.connected_elements = []

    def __repr__(self):
        return f"Node({self.x}, {self.y}, Boundary={self.boundary})"
    
    def add_element(self, element: "Element"):
        """
        Links an element to the node, to track which elements it belongs to.
        :param element: Element instance
        """
        self.connected_elements.append(element)
    
class Element:
    def __init__(self, node1: Node, node2: Node, node3: Node):
        """
        Initializes an element (triangle) using three nodes.

        :param node1: First node.
        :param node2: Second node.
        :param node3: Third node.
        """
        self.nodes = [node1, node2, node3]

        # Add this element to the node's connected elements
        node1.add_element(self)
        node2.add_element(self)
        node2.add_element(self)

    def __repr__(self):
        return f"Element({",".join(node for node in self.nodes)})"
    
    def compute_quality(self):
        """
        Compute the quality of the triangle based on angles.
        """
        angles = self.compute_angles()
        return min(angles), max(angles)
    
    def compute_angles(self):
        """
        Calculate angles for the triangle based on the node coordinates.
        :return: A tuple with three angles in degrees.
        """
        # Extract the coordinates from the nodes
        A, B, C = self.nodes

        # Lengths of the triangle sides
        a = math.dist((B.x, B.y), (C.x, C.y))
        b = math.dist((A.x, A.y), (C.x, C.y))
        c = math.dist((A.x, A.y), (B.x, B.y))

        # Calculate the angles using Law of Cosines
        alpha = math.degrees(math.acos((b**2 + c**2 - a** 2) / (2 * b * c)))
        beta = math.degrees(math.acos((a**2 + c**2 - b**2) / (2 * a * c)))
        gamma = 180 - alpha - beta

        return (alpha, beta, gamma)
    
class Mesh:
    def __init__(self, nodes: list[Node] | None=None, elements: list[Element] | None=None):
        """
        Initializes the mesh with nodes and elements.

        :param nodes: List of node objects.
        :param elements: List of element objects.
        """
        self.nodes = nodes if nodes is not None else []
        self.elements = elements if elements is not None else []

    def __repr__(self):
        return f"Mesh({len(self.nodes)} nodes, {len(self.elements)} elements)"
    
    def add_node(self, x: float, y: float, boundary: bool=False):
        """Add a new node to the mesh and returns the node."""
        node = Node(x, y, boundary, id=len(self.nodes))
        self.nodes.append(node)
        return node
    
    def add_element(self, node1: Node, node2: Node, node3: Node):
        """Add an element to the mesh."""
        element = Element(node1, node2, node3)
        self.elements.append(element)
        return element
    
    def find_bad_elements(self):
        """
        Returns a list of elements that need refinement (bad quality).
        Bad elements are defined as those having obtuse angles or angles < 30 degrees.
        """
        bad_elements = []
        for element in self.elements:
            min_angle, max_angle = element.compute_quality()
            if max_angle > 90 or min_angle < 30:
                bad_elements.append(element)
        return bad_elements
    
    def identify_local_cluster(self, bad_triangle):
        """
        Identifies the local cluster for a bad triangle, including neighboring triangles.
        
        :param bad_triangle: The triangle (element) that needs refinement.
        :return: A list of neighboring triangles forming the local cluster.
        """
        # Find the neighbors of the bad triangle and return them
        neighbors = self.get_neighbors(bad_triangle)
    
    def refine_mesh(self):
        """Refines the mesh by refining bad elements."""
        bad_elements = self.find_bad_elements()
        pass
    
class Mesher:
    def __init__(self, geometry: Geometry, grid_resolution=0.01):
        """
        Initializes the Mesher with a given geometry and grid resolution.

        :param geometry: The predefined geometry (polygon or boundary shape).
        :param grid_resolution: The resolution of the grid. Controls how many and how dense the points of the mesh are generated.
        """
        self.geometry = geometry
        self.grid_resolution = grid_resolution
        self.initial_points = []

    def generate_mesh(self):
        """
        Generates a mesh using VCT and mesh refinement.
        Check out mesh refinement here: https://www.sciencedirect.com/science/article/pii/S0898122117306211.
        """
        # Generate initial grid of points
        self.initial_points = self.generate_initial_grid()
        print(f"Generated {len(self.initial_points)} initial points")

        # Generate CVT mesh
        optimized_points = self.generate_cvt(self.initial_points)
        print(f"CVT resulted in {len(optimized_points)} optimized points")

        return self.initial_points
    
    def generate_initial_grid(self):
        """
        Generates a grid of points within the geometry using a uniform grid,
        including points on the boundary.

        :return: A list of points (x, y) inside and on the boundary of the geometry.
        """
        points = []

        # Get geometry bounds
        min_x, min_y, max_x, max_y = self.geometry.get_bounds()

        def f_range(start, stop, step):
            values = []
            while start <= stop:
                values.append(start)
                start += step

            return values

        # Generate grid
        x = f_range(min_x, max_x, self.grid_resolution)
        y = f_range(min_y, max_y, self.grid_resolution)

        for x_point in x:
            for y_point in y:
                points.append((x_point, y_point))

        print(f"Generated grid: x range [{min_x}, {max_x}], y range [{min_y}, {max_y}]")
        print(f"Total points generated: {len(points)}")
        return points
    
    def generate_cvt(self, points, max_iterations=50, tolerance=1e-5):
        """
        Implements the CVT algorithm to optimize the distribution of points within the geometry.

        :param points: Initial set of points (seeds).
        :param max_iteration: Maximum number of iterations for the algorithm.
        :param tolerance: Minimum movement threshold for convergence.
        :return: Optimized set of points after CVT.
        """
        points = np.array(points)
        geometry_polygon = Polygon([(point.x, point.y) for point in self.geometry.points])
        
        # Check if the geometry polygon is valid
        if not geometry_polygon.is_valid:
            print("Warning: The geometry polygon is not valid. Attempting to fix...")
            geometry_polygon = geometry_polygon.buffer(0)
            if not geometry_polygon.is_valid:
                print("Error: Unable to fix the geometry polygon. CVT cannot proceed.")
                return points

        for iteration in range(max_iterations):
            vor = Voronoi(points)
            new_points = []

            for i, point in enumerate(points):
                # Get the Voronoi region (indices of vertices)
                region = vor.regions[vor.point_region[i]]

                if not region or -1 in region:
                    # For points with no valid Voronoi cell (likely on the boundary),
                    # we'll use a different approach
                    new_point = self.handle_boundary_point(point, points, geometry_polygon)
                    new_points.append(new_point)
                    continue

                # Create a polygon for the Voronoi cell
                cell = [vor.vertices[i] for i in region]
                cell_polygon = Polygon(cell)

                try:
                    # Clip the Voronoi cell to the geometry
                    clipped_cell = cell_polygon.intersection(geometry_polygon)

                    if not clipped_cell.is_empty and isinstance(clipped_cell, Polygon):
                        # Compute the centroid of the clipped Voronoi cell
                        centroid = clipped_cell.centroid
                        new_points.append((centroid.x, centroid.y))
                    else:
                        # If clipping fails, use the original point
                        new_points.append(point)
                except GEOSException as e:
                    print(f"Topology error at point {point}: {e}")
                    new_points.append(point)

            # Convert to np.array
            new_points = np.array(new_points)

            # Check for convergence
            movement = np.linalg.norm(points - new_points, axis=1).max()
            print(f"Iteration {iteration}: Max movement = {movement}")
            
            if movement < tolerance:
                print(f"Converged after {iteration} iterations.")
                break
            
            points = new_points

        return points

    def handle_boundary_point(self, point, all_points, geometry_polygon):
        """
        Handle points on or near the boundary of the geometry. Boundary points do not need a Voronoi cell.
        
        :param point: The point to handle.
        :param all_points: All points in the current iteration.
        :param geometry_polygon: The polygon representing the geometry.
        :return: The boundary point itself (or slightly adjusted).
        """
        point_shapely = Point(point[0], point[1])
        
        # Check if the point is very close to the boundary
        if geometry_polygon.boundary.distance(point_shapely) < 1e-6:
            # If the point is near or on the boundary, snap it to the nearest point on the boundary
            nearest_boundary_point = nearest_points(geometry_polygon.boundary, point_shapely)
            
            # Ensure the result is valid and not empty
            if nearest_boundary_point and not nearest_boundary_point[0].is_empty:
                return (nearest_boundary_point[0].x, nearest_boundary_point[0].y)
            else:
                print(f"Warning: Nearest point on boundary not found for {point}. Returning original point.")
                return point

        # For non-boundary points, return the original point as is
        return point

    
    def refine_mesh(self, mesh: Mesh):
        """
        Refines the mesh based on the technical paper. Iteratively improves the mesh by
        removing bad triangles and inserting new points in a controlled manner.

        :param mesh: The mesh object containing nodes and elements.
        :return: Refined mesh.
        """
        refinement_needed = True

        while refinement_needed:
            refinement_needed = False

            # Step 1: Find bad triangles
            bad_triangles = mesh.find_bad_elements()

            if not bad_triangles:
                break # No more bad triangles, mesh is refined

            # Step 2: Iterate over each bad triangle and refine its local cluster.
            for bad_triangle in bad_triangles:
                local_cluster = mesh.identify_local_cluster(bad_triangle)

                # Step 3: Construct feasible polygon within local cluster
                feasible_polygon = self.construct_feasible_cluster(local_cluster)

                # Step 4: Insert a new point in the feasible polygon
                new_point = self.insert_point_in_feasible_polygon(feasible_polygon)

                # Step 5: Add new point to mesh
                mesh.add_node(new_point)

                # Step 6: Apply local smoothening and optimization
                mesh.local_cluster_smoothening(local_cluster)

                # Mark refinement as still needed for the next iteration
                refinement_needed = True
            
            # Remove unnecessary vertices
            mesh.remove_bad_vertices()

        return mesh
    
    def construct_feasible_polygon(self, local_cluster):
        """
        Constructs a feasible polygon for point insertion by finding the innermost polygon.

        :param local_cluster: A list of triangles forming the local cluster.
        :return: A polygon that defines the region where a new point can be inserted.
        """
        circles = []
        for edge in local_cluster.boundary_edges():
            circle = self.draw_circle_with_edge_as_dimeter(edge)
            circles.append(circle)

        # Compute the intersection points of the circles
        intersection_points = self.compute_circle_intersection(circles)

        # Construct and return the innermost feasible polygon
        feasible_polygon = self.get_innermost_polygon(intersection_points)
        return feasible_polygon
    
    def insert_point_in_feasible_polygon(self, polygon: Polygon):
        """
        Randomly selects a point inside the feasible polygon and returns it.

        :param polygon: The feasible polygon where the new point will be inserted.
        :return: A tuple (x, y) representing the new point inside the polygon.
        """
        # Use the centroid-based approach for now
        centroid = polygon.centroid
        return (centroid.x, centroid.y)

    def visualize_mesh(self, optimized_points):
        """Visualize the generated mesh."""
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Plot optimized points
        x = [point[0] for point in optimized_points]
        y = [point[1] for point in optimized_points]
        ax.scatter(x, y, c='b', s=10, label='Optimized Points')

        # Plot original geometry
        lines = [[(line.start.x, line.start.y), (line.end.x, line.end.y)] for line in self.geometry.lines]
        lc = LineCollection(lines, color='r', linewidths=2)
        ax.add_collection(lc)

        # Plot initial grid points (optional)
        if hasattr(self, 'initial_points') and self.initial_points:
            initial_x = [point[0] for point in self.initial_points]
            initial_y = [point[1] for point in self.initial_points]
            ax.scatter(initial_x, initial_y, c='g', s=5, alpha=0.5, label='Initial Grid Points')

        ax.set_title('Mesh Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()

        # Set the plot limits to match the geometry bounds
        min_x, min_y, max_x, max_y = self.geometry.get_bounds()
        ax.set_xlim(min_x - 0.1, max_x + 0.1)
        ax.set_ylim(min_y - 0.1, max_y + 0.1)

        plt.show()

if __name__ == "__main__":
    geo = Geometry()
    iron = Material('Pure Iron', 5000)
    geo.add_rectangle(0.0, 0.0, 3.0, 3.0, iron)

    mesher = Mesher(geo, grid_resolution=0.1)
    
    print("Geometry points:", [(p.x, p.y) for p in geo.points])
    
    cvt_points = mesher.generate_mesh()
    print("CVT-Optimised points:", cvt_points)

    # Visualize the mesh
    mesher.visualize_mesh(cvt_points)

