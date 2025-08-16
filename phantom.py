import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def create_vector_field_phantom(shape, centers, radii, domain_vectors, transition_width=2.0):
    """
    Create a 2D vector field phantom with multiple domains that can overlap.
    All vectors have unit magnitude, with smooth transition at domain interfaces.
    Background is defined by the first domain's boundary.
    
    Args:
        shape: tuple (height, width) of the output grid
        centers: array of [(x1, y1), (x2, y2), ...] for each circle's center
                 where (0,0) is at top-left, y increases downward
        radii: array of [r1, r2, ...] for each domain's radius
        domain_vectors: array of [vec1, vec2, ...] where each is (vx, vy) in physical coordinates
        transition_width: width of the smooth transition at interfaces (in pixels)
    
    Returns:
        tuple of:
        - 3D numpy array of shape (height, width, 2) containing the vector field
        - 2D numpy array of shape (height, width) containing binary mask (1 inside, 0 outside)
    """
    # Create coordinate grids
    x, y = np.indices(shape)
    
    # Use centers directly in image coordinates
    centers_img = np.array(centers)
    
    # Initialize vector field with zeros
    vec_field = np.zeros((*shape, 2))
    
    # Calculate distance from first domain center
    r_first = np.sqrt((x - centers_img[0][0])**2 + (y - centers_img[0][1])**2)
    
    # Create binary mask (1 inside first domain, 0 outside)
    mask = (r_first <= radii[0]).astype(np.float32)
    
    # Normalize first domain vector
    vec_first = np.array(domain_vectors[0])
    vec_first = vec_first / np.linalg.norm(vec_first)
    
    # Set first domain and define background
    vec_field[r_first <= radii[0]] = vec_first  # First domain
    background_mask = r_first > radii[0]  # Background is outside first domain
    
    # Recursively add subsequent domains
    def add_domain(domain_idx):
        if domain_idx >= len(centers):
            return
        
        # Calculate distance from current domain center
        r_current = np.sqrt((x - centers_img[domain_idx][0])**2 + (y - centers_img[domain_idx][1])**2)
        
        # Update binary mask to include current domain
        mask[r_current <= radii[domain_idx]] = 1
        
        # Normalize current domain vector
        vec_current = np.array(domain_vectors[domain_idx])
        vec_current = vec_current / np.linalg.norm(vec_current)
        
        # Create smooth transition mask for current domain
        if transition_width <= 1e-10:  # Handle zero or very small transition_width
            # Sharp transition: 1 inside the domain, 0 outside
            transition = (r_current <= radii[domain_idx]).astype(float)
        else:
            # Smooth transition using error function
            transition = 0.5 * (1 + erf((radii[domain_idx] - r_current) / transition_width))
        
        # For each point in the grid
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Skip background points
                if background_mask[i,j]:
                    continue
                    
                # Only process points where there's an interface (transition region)
                if 0 < transition[i,j] < 1:
                    # Linear interpolation between vectors
                    vec_field[i,j] = (1-transition[i,j]) * vec_field[i,j] + transition[i,j] * vec_current
                    # Normalize the interpolated vector
                    vec_field[i,j] = vec_field[i,j] / (np.linalg.norm(vec_field[i,j])+1e-5)
                # If fully inside current domain, use current vector
                elif transition[i,j] >= 1:
                    vec_field[i,j] = vec_current
        
        # Recursively add next domain
        add_domain(domain_idx + 1)
    
    # Start recursion with the second domain
    add_domain(1)
    
    return vec_field, mask

def create_scalar_field_phantom(shape, centers, radii, values, transition_width=2.0):
    """
    Create a 2D scalar field phantom with multiple domains that can overlap.
    Each domain has a specific scalar value with smooth transitions at interfaces.
    Background is defined by the first domain's boundary.
    
    Args:
        shape: tuple (height, width) of the output grid
        centers: array of [(x1, y1), (x2, y2), ...] for each circle's center
                 where (0,0) is at top-left, y increases downward
        radii: array of [r1, r2, ...] for each domain's radius
        values: array of [val1, val2, ...] scalar values for each domain
        transition_width: width of the smooth transition at interfaces (in pixels)
    
    Returns:
        tuple of:
        - 2D numpy array of shape (height, width) containing the scalar field
        - 2D numpy array of shape (height, width) containing binary mask (1 inside, 0 outside)
    """
    # Create coordinate grids
    x, y = np.indices(shape)
    
    # Use centers directly in image coordinates
    centers_img = np.array(centers)
    
    # Initialize scalar field with zeros
    scalar_field = np.zeros(shape)
    
    # Calculate distance from first domain center
    r_first = np.sqrt((x - centers_img[0][0])**2 + (y - centers_img[0][1])**2)
    
    # Create binary mask (1 inside first domain, 0 outside)
    mask = (r_first <= radii[0]).astype(np.float32)
    
    # Set first domain and define background
    scalar_field[r_first <= radii[0]] = values[0]  # First domain
    background_mask = r_first > radii[0]  # Background is outside first domain
    
    # Recursively add subsequent domains
    def add_domain(domain_idx):
        if domain_idx >= len(centers):
            return
        
        # Calculate distance from current domain center
        r_current = np.sqrt((x - centers_img[domain_idx][0])**2 + (y - centers_img[domain_idx][1])**2)
        
        # Update binary mask to include current domain
        mask[r_current <= radii[domain_idx]] = 1
        
        # Create smooth transition mask for current domain
        if transition_width <= 1e-10:  # Handle zero or very small transition_width
            # Sharp transition: 1 inside the domain, 0 outside
            transition = (r_current <= radii[domain_idx]).astype(float)
        else:
            # Smooth transition using error function
            transition = 0.5 * (1 + erf((radii[domain_idx] - r_current) / transition_width))
        
        # For each point in the grid
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Skip background points
                if background_mask[i,j]:
                    continue
                    
                # Only process points where there's an interface (transition region)
                if 0 < transition[i,j] < 1:
                    # Linear interpolation between scalar values
                    scalar_field[i,j] = (1-transition[i,j]) * scalar_field[i,j] + transition[i,j] * values[domain_idx]
                # If fully inside current domain, use current value
                elif transition[i,j] >= 1:
                    scalar_field[i,j] = values[domain_idx]
        
        # Recursively add next domain
        add_domain(domain_idx + 1)
    
    # Start recursion with the second domain
    add_domain(1)
    
    return scalar_field, mask

def create_rectangular_phantom(shape, split_position, domain_vectors, transition_width=2.0):
    """
    Create a 2D vector field phantom with a rectangular shape split into two parts.
    The rectangle is 70% of the input shape size.
    Each part has its own domain vector with a smooth transition at the interface.
    
    Args:
        shape: tuple (height, width) of the output grid
        split_position: tuple (x, y) or string ('horizontal' or 'vertical') indicating how to split
                       If tuple, it's the position of the split line
                       If string, it's the type of split (horizontal or vertical)
        domain_vectors: array of [vec1, vec2] where each is (vx, vy) in physical coordinates
                       vec1 for left/top part, vec2 for right/bottom part
        transition_width: width of the smooth transition at the interface (in pixels)
    
    Returns:
        tuple of:
        - 3D numpy array of shape (height, width, 2) containing the vector field
        - 2D numpy array of shape (height, width) containing binary mask (1 inside rectangle, 0 outside)
    """
    # Create coordinate grids
    x, y = np.indices(shape)
    
    # Calculate rectangle dimensions (70% of input shape)
    rect_height = int(shape[0] * 0.7)
    rect_width = int(shape[1] * 0.7)
    
    # Calculate rectangle boundaries
    start_y = (shape[0] - rect_height) // 2
    end_y = start_y + rect_height
    start_x = (shape[1] - rect_width) // 2
    end_x = start_x + rect_width
    
    # Initialize vector field and mask
    vec_field = np.zeros((*shape, 2))
    mask = np.zeros(shape, dtype=np.float32)
    
    # Create rectangle mask
    mask[start_y:end_y, start_x:end_x] = 1
    
    # Normalize domain vectors
    vec1 = np.array(domain_vectors[0])
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = np.array(domain_vectors[1])
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Handle different types of splits
    if isinstance(split_position, str):
        if split_position.lower() == 'horizontal':
            # Split horizontally (y-axis split)
            split_line = start_y + rect_height // 2
            transition = 0.5 * (1 + erf((split_line - x) / transition_width))
        elif split_position.lower() == 'vertical':
            # Split vertically (x-axis split)
            split_line = start_x + rect_width // 2
            transition = 0.5 * (1 + erf((split_line - y) / transition_width))
        else:
            raise ValueError("split_position must be 'horizontal' or 'vertical'")
    else:
        # Custom split line (relative to rectangle)
        split_x, split_y = split_position
        # Adjust split position to be relative to rectangle
        split_x = start_x + split_x
        split_y = start_y + split_y
        # Calculate distance from split line
        transition = 0.5 * (1 + erf((split_x - x + split_y - y) / transition_width))
    
    # Apply vectors with smooth transition only within the rectangle
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mask[i,j] == 1:  # Only process points inside the rectangle
                if transition[i,j] >= 0.5:
                    vec_field[i,j] = vec1
                else:
                    vec_field[i,j] = vec2
                # Apply smooth transition
                if 0 < transition[i,j] < 1:
                    vec_field[i,j] = (1-transition[i,j]) * vec2 + transition[i,j] * vec1
                    vec_field[i,j] = vec_field[i,j] / np.linalg.norm(vec_field[i,j])
    
    return vec_field, mask

# Example usage:
if __name__ == "__main__":
    # Create a test vector field with overlapping domains
    shape = (64, 64)
    centers = np.array([(32, 32), (24 , 36),  (46, 24), (46, 42), (32, 14),])
    radii = np.array([28, 14, 7, 4,  3]) 
    domains = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, -1.0]
    ])
    
    field = create_vector_field_phantom(
        shape, centers, radii, domains, transition_width=3.0)

    # Create a rectangular phantom split horizontally
    shape = (64, 64)
    domain_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])  # Right and up vectors
    
    field, mask = create_rectangular_phantom(
        shape=shape,
        split_position='horizontal',
        domain_vectors=domain_vectors,
        transition_width=3.0
    )

    # For a horizontal split
    field, mask = create_rectangular_phantom(
        shape=(64, 64),
        split_position='horizontal',
        domain_vectors=[[1.0, 0.0], [0.0, 1.0]],  # Right vector for top, Up vector for bottom
        transition_width=3.0
    )

    # For a vertical split
    field, mask = create_rectangular_phantom(
        shape=(64, 64),
        split_position='vertical',
        domain_vectors=[[1.0, 0.0], [0.0, 1.0]],  # Right vector for left, Up vector for right
        transition_width=3.0
    )

    # For a custom split line
    field, mask = create_rectangular_phantom(
        shape=(64, 64),
        split_position=(32, 32),  # Split line passes through this point
        domain_vectors=[[1.0, 0.0], [0.0, 1.0]],
        transition_width=3.0
    )
