import numpy as np
seed = 0
rv = np.random.RandomState(seed)

def generate_sequences(n=128):
    """
    Generates sequences of points forming squares, either clockwise or counter-clockwise.

    Each sequence consists of points that represent the corners of a square. The sequence can
    either go around the square in a clockwise or counter-clockwise direction. This is determined
    randomly for each sequence. The sequences are also slightly randomized by adding a small
    noise to each point.

    Args:
        n (int): The number of sequences to generate. Default is 128.

    Returns:
        tuple: A tuple containing two elements:
               - A list of arrays, where each array represents a sequence of points (corners of a square).
               - An array indicating the direction of each sequence (0 for counter-clockwise, 1 for clockwise).

    Example:
        >>> sequences, directions = generate_sequences(n=5)
        >>> print(sequences[0])  # Prints first sequence of points
        >>> print(directions[0])  # Prints direction of the first sequence (0 or 1)
    """
    basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    bases = rv.randint(4, size=n)  # Starting corner indices for each sequence.
    directions = np.random.randint(2, size=n)  # Direction (0 for CCW, 1 for CW) for each sequence.

    # Generating the point sequences.
    points = [basic_corners[[(b + i) % 4 for i in range(4)]][::d*2-1] + np.random.randn(4, 2) * 0.1
              for b, d in zip(bases, directions)]

    return points, directions