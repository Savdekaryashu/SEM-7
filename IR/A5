# PageRank Algorithm Implementation
import numpy as np

# Step 1: Define the links between pages
# Adjacency matrix representation
# rows -> from page, columns -> to page
links = {
    'A': ['B', 'C'],
    'B': ['C'],
    'C': ['A'],
    'D': ['C']
}

pages = list(links.keys())
N = len(pages)

# Step 2: Create transition matrix (M)
M = np.zeros((N, N))

for i, p1 in enumerate(pages):
    for j, p2 in enumerate(pages):
        if p1 in links[p2]:  # if page p2 links to p1
            M[i][j] = 1 / len(links[p2])

# Step 3: Initialize parameters
d = 0.85  # Damping factor
num_iterations = 100
PR = np.ones(N) / N  # Initial PageRank for each page

# Step 4: Iteratively calculate PageRank
for _ in range(num_iterations):
    PR = (1 - d) / N + d * np.dot(M, PR)

# Step 5: Display results
print("Final PageRank Scores:")
for i, page in enumerate(pages):
    print(f"{page}: {PR[i]:.4f}")
