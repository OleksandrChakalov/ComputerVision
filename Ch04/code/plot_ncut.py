import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from skimage.data import astronaut
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.future import graph

# Load image
img = astronaut()

# Generate superpixels
segments = slic(img, n_segments=200, compactness=10, sigma=1)

# Compute region adjacency graph (RAG)
rag = graph.rag_mean_color(img, segments)

# Refine RAG with Normalized Cuts
num_cuts = 10
labels = graph.cut_normalized(segments, rag, num_cuts=num_cuts)

# Generate the sparse adjacency matrix
adj_matrix = csr_matrix(graph.rag.adjacency_matrix(rag))

# Refine RAG with Normalized Cuts and connectivity
labels = graph.cut_normalized(segments, rag, num_cuts=num_cuts, connectivity=adj_matrix)

# Color the labels and show the result
labels_rgb = label2rgb(labels, img, kind='avg')
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 4))

ax[0].imshow(img)
ax[0].set_title('Original image')
ax[1].imshow(labels_rgb)
ax[1].set_title('Segmented image')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
