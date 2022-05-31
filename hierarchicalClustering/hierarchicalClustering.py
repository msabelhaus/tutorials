 # Libraries
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
plt.style.use("seaborn")


# For first plot
# Create synthetic clusters
X, y = make_blobs(n_samples=10, centers=3, n_features=2, random_state=123)

# Plot
Z = linkage(X, 'complete')
plt.figure(figsize=(12, 10))
plt.title('Dendrogram')
plt.xlabel('Observation')
plt.ylabel('Distance')
plt.axhline(y=8, c='k', linestyle='--')
plt.axhline(y=5, c='b')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()









# Create synthetic clusters
X, y = make_blobs(n_samples=50, centers=3, n_features=2, random_state=123)

# Plot
plt.scatter(X[:,0], X[:,1], c=y, cmap='rainbow', edgecolor='k')
plt.title("True cluster labels")
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.25)

# Assign linkage
Z = linkage(X, 'complete')

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

fcluster(Z, t=3, criterion='maxclust')

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def 







