import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 4)])

nx.draw(G, with_labels=True)
plt.savefig("graph.png")

# print("节点数量:", G.number_of_nodes())
# print("边的数量:", G.number_of_edges())
# print("是否连通:", nx.is_connected(G))
