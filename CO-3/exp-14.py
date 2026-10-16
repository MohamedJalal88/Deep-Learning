import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
time_steps = 5

for t in range(time_steps):
    G.add_node(f"X_{t}")
    G.add_node(f"h_{t}")
    G.add_node(f"Y_{t}")
    
    G.add_edge(f"X_{t}", f"h_{t}", label="W_xh")
    G.add_edge(f"h_{t}", f"Y_{t}", label="W_hy")
    if t > 0:
        G.add_edge(f"h_{t-1}", f"h_{t}", label="W_hh")

pos = {}
for t in range(time_steps):
    pos[f"X_{t}"] = (t, 0)
    pos[f"h_{t}"] = (t, 1)
    pos[f"Y_{t}"] = (t, 2)

plt.figure(figsize=(8, 4))
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue')
plt.title("Unfolded RNN Architecture Graph")
plt.savefig("exp14_unfolded_rnn.png")
plt.close()
