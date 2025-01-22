import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = np.loadtxt("ca-Astroph.txt")

print(dataset.shape)

G = nx.Graph()
i = 0
for idx, e in enumerate(dataset):
    # if idx+1 == int(len(dataset)/10000):
    #     pos = nx.spring_layout(G)
    #     nx.draw(G, with_labels=True)
    #     plt.show()
    print(f'Procesez {idx+1}/{len(dataset)}')
    i = i + 1
    G.add_edge(e[0], e[1], weight=i)

features = {}
for node in G.nodes():
    egonet = nx.ego_graph(G, node)

    Ni = len(list(egonet.neighbors(node)))

    Ei = egonet.number_of_edges()

    Wi = sum(data['weight'] for _, _, data in egonet.edges(data=True))

    adjacency_matrix = nx.to_numpy_array(egonet, weight='weight')
    eigenvalues = np.linalg.eigvals(adjacency_matrix)
    lambda_wi = max(eigenvalues)

    features[node] = {
        'Ni': Ni,
        'Ei': Ei,
        'Wi': Wi,
        'lambda_wi': lambda_wi.real
    }

nx.set_node_attributes(G, features)

print(G)
data = pd.DataFrame.from_dict(features, orient='index')

Ei = np.array(data['Ei'])
Ni = np.array(data['Ni'])
y = Ei * Ni

log_Ei = np.log(Ei)
log_Ni = np.log(Ni)

X = np.column_stack((y, log_Ei, log_Ni))
model = LinearRegression()
model.fit(X, y)
log_C = np.log(model.coef_)
print(model.intercept_)
print(model.coef_)

C = np.exp(log_C)

print(f'C = {C}')
print(f'log_C = {log_C}')
print(f'log_Ei = {log_Ei}')
print(f'log_Ni = {log_Ni}')

predicted_log_y = model.predict(X)
print(f'predicted_log_y = {predicted_log_y}')
predicted_y = np.exp(predicted_log_y)

anomaly_scores = (np.maximum(y, predicted_y) / np.minimum(y, predicted_y)) * np.log(np.abs(y - predicted_y) + 1)

print("Anomaly Scores:", anomaly_scores)


# --- Part 1: Exercise 2.2 ---
# Generating a regular graph and merging with cliques
def generate_graph_with_cliques():
    regular_graph = nx.random_regular_graph(3, 100)  # Regular graph with degree 3
    caveman_graph = nx.connected_caveman_graph(10, 20)  # 10 cliques with 20 nodes each

    merged_graph = nx.union(regular_graph, caveman_graph)

    # Add random edges to connect the graph
    while not nx.is_connected(merged_graph):
        node1 = np.random.choice(list(merged_graph.nodes))
        node2 = np.random.choice(list(merged_graph.nodes))
        merged_graph.add_edge(node1, node2)

    return merged_graph

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

# Detecting anomalies using Ei and Ni
def detect_clique_anomalies(G):
    features = {}
    for node in G.nodes():
        egonet = nx.ego_graph(G, node)

        Ni = len(list(egonet.neighbors(node)))
        Ei = egonet.number_of_edges()

        features[node] = {'Ni': Ni, 'Ei': Ei}

    nx.set_node_attributes(G, features)

    data = pd.DataFrame.from_dict(features, orient='index')
    top_clique_nodes = data.sort_values(by=['Ei', 'Ni'], ascending=False).head(10).index

    return top_clique_nodes


# Visualization
def visualize_cliques(G, highlighted_nodes):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color='lightblue', with_labels=False)
    nx.draw_networkx_nodes(G, pos, nodelist=highlighted_nodes, node_color='red')
    plt.show()


# Running Part 1
G = generate_graph_with_cliques()
highlighted_nodes = detect_clique_anomalies(G)
visualize_cliques(G, highlighted_nodes)


# --- Part 2: HeavyVicinity anomalies ---
def generate_heavy_vicinity_graph():
    regular_graph_3 = nx.random_regular_graph(3, 100)  # Regular graph with degree 3
    regular_graph_5 = nx.random_regular_graph(5, 100)  # Regular graph with degree 5

    merged_graph = nx.union(regular_graph_3, regular_graph_5)

    for edge in merged_graph.edges():
        merged_graph[edge[0]][edge[1]]['weight'] = 1

    # Adding heavy egonets
    random_nodes = np.random.choice(list(merged_graph.nodes), 2, replace=False)
    for node in random_nodes:
        for neighbor in merged_graph[node]:
            merged_graph[node][neighbor]['weight'] += 10

    return merged_graph, random_nodes


# Detecting HeavyVicinity anomalies
def detect_heavy_vicinity_anomalies(G):
    features = {}
    for node in G.nodes():
        egonet = nx.ego_graph(G, node)

        Wi = sum(data['weight'] for _, _, data in egonet.edges(data=True))
        Ei = egonet.number_of_edges()

        features[node] = {'Wi': Wi, 'Ei': Ei}

    nx.set_node_attributes(G, features)

    data = pd.DataFrame.from_dict(features, orient='index')
    top_heavy_nodes = data.sort_values(by=['Wi', 'Ei'], ascending=False).head(4).index

    return top_heavy_nodes


# Visualization
def visualize_heavy_vicinity(G, highlighted_nodes):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color='lightblue', with_labels=False)
    nx.draw_networkx_nodes(G, pos, nodelist=highlighted_nodes, node_color='orange')
    plt.show()


# Running Part 2
G_hv, random_nodes = generate_heavy_vicinity_graph()
highlighted_nodes_hv = detect_heavy_vicinity_anomalies(G_hv)
visualize_heavy_vicinity(G_hv, highlighted_nodes_hv)


# --- Part 3: Exercise 2.3 ---
# Designing a Graph Autoencoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        return x


class AttributeDecoder(nn.Module):
    def __init__(self, hidden_dim2, hidden_dim1, output_dim):
        super(AttributeDecoder, self).__init__()
        self.conv1 = GCNConv(hidden_dim2, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, output_dim)

    def forward(self, z, edge_index):
        x_hat = self.conv1(z, edge_index)
        x_hat = torch.relu(x_hat)
        x_hat = self.conv2(x_hat, edge_index)
        return x_hat


class StructureDecoder(nn.Module):
    def __init__(self, hidden_dim2):
        super(StructureDecoder, self).__init__()
        self.conv = GCNConv(hidden_dim2, hidden_dim2)

    def forward(self, z, edge_index):
        z = self.conv(z, edge_index)
        z = torch.relu(z)
        a_hat = torch.matmul(z, z.T)
        return a_hat


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(GraphAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2)
        self.attr_decoder = AttributeDecoder(hidden_dim2, hidden_dim1, input_dim)
        self.struct_decoder = StructureDecoder(hidden_dim2)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.attr_decoder(z, edge_index)
        a_hat = self.struct_decoder(z, edge_index)
        return x_hat, a_hat


# Loss function
def loss_function(x, x_hat, a, a_hat, alpha=0.8):
    attr_loss = torch.norm(x - x_hat, p='fro')
    struct_loss = torch.norm(a - a_hat, p='fro')
    return alpha * attr_loss + (1 - alpha) * struct_loss


# Training procedure
def train_gae(model, data, epochs=100, lr=0.004):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x, edge_index, a = data

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_hat, a_hat = model(x, edge_index)
        loss = loss_function(x, x_hat, a, a_hat)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            with torch.no_grad():
                reconstruction_error = torch.norm(x - x_hat, p='fro')
                auc_score = roc_auc_score(x.flatten().cpu(), x_hat.flatten().cpu())
                print(f"Epoch {epoch}, Loss: {loss.item()}, AUC: {auc_score}")


# Load ACM dataset
data = loadmat('ACM.mat')
x = torch.tensor(data['Attributes'].toarray(), dtype=torch.float)
a = torch.tensor(data['Network'].toarray(), dtype=torch.float)
edge_index, _ = from_scipy_sparse_matrix(data['Network'])

# Preparing the data and model
input_dim = x.shape[1]
hidden_dim1 = 128
hidden_dim2 = 64
model = GraphAutoencoder(input_dim, hidden_dim1, hidden_dim2)
data = (x, edge_index, a)

# Train the model
train_gae(model, data)

