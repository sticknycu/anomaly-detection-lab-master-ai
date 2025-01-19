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

