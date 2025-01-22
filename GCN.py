import os
import networkx as nx
import numpy as np
from torch_geometric.loader import DataLoader
import pandas as pd
from torch_geometric.data import InMemoryDataset
import torch_geometric
from model_handler import *
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool

#-------------------------------------- Hyper-Parameters Definition -----------------------------------------
BATCH_SIZE = 5
LEARNING_RATE = 0.0005
EPOCHS = 100
HIDDEN_CHANNELS = 256
WEIGHT_DECAY = 1e-4


#---------------------------------- Model and Custom Dataset Definitions -------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.lin1 = Linear(num_node_features, num_node_features)
        self.lin2 = Linear(hidden_channels//2, num_classes)
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.lin1(x)
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.lin2(x)
        return x

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        # Store the list of Data objects (graphs)
        self.data_list = data_list
        super(CustomGraphDataset, self).__init__(root=None)

    def len(self):
        # Return the number of graphs in the dataset
        return len(self.data_list)

    def get(self, idx):
        # Return the graph at index `idx`
        return self.data_list[idx]

# ---------------------- UTILS and Pre-Processing -------------------------------
def preprocess_data(graph):
    """
    Pre-processing each graph by adding node and edge attributes
    :param graph:
    :return:
    """
    # Convert the graph to a NetworkX graph
    G = nx.Graph()
    for i, node in enumerate(graph.x):
        G.add_node(i)
    for edge in graph.edge_index.t().tolist():
        G.add_edge(edge[0], edge[1])

    ### Node (Atom) Centrality Measures ###
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, tol=1e-3)
    pagerank_centrality = nx.pagerank(G)
    katz_centrality = nx.katz_centrality(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    load_centrality = nx.load_centrality(G)

    # Additional node features (clustering coefficient and weighted degree)
    clustering_coefficient = nx.clustering(G)
    weighted_degree = dict(G.degree(weight='weight'))  # Requires edge weights, if applicable

    ### Add node (atom) features ###
    centrality_features = []
    for i in range(graph.num_nodes):
        centrality_features.append([
            degree_centrality[i],
            closeness_centrality[i],
            betweenness_centrality[i],
            eigenvector_centrality[i],
            pagerank_centrality[i],
            katz_centrality[i],
            harmonic_centrality[i],
            load_centrality[i],
            clustering_coefficient[i],  # Clustering coefficient
            weighted_degree.get(i, 0),  # Weighted degree (if weights exist, otherwise 0)
        ])

    # Convert centrality features to a tensor and concatenate with existing features
    centrality_features = torch.tensor(centrality_features, dtype=torch.float)
    graph.x = torch.cat([graph.x, centrality_features], dim=1)

    ### Edge (Bond) Centrality Measures ###
    edge_betweenness = nx.edge_betweenness_centrality(G)

    # Additional edge features (edge clustering coefficient and Jaccard coefficient)
    edge_load_centrality = nx.edge_load_centrality(G)
    edge_flow_centrality = nx.edge_current_flow_betweenness_centrality(G)
    jaccard_coefficient = nx.jaccard_coefficient(G)
    jaccard_dict = {(u, v): p for u, v, p in jaccard_coefficient}  # Jaccard values for edges

    ### Add edge (bond) features ###
    edge_features = []
    for edge in graph.edge_index.t().tolist():
        u, v = sorted((edge[0], edge[1]))
        edge_features.append([
            edge_betweenness[(u, v)],
            edge_load_centrality.get((u, v), 0),  # Edge clustering coefficient
            edge_load_centrality.get((v, u), 0),  # Edge clustering coefficient
            edge_flow_centrality.get((u, v), 0),  # Edge clustering coefficient
            jaccard_dict.get((u, v), 0)  # Jaccard coefficient for the edge
        ])

    # Convert edge features to a tensor and set it as edge attributes
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    if graph.edge_attr is not None:
        graph.edge_attr = torch.cat([graph.edge_attr, edge_features], dim=1)
    else:
        graph.edge_attr = edge_features

    # Update the number of node and edge features
    graph.num_features = graph.x.shape[1]
    graph.num_edge_features = graph.edge_attr.shape[1]

    return graph


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_acc = 0
        self.counter = 0

    def __call__(self, val_loss, val_acc):
        if val_loss < self.best_loss - self.min_delta or val_acc >= self.best_acc:
            self.best_loss = min(val_loss, self.best_loss)
            self.best_acc = max(val_acc, self.best_acc)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == 3 and val_acc <=0.8:
                self.counter-=1
        return self.counter >= self.patience


def save_predictions(predictions,confs, filename):
    with open(filename, 'w') as f:
        f.write("idx,label,score\n")
        for idx, (pred,conf) in enumerate(zip(predictions,confs)):
            f.write(f"{idx},{pred},{conf}\n")


#----------------------------- Model Handler Functions ----------------------------------------
def train_model(model, optimizer, train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return model, total_loss / len(train_loader)

# Validation function
def validate(model, val_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    return correct / len(val_loader.dataset)

def predict(model, test_loader):
    # Generate predictions for test set
    model.eval()
    test_predictions = []
    pred_conf = []
    test_y = []
    with torch.no_grad():
        for data in test_loader:
            if data.y is not None:
                test_y.extend(data.y.cpu().numpy())
            out = model(data.x, data.edge_index, data.batch)
            confidence = torch.sigmoid(out).max(dim=1).values
            pred = out.argmax(dim=1)
            test_predictions.extend(pred.cpu().numpy())
            pred_conf.extend(confidence.cpu().numpy())

    if len(test_y)==len(test_predictions):
        correct = sum([1 for i in range(len(test_y)) if test_y[i] == test_predictions[i]])
        test_accuracy = correct / len(test_y)
        print(f"Test Acc. = {round(test_accuracy*100,2)}%")
    return test_predictions, pred_conf


#----------------------------- Main Functions ----------------------------------------
def run(seed):
    torch_geometric.seed_everything(41 + seed)
    torch.manual_seed(41+seed)
    # Load the dataset and split into train, val and test
    data_dir = r"C:\Users\soldier109\Documents\Technion\Semester 10\097922 - Geometric Deep Learning\Project\Code New\data"
    os.makedirs(data_dir,exist_ok=True)

    train, val,test =tuple([os.path.join(data_dir,x+".pt") for x in ['train','val','test']]) # Paths
    train,val,test = torch.load(train), torch.load(val), torch.load(test)
    sample_train_to_val = 11
    train_list = [preprocess_data(x) for x in train]
    val_list = [preprocess_data(x) for x in val] + train_list[-sample_train_to_val:]
    train_list = train_list[:-sample_train_to_val]
    test_list = [preprocess_data(x) for x in test]

    # Create custom datasets and wrap in dataloaders
    train_dataset = CustomGraphDataset(train_list)
    val_dataset = CustomGraphDataset(val_list)
    test_dataset = CustomGraphDataset(test_list)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model and optimizer setup
    model = GCN(hidden_channels=HIDDEN_CHANNELS, num_classes=2, num_node_features=train_dataset.num_features)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop - using the early stopping class as stopping criteria
    patience = 3
    min_delta = 0.001
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    for epoch in range(EPOCHS):
        model, loss = train_model(model, optimizer, train_loader)
        val_acc = validate(model, val_loader)

        if early_stopping(loss, val_acc):
            print(f"Early stopping at epoch {epoch}.\tLoss = {loss}  Accuracy = {val_acc}")
            break
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    pred, conf = predict(model, test_loader) # Create prediction for this seed
    return pred, conf, val_acc


def main(num_of_seeds=25,save_pred_and_conf_matrices=False):
    # Run for different seeds. Odd number for tie breaking.
    val_list = []
    predictions = []
    confidence = []
    for s in range(num_of_seeds):
        print(f"\nStarting iteration with seed={s}")
        pred, conf_scores, val_accuracy = run(s)
        predictions.append(pred)
        confidence.append(conf_scores)
        val_list.append(val_accuracy)
    pred_matrix = np.array(predictions).T
    conf_matrix = np.array(confidence).T
    pred_final = [np.bincount(pred_matrix[i]).argmax() for i in range(len(pred_matrix))]
    # We take only the confidence
    conf_final = [np.mean([conf_matrix[y][x] for x in range(len(conf_matrix[y])) if pred_matrix[y][x] == pred_final[y]])
                  for y in range(len(pred_final))]
    if save_pred_and_conf_matrices:
        pd.DataFrame(pred_matrix).to_csv("full_predictions.csv")  # Save predictions for each iteration and each sample
        pd.DataFrame(conf_matrix).to_csv(
            "full_confidence_scores.csv")  # Save confidence for each iteration and each sample
    save_predictions(pred_final, conf_final, 'predictions.csv')

    var_sum = 0
    for i, p in enumerate(pred_matrix):
        print(f"Index {i}: Variance: {np.var(p)}")
        var_sum += np.var(p)
    print(f"Mean variance: {var_sum / len(pred_matrix)}")

if __name__=="__main__":
    """
    !!! TO REPRODUCE ONE TIME FOR seed=42, SET num_of_seeds=1 !!!
    Set save_pred_and_conf_matrices=False to avoid creation of per seed predictions and confidences csv files.
    """
    main(num_of_seeds=25, save_pred_and_conf_matrices=False)