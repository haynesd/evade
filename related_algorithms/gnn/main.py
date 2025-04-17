import argparse
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from data_loader import getDataFromCSV


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_graph(X, y, k=5):
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y.values, dtype=torch.float)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
    knn_graph = nbrs.kneighbors_graph(X, mode='connectivity')
    coo = knn_graph.tocoo()
    edge_index = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


class GCN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x).squeeze()  # output logits (no sigmoid)


def train_and_eval(csv_path, epochs):
    print(f"[INFO] Loading data from: {csv_path}")
    X_train, y_train, X_test, y_test = getDataFromCSV(csv_path)
    train_graph = build_graph(X_train, y_train)
    test_graph = build_graph(X_test, y_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f"Using {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    model = GCN(train_graph.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # handle class imbalance
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_graph = train_graph.to(device)
    test_graph = test_graph.to(device)

    print("[INFO] Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_graph.x, train_graph.edge_index)
        loss = loss_fn(out, train_graph.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    print("\n[INFO] Running timed inference on test set...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        out = model(test_graph.x, test_graph.edge_index)
        end_time = time.time()

        probs = torch.sigmoid(out)
        preds = probs >= 0.5
        accuracy = (preds == test_graph.y.bool()).sum().item() / \
            test_graph.num_nodes
        roc_auc = roc_auc_score(
            test_graph.y.cpu().numpy(), probs.cpu().numpy())

        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(
            f"Inference Time: {(end_time - start_time)*1000:.2f} ms for {test_graph.num_nodes} nodes")

        print("\nClassification Report:")
        print(classification_report(
            test_graph.y.cpu().numpy(), preds.cpu().numpy(), digits=4))


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Run GNN-IDS on CIC-IoT2023")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CIC-IoT2023 CSV file")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    args = parser.parse_args()
    train_and_eval(args.csv, args.epochs)


if __name__ == "__main__":
    main()
