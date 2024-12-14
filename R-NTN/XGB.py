import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import KFold
from torch_geometric.data import Data, InMemoryDataset


class CustomDataset(InMemoryDataset):
    def __init__(self, root, edge_file, node_file, transform=None, pre_transform=None):
        self.edge_file = edge_file
        self.node_file = node_file
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data = load_custom_data(self.edge_file, self.node_file)
        torch.save((self.collate([data])), self.processed_paths[0])


def load_custom_data(edge_file, node_file):
    edges = pd.read_csv(edge_file, sep=',', header=None)
    edges = edges.dropna()
    edges = edges.astype(int)
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)
    node_data = pd.read_csv(node_file, sep=',', header=None)
    x = torch.tensor(node_data.iloc[:, 1:-1].values, dtype=torch.float)
    labels = pd.to_numeric(node_data.iloc[:, -1], errors='coerce').fillna(-1).astype(int)
    y = torch.tensor(labels.values, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y, num_features=x.size(1), num_classes=len(torch.unique(y)))
    return data


edge_file = 'edge.cites'
node_file = 'Node_features.content'
root = './dataset'
data = CustomDataset(root=root, edge_file=edge_file, node_file=node_file)

X = data.x
y = data.y

model = xgb.XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.5,
    n_estimators=100,
    max_depth=5
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_list = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_list.append(y_pred)
    cr = classification_report(y_test, y_pred, digits=4)
    print(f"Classification report for fold {kf.get_n_splits(X) - len(y_pred_list)}:")
    print(cr)
    print("\n")
    y_scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)
    print(f"AUC: {auc:.4f}")
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
