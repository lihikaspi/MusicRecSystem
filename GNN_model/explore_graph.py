import torch
from config import config

# Path to the saved graph
graph_path = config.paths.train_graph_file

# Load the graph
data = torch.load(graph_path)

# Number of nodes
num_users = data['user'].num_nodes
num_items = data['item'].num_nodes

# Number of edges
num_edges = data['user', 'interacts', 'item'].edge_index.size(1)

print(f"Number of user nodes: {num_users}")
print(f"Number of item (song) nodes: {num_items}")
print(f"Number of edges: {num_edges}")
print("=" * 60)

# --- Example user ---
example_user_id = 0
if 'user_id' in data['user']:
    print("Example user:")
    print(f"  user_id: {data['user'].user_id[example_user_id].item()}")
    print(f"  uid: {data['user'].uid[example_user_id].item()}")
else:
    print("User IDs not found in graph data.")
print("-" * 60)

# --- Example item ---
example_item_id = 0
if 'x' in data['item']:
    print("Example item:")
    print(f"  item_id: {data['item'].item_id[example_item_id].item()}")
    print(f"  artist_id: {data['item'].artist_id[example_item_id].item()}")
    print(f"  album_id: {data['item'].album_id[example_item_id].item()}")
    print(f"  embedding (first 5 dims): {data['item'].x[example_item_id][:5]}")
else:
    print("Item features not found in graph data.")
print("-" * 60)

# --- Example edge ---
edge_index = data['user', 'interacts', 'item'].edge_index
edge_attr = data['user', 'interacts', 'item'].edge_attr
edge_weight_init = data['user', 'interacts', 'item'].edge_weight_init

example_edge_id = 0
src_user = edge_index[0, example_edge_id].item()
dst_item = edge_index[1, example_edge_id].item()
edge_features = edge_attr[example_edge_id]
edge_weight = edge_weight_init[example_edge_id]

print("Example edge:")
print(f"  source (user_idx): {src_user}")
print(f"  destination (item_train_idx): {dst_item}")
print(f"  edge_attr: {edge_features.tolist()}")
print(f"  edge_weight_init: {edge_weight.item()}")

print("=" * 60)
