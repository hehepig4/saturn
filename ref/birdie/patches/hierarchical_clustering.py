"""
Birdie Hierarchical Clustering for Semantic ID Generation

Modified from original BIRDIE to work with our embedding format:
- Accepts embedding path directly instead of constructing from data_path
- Uses table_embeddings.npy from our emb.py patch
- Uses table_data.json for table IDs

Usage:
    python hierarchical_clustering.py \
        --embedding_path <path>/table_embeddings.npy \
        --table_data_path <path>/table_data.json \
        --output_dir <output_dir> \
        --depth 3
"""

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize, LabelEncoder
import numpy as np
import json
import os
import argparse
import pickle
import math

# Try to use Intel optimized sklearn if available
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Using Intel optimized sklearn (sklearnex)")
except ImportError:
    print("sklearnex not available, using standard sklearn")

# cluster_node class definition (originally from cluster_tree.py)
class cluster_node:
    def __init__(self, node_id, father=None, center=None, radius=None, cohesion=None, table_ids=None):
        self.node_id = node_id
        self.father = father
        self.center = center
        self.radius = radius
        self.cohesion = cohesion
        self.table_ids = table_ids if table_ids else []
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
        child.father = self
    
    def add_table(self, table_list):
        """Add table IDs to this node"""
        if isinstance(table_list, list):
            self.table_ids.extend(table_list)
        else:
            self.table_ids.append(table_list)


def find_root(node):
    while node.father is not None:
        node = node.father
    return node


def collect_nodes_at_level(root, level, current_level=0):
    if current_level == level:
        return [root]

    nodes_at_level = []
    for child in root.children:
        nodes_at_level.extend(collect_nodes_at_level(child, level, current_level + 1))

    return nodes_at_level


def average_radius_and_cohesion(node, level):
    root = find_root(node)
    nodes_at_same_level = collect_nodes_at_level(root, level)

    if not nodes_at_same_level:
        return node.radius, node.cohesion

    radii = [n.radius for n in nodes_at_same_level if n.radius is not None]
    cohesions = [n.cohesion for n in nodes_at_same_level if n.cohesion is not None]

    avg_radius = np.nanmean(radii) if radii else node.radius
    avg_cohesion = np.nanmean(cohesions) if cohesions else node.cohesion

    avg_radius = avg_radius if avg_radius > 0 else node.radius
    avg_cohesion = avg_cohesion if avg_cohesion > 0 else node.cohesion

    if math.isnan(avg_radius) or avg_radius <= 0:
        raise ValueError(f"Invalid avg_radius: {avg_radius}")

    if math.isnan(avg_cohesion) or avg_cohesion <= 0:
        raise ValueError(f"Invalid avg_cohesion: {avg_cohesion}")

    return avg_radius, avg_cohesion


def cluster_recursion(
    x_data_pos,
    new_docid,
    indicate,
    kmeans,
    mini_kmeans,
    emd,
    father_node,
    k, c
):
    """Recursive clustering function."""
    if x_data_pos.shape[0] <= c:
        for idx, pos in enumerate(x_data_pos):
            new_docid[pos].append(idx)
        return

    if x_data_pos.shape[0] >= 1e3:
        pred = mini_kmeans.fit_predict(emd[x_data_pos])
    else:
        pred = kmeans.fit_predict(emd[x_data_pos])

    pred = LabelEncoder().fit_transform(pred)
    uni_clusters = [int(i) for i in np.unique(pred)]
    if len(uni_clusters) == 1:
        for idx, pos in enumerate(x_data_pos):
            new_docid[pos].append(idx)
        return

    for i in uni_clusters:
        pos_lists = []
        for id_, class_ in enumerate(pred):
            if class_ == i:
                pos_lists.append(x_data_pos[id_])
                new_docid[x_data_pos[id_]].append(i)
        data_pos = np.array(pos_lists)
        if 0 < len(pos_lists) <= c:
            if len(pos_lists) == 1:
                node_radius, node_cohesion = average_radius_and_cohesion(
                    father_node, indicate + 1
                )
                node = cluster_node(
                    node_id=i,
                    father=father_node,
                    center=emd[data_pos[0]],
                    radius=node_radius,
                    cohesion=node_cohesion,
                )

                node.add_table(pos_lists)
                father_node.add_child(node)
                cluster_recursion(
                    data_pos,
                    new_docid,
                    indicate + 1,
                    kmeans,
                    mini_kmeans,
                    emd,
                    node,
                    k, c
                )
            else:
                node_distance = np.linalg.norm(
                    emd[data_pos] - np.mean(emd[data_pos], axis=0), axis=1
                )
                node = cluster_node(
                    node_id=i,
                    father=father_node,
                    center=np.mean(emd[data_pos], axis=0),
                    radius=np.max(node_distance),
                    cohesion=np.mean(node_distance),
                )

                node.add_table(pos_lists)
                father_node.add_child(node)
                cluster_recursion(
                    data_pos,
                    new_docid,
                    indicate + 1,
                    kmeans,
                    mini_kmeans,
                    emd,
                    node,
                    k, c
                )
        elif len(pos_lists) > c:
            node_distance = np.linalg.norm(
                emd[data_pos] - np.mean(emd[data_pos], axis=0), axis=1
            )
            node = cluster_node(
                node_id=i,
                father=father_node,
                center=np.mean(emd[data_pos], axis=0),
                radius=np.max(node_distance),
                cohesion=np.mean(node_distance),
            )
            node.add_table(pos_lists)
            father_node.add_child(node)
            cluster_recursion(
                data_pos, new_docid, indicate + 1, kmeans, mini_kmeans, emd, node, k, c
            )

    return


def run_clustering(embedding_path, table_data_path, output_dir, k=20, c=20, seed=7, depth=3):
    """Run hierarchical clustering on table embeddings."""
    print(f"Loading embeddings from {embedding_path}")
    embeddings = np.load(embedding_path)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Load table IDs from table_data.json
    print(f"Loading table data from {table_data_path}")
    with open(table_data_path, 'r') as f:
        table_data = json.load(f)
    
    table_ids = list(table_data.keys())
    print(f"Loaded {len(table_ids)} table IDs")
    
    # Validate dimensions match
    if len(table_ids) != embeddings.shape[0]:
        raise ValueError(f"Mismatch: {len(table_ids)} tables vs {embeddings.shape[0]} embeddings")
    
    # Normalize embeddings
    embeddings = normalize(embeddings, norm="l2")
    
    # Initialize k-means
    kmeans = KMeans(
        n_clusters=k,
        max_iter=300,
        n_init=100,
        init="k-means++",
        random_state=seed,
        tol=1e-7,
    )
    
    mini_kmeans = MiniBatchKMeans(
        n_clusters=k,
        max_iter=300,
        n_init=100,
        init="k-means++",
        random_state=seed,
        batch_size=1000,
        reassignment_ratio=0.01,
        max_no_improvement=20,
        tol=1e-7,
    )
    
    # Initialize semantic IDs
    new_docid = [[] for _ in range(len(embeddings))]
    
    # Create root node
    root_center = np.mean(embeddings, axis=0)
    root_radius = np.max(np.linalg.norm(embeddings - root_center, axis=1))
    root_cohesion = np.mean(np.linalg.norm(embeddings - root_center, axis=1))
    root_node = cluster_node(
        node_id=0,
        father=None,
        center=root_center,
        radius=root_radius,
        cohesion=root_cohesion,
    )
    
    # Run recursive clustering
    print("Running hierarchical clustering...")
    cluster_recursion(
        np.array(range(len(embeddings))),
        new_docid,
        0,
        kmeans,
        mini_kmeans,
        embeddings,
        root_node,
        k, c
    )
    
    # Generate string semantic IDs
    string_semantic_id = [
        "".join([str(x).zfill(2) for x in new_docid[i]]) for i in range(len(new_docid))
    ]
    
    origin_length = len(
        set(["".join([str(x) for x in new_docid[i]]) for i in range(len(new_docid))])
    )
    final_length = len(set(string_semantic_id))
    print(f"Unique IDs: {origin_length}, Semantic IDs: {final_length}, Total: {len(new_docid)}")
    
    # Create ID map
    id_map = []
    for i in range(len(string_semantic_id)):
        id_map.append({
            "text_id": i,
            "tableID": table_ids[i],
            "semantic_id": string_semantic_id[i],
            "semantic_id_list": new_docid[i],
        })
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # Save id_map.json (for compatibility with our pipeline)
    id_map_path = os.path.join(output_dir, "id_map.json")
    with open(id_map_path, "w") as f:
        json.dump(id_map, f, indent=2)
    print(f"Saved ID map to {id_map_path}")
    
    # Also save in BIRDIE's original format (one JSON per line)
    id_map_jsonl_path = os.path.join(output_dir, "id_map_jsonl.json")
    with open(id_map_jsonl_path, "w") as f:
        for item in id_map:
            f.write(json.dumps(item) + "\n")
    
    # Save cluster tree
    tree_path = os.path.join(output_dir, "cluster_tree.pkl")
    with open(tree_path, "wb") as f:
        pickle.dump(root_node, f)
    print(f"Saved cluster tree to {tree_path}")
    
    return id_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical clustering for semantic IDs")
    parser.add_argument("--embedding_path", required=True, type=str,
                        help="Path to table_embeddings.npy")
    parser.add_argument("--table_data_path", required=True, type=str,
                        help="Path to table_data.json")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory for semantic IDs")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of clusters for k-means (default: 20)")
    parser.add_argument("--c", type=int, default=20,
                        help="Minimum cluster size threshold (default: 20)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed (default: 7)")
    parser.add_argument("--depth", type=int, default=3,
                        help="Maximum clustering depth (not directly used, for compatibility)")
    
    args = parser.parse_args()
    
    run_clustering(
        args.embedding_path,
        args.table_data_path,
        args.output_dir,
        args.k,
        args.c,
        args.seed,
        args.depth
    )
