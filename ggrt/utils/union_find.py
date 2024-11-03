import os


class UnionFind():
    def __init__(self, size: int, max_num_per_set=None) -> None:
        # The maximum size for each component.
        self.max_num_per_set = max_num_per_set

        # Tracking the rank for each node.
        self.ranks = [0 for i in range(size)]
        
        # Tracking the root node for each node.
        self.parents = [i for i in range(size)]
        
        self.nodes = [i for i in range(size)]
        
        # For nodes which indices are not in sequential, we map the id of each
        # node into a sequential index.
        self.node_mapper = {i:i for i in range(size)}

        # Tracking of the size of each component, such that we are able to
        # truncate too large components.
        self.component_size = {i:1 for i in range(size)}

    def init_with_nodes(self, nodes: list):
        self.node_mapper.clear()
        self.nodes = nodes
        for i, node_idx in enumerate(nodes):
            self.node_mapper[node_idx] = i
    
    def union(self, x, y):
        x = self.find_root(x)
        y = self.find_root(y)

        # If the nodes are already part of the same connected component then do nothing.
        if x == y:
            return
        
        # If merging the connected components will create a connected component larger 
        # than the maximum size then do nothing.
        if (self.max_num_per_set != None) and \
            (self.component_size[x] + self.component_size[y] > self.max_num_per_set):
            return
        
        if self.ranks[x] < self.ranks[y]:
            self.component_size[y] += self.component_size[x]
            self.parents[x] = y
        else:
            self.component_size[x] += self.component_size[y]
            self.parents[y] = x
            if self.ranks[x] == self.ranks[y]:
                self.ranks[x] += 1

    def find_root(self, x):
        idx = self.node_mapper[x]
        if self.parents[idx] == idx:
            return idx
        else:
            self.parents[idx] = self.find_root(self.nodes[self.parents[idx]])
            return self.parents[idx]

    def get_connected_components():
        return None

    def validate(self):
        union_set = {}
        for node_id in self.nodes:
            root_id = self.find_root(node_id)
            if root_id not in union_set.keys():
                union_set[root_id] = []
            union_set[root_id].append(node_id)
        
        for key in union_set.keys():
            assert len(union_set[key]) <= self.max_num_per_set, \
                f"elements number of set {key}: {len(union_set[key])}"
