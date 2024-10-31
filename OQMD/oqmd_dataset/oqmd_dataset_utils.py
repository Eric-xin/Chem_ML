import numpy as np

def load_graph_data(graph_data_path):
  """
  This function loads graph data from the specified file.

  Example:
    >>> graphs = load_graph_data("oqmd_dataset/graph_data.npz")
    >>> oqmd_id = "oqmd-758369"
    >>> nodes, neighbors = graphs[oqmd_id]
    >>> nodes
    [38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39]
    >>> neighbors
    [[1, 6, 6, 7, 7, 13, 13],
     [0, 0, 2, 7, 7, 11, 11, 13, 13, 14, 14],
     [1, 3, 3, 4, 4, 11, 11, 14, 14, 15, 15],
    ...
  """
  graphs = np.load(graph_data_path, allow_pickle=True, encoding='latin1')['graph_dict'].item()
  graphs = { k.decode() : v for k, v in graphs.items() }
  return graphs
