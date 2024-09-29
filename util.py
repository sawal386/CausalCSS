import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

def make_causal_graph(var_treatment, var_outcome, var_mediator=None, vars_confounders=None,
                      vars_collider=None, plot=False, save_loc="figures/causal_graphs",
                      name="sample_graph.pdf"):
    """
    makes a causal graph

    Args:
        var_treatment: (str) name of the treatment/intervention variable
        var_outcome: (str) name of the outcome variable
        var_mediator: (str) the mediator variable
        vars_confounders: (List[str]) the list of confounders
        vars_collider: (List[str]) the list of colliders
        plot: (bool) whether to save the plot or not
        save_loc: (str) the path to the folder where the file is saved
        name: (str) name by which the file is saved

    Returns:
        (nx.DiGraph) the DAG showing the causal relation
    """

    G = nx.DiGraph()
    edges = []
    if var_mediator is None:
        edges.append((var_treatment, var_outcome))
    else:
        edges.append((var_treatment, var_mediator))
        edges.append((var_mediator, var_outcome))

    for conf in vars_confounders:
        edges.append((conf, var_treatment))
        edges.append((conf, var_outcome))

    for coll in vars_collider:
        edges.append((var_treatment, coll))
        edges.append((var_outcome, coll))

    G.add_edges_from(edges)

    if plot:
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        nx.draw_planar(G, ax=axes, with_labels=True, font_size=7)
        path = Path(save_loc)
        path.mkdir(exist_ok=True, parents=True)
        full_path = path / name if ".pdf" in name else path / f"{name}.pdf"
        fig.savefig(full_path)

    return G

def format_graph_DOT(graph):
    """
    Re-formats the graph in DOT format. The resulting expression is formatted to be
    compatible with dowhy package

    Args:
         graph: (nx.Graph) causal graph

    Returns:
        (str) string representation of the graph
    """

    str_graph = "digraph {\n"
    for edge in graph.edges():
        str_graph += f"{edge[0]} -> {edge[1]};\n"

    str_graph += "}"

    return str_graph


