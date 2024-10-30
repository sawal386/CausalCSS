## This file defines functions
import re

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

def filter_str(text):
    """
    removes non-alphabetical characters in the text
    Args:
        text: (str) input text

    Returns:
        (str)
    """
    cleaned = "_".join(text.strip().strip(".").split())

    return cleaned




   # return re.sub("[^a-zA-Z]+", "", text)
