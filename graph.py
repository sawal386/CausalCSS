import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from util import format_graph_DOT
import dowhy
from IPython.display import Image, display


class CausalGraph:
    """
    Base data structure to represent the causal graph
    """

    def __init__(self, treat_var, outcome_var, other_vars, edge_list, unobserved_vars=None, unobserved_edges=None):

        self.treat_var = treat_var
        self.outcome_var = outcome_var
        self.other_vars = other_vars
        self.edge_list = edge_list
        self.unobserved_vars = unobserved_vars
        self.unobserved_edges = unobserved_edges
        self.graph = nx.DiGraph()
        self.update_graph()

        print(self.graph.nodes(data=True))

    def get_treatment_var(self):
        """
        returns the treatment variable
        Returns:
            (str)
        """

        return self.treat_var

    def get_outcome_var(self):
        """
        returns the treatment variable
        Returns:
            (str)
        """

        return self.outcome_var


    def update_graph(self):
        """
        updates the nodes and edges of the graph
        """

        self.graph.add_node(self.treat_var, observed=True, treatment=True, outcome=False)
        self.graph.add_node(self.outcome_var, observed=True, treatment=False, outcome=True)
        for node in self.other_vars:
            self.graph.add_node(node, observed=True, treatment=False, outcome=False)
        self.graph.add_edges_from(self.edge_list)
        if self.unobserved_vars is not None:
            for node in self.unobserved_vars:
                self.graph.add_node(node, observed=False, treatment=False, outcome=False)
        if self.unobserved_edges is not None:
            self.graph.add_edges_from(self.unobserved_edges)

    def plot_graph(self, save_loc="figures/causal_graphs", name="sample_graph.pdf"):
        """
        plots the graph

        Args:
            save_loc: (str) path to the folder where the graph is saved
            name: (str) name of the saved file
        """
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        colors = []
        for node, attr in self.graph.nodes(data=True):
            print(node, attr)
            color = "lightblue" if attr["observed"] else "lightgrey"
            colors.append(color)

        nx.draw(self.graph, ax=axes, with_labels=True, font_size=7, node_color=colors)
        path = Path(save_loc)
        path.mkdir(exist_ok=True, parents=True)
        full_path = path / name if ".pdf" in name else path / f"{name}.pdf"
        fig.savefig(full_path)

    def generate_synthetic_data(self, size=100):
        """
        generates synthetic data via the structural equations
        Args:
            size: (int)

        Returns:
            (DataFrame)
        """

        sorted_nodes = list(nx.topological_sort(self.graph))
        data_dict_all = {}
        data_dict_observed = {}
        for node in sorted_nodes:
            incoming = self.graph.in_edges(node)
            data_dict_all[node] = np.random.normal(size=size)
            if len(incoming) != 0:
                for edge in incoming:
                    #print(node, edge[0])
                    data_dict_all[node] += data_dict_all[edge[0]]

        for node, attr in self.graph.nodes(data=True):
            if attr['observed']:
                data_dict_observed[node] = data_dict_all[node]

        return pd.DataFrame.from_dict(data_dict_observed)

    ## This will be modified. It will make use of class / methods in inference.py
    def estimate_treatment_effect(self, data_df=None, identification="backdoor",
                                  method="linear_regression"):
        """
        estimate the causal effect in the data
        Args:
            data_df: (pd.DataFrame) the data
                    ## ToDo: need to check if the data is compatible
            method: (str) the name of the estimation method
            identification: (str) the method used for identification

        Returns:
            (float) the treatment effect
        """

        if data_df is None:
            print("--------------\nData is not provided. Making inference using synthetic data of size {} "
                  "\n--------------".format(100))
            data_df = self.generate_synthetic_data()

        graph_dot = format_graph_DOT(self.graph)
        model = dowhy.CausalModel(data=data_df, treatment=self.treat_var, outcome=self.outcome_var,
                               graph=graph_dot)
        model.view_model()
        print("Displaying graphical model\n")
        display(Image(filename="causal_model.png"))

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print(identified_estimand.get_frontdoor_variables())
        print(identified_estimand.get_backdoor_variables())

        if len(identified_estimand.get_backdoor_variables()) != 0:
            print("xxxxxxxxxxxxxxxxxxxxxx\nEstimating via backdoor\nxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
            ate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
            print("backdoor ate: ", ate)

        else:
            if len(identified_estimand.get_frontdoor_variables()) != 0:
                print("xxxxxxxxxxxxxxxxxxxxxx\nEstimating via frontdoor\nxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
                if len(identified_estimand.get_frontdoor_variables()) != 0:
                    ate = model.estimate_effect(identified_estimand, method_name='frontdoor.two_stage_regression')
                    print("Frontdoor ate:", ate)

