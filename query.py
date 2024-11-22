## This files defines classes to represent queries and prompt to handle queries

from gpt import restructure_gpt_response
from graph import CausalGraph
from prompt import CausalPrompt

class CausalQuery:
    """
    base class for representing the query and the graphical structure
    Attributes:
        query: (str) the original query from the user
        data: (df) the data on which we run the inference
        hidden_vars: (bool) wehther to include hidden vars or not
    """

    def __init__(self, query, data=None, hidden_vars=False, additional_info=""):

        self.query = query
        self.prompt = CausalPrompt(query, data=data, additional_info=additional_info)
        # if data is None:
        #     data = find_data(query) # retrieve the dataset that is best for the query
        self.data = data
        self.hidden_vars = hidden_vars
        while True:
            print("Building graph")
            self.formalized_query = self.formalize_query()
            
            self.causal_graph = CausalGraph(self.formalized_query["treatment"], self.formalized_query["outcome"],
                                        self.formalized_query["other_vars"], self.formalized_query["edges"], self.data,
                                        self.formalized_query["unobserved_vars"], self.formalized_query["unobserved_edges"])
            contains_cycle = self.causal_graph.detect_cycles()
            if not contains_cycle:
                print("Graph does not contain cycles")
                break 
            else:
                print("Detected cycles. Re-creating the graph")

        self.additional_info = additional_info

    def formalize_query(self):

        ## these are some examples. The responses from gpt are structured in this format before using them to build the
        ## graph.

        cycle_response = ["yes", "A", "B", "C", "A -> B \nB -> C \nC -> A"]
        example = {"query": "yes", "treat":"email", "outcome":"payments", "covar":"opened, agreement, credit_limit, risk_score", 
                  "edges":"email -> opened \nemail -> agreement \nemail -> payments \nopened -> agreement \nagreement -> payments \ncredit_limit -> payments \n credit_limit -> agreement \nrisk_score -> payments \nrisk_score -> agreement \ncredit_limit -> risk_score"}

        ## Uncomment this
        raw_response = self.prompt.send_query_gpt(self.hidden_vars)
        print(example)
        #print(raw_response1)

        return restructure_gpt_response(example)

    def plot_graph(self):
        """
        makes a plot of the graph
        """

        self.causal_graph.plot_graph()

    def get_query(self):
        """
        returns the original query
        """

        return self.query

    def get_graph(self):
        """
        returns the causal graph
        """

        return self.causal_graph





