from query import CausalQuery
from inference import infer_causal_effect
if __name__ == "__main__":



    default_query = "Does stress affect sleep?"
    cq = CausalQuery(default_query, hidden_vars=True)
    graph = cq.get_graph()
    ce = infer_causal_effect(graph)
    print("causal effect: {}".format(ce))






