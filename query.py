## This files defines classes to represent queries

from util import make_causal_graph
class Prompt:
    ####################
    # Note: For test purposes, I have set the default prompt to be:
    # "Does high stress affect sleep?"
    ###################

    """
    base class that to represent the general structure and order of prompts to ChatGPT
    for answering a causal query

    Attributes:
        query: (str) the original query from the user
    """

    def __init__(self, query):
        self.query = query
        self.prompt0 = "Clear Memory"
        self.prompt_query = 'I am interested in answering the following question using tools from causal inference'\
                                ':"{}"? Give a precise answer to the next prompts. Do not elaborate.'.format(query)
        self.prompt_treat = "What is the independent/treatment variable?"
        self.prompt_outcome = "What is the outcome variable?"
        self.prompt_confounder = "What are some of the confounders?" ## can restrict the number
        self.prompt_mediator = "What are some of the mediators?"
        self.prompt_collider = "What are some of the colliders?"

        # now we return the answers based on the response

    def setup(self):
        ## sends prompt0 and prompt_query to the llm
        pass

    def get_treatment_var(self, default="s"):
        ## asks the llm the treatment variable

        return "stress"

    def get_outcome_var(self):
        ## asks the llm about the independent variable

        return "sleep"

    def get_mediator(self):
        ## asks the llm the mediator variable

        return "cortisol_level"

    def get_confounders(self):
        ## asks the llm about the confounding variables

        return ["exercise"]

    def get_colliders(self):
        ## asks the colliders

        return ["fatigue"]


class CausalQuery:
    """
    base class for representing the query and the graphical structure
    Attributes:
        query: (str) the original query from the user
        data: (df) the data on which we run the inference
        treatment_var: (str) the treatment variable
        outcome_var: (str) the outcome variable
        mediator: (str) the mediator between the treatment and outcome variables;
                    T -> M -> Y
        confounders: (List[str]) the list of confounding variables; T <- C -> Y
        colliders: (List[str]) the list of colliders; T -> V <- Y
    """

    def __init__(self, query, factor_collider=False, data=None):

        self.query = query
        self.prompt = Prompt(query)
        # if data is None:
        #     data = find_data(query) # retrieve the dataset that is best for the query
        self.data = data

        ## for now, the model variables are obtained by prompting an LLM. Alternatively, one could get the variables
        # from the data by doing something  like,
        ## model_variables = get_model_variables(data)
        ## self.treatment_var, self.outcome_var = model_variables["T"], model_variables["Y"]

        self.treatment_var = self.prompt.get_treatment_var()
        self.outcome_var = self.prompt.get_outcome_var()
        self.mediator = self.prompt.get_mediator()
        self.confounders = self.prompt.get_confounders()
        if factor_collider:
            self.colliders = self.prompt.get_colliders()
        else:
            self.colliders = []

    def make_graph(self, plot_graph=True):
        """
        makes the causal graph
        """
        causal_graph = make_causal_graph(self.treatment_var, self.outcome_var, self.mediator,
                                         self.confounders, self.colliders, plot=plot_graph)

        return causal_graph
