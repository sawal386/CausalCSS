## This file contains classes / functions for representing prompts
from gpt import interface_gpt
class Prompt:

    """
    Base class to represent the general structure and order of prompts to ChatGPT
    for answering a causal query.

    Attributes:
        query: (str) the natural language query
        prompt0: (str) the opening prompt to setup the communication
        prompt1: (str) prompt that presents the query to GPT
        prompt2: (str) prompt asking th treatment variable
        prompt3: (str) prompt asking the outcome variable
        prompt4: (str) prompt asking other variables of interest
        prompt5: (str) prompt asking the edges
    """

    def __init__(self, query, make_small=True):
        self.query = query
        self.make_small = make_small
        self.prompt0 = "You are a helpful assistant. I am interested in building a " \
                        " causal graph to model a causal query. To help me build the graph, I will be asking you some" \
                        " questions. I need to eventually translate your text-based response to a graph. Hence, it " \
                        " will be immensely useful if your responses are concise. Further explanations and" \
                        " elaborations are not needed. Whenever possible, answer in one word. "\
                        "Also, no need to end answer with periods or enclose them with. Simple text suffices"
        self.prompt1 = 'The question of interest is: {}?. Make '.format(query)
        self.prompt2 = "What is the treatment variable?"
        self.prompt3 = "Likewise, what is the outcome variable?"
        if make_small:
            self.prompt4 = ("What are 5 other variables that should be taken into account for this model? Separate them "
                            "by ,")
        else:
            self.prompt4 = ("I want the causal graph to be informative. What would be other variables of interest? "
                        "Make sure to include forks, colliders, and mediators in addition to other variables. "
                        "Provide a comprehensive list. In the output only include the variable names.")
        self.prompt5 = ("List all the edges associated with the above variables in the graph. I want the graph to be as "
                        "expressive as possible. Include edges for mediators, colliders, and forks "
                        "Use  -> to indicate to indicate an edge between two variables. The format is node1 -> node2. "
                        "Also no need to number the edges. Include one set of edge in one line. Double check to ensure "
                        "the edges do not form a cycle.")

        if make_small:
            self.prompt6 = ("Provide me with 1 possible unobserved variables related to this model. At least 1 must be "
                           "an unobserved confounder")
        else:
            self.prompt6 = ("Could there be any unobserved variables in the model, especially unobserved confounders? "
                        "If yes, what are they? Only include the variable names in your response.")
        self.prompt7 = "List the edges involving the unobserved variables? Use -> to indicate an edge."


    def send_query_gpt(self, include_confounder=False):
        """
        Sends a sequence of queries to GPT and collects responses
        Args:
            include_confounder: (bool)
        Returns:
            (List[str]) the response from the GPT to the prompts. The indices correspond to the following information
            0: Direct answer of the causal query (Not Required. IGNORE)
            1. Treatment Variable
            2. Outcome Variable
            3. Other variables
            4. Edges between the variables
        """

        all_prompts = [self.prompt1, self.prompt2, self.prompt3, self.prompt4, self.prompt5]
        if include_confounder:
            all_prompts = all_prompts + [self.prompt6] + [self.prompt7]
        answers = []
        print("Asking GPT to help answer the query: {}".format(self.query))
        print("------------------------------------------------------")
        all_history = [ {"role": "system", "content": "Clear memory. Start fresh."},
                        {"role":"system", "content": self.prompt0}]
        for p in all_prompts:
            answer = interface_gpt(all_history, p)
            print(f"Q: {p}\nA: {answer}\n")
            all_history.append({"role": "assistant", "content": answer})
            answers.append(answer)
        print("-------------------Done----------------\n")
        return answers