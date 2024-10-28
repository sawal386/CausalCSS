## This files defines classes to represent queries and prompt to handle queries

from gpt import interface_gpt, restructure_gpt_response
from graph import CausalGraph

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

    def __init__(self, query):
        self.query = query

        self.prompt0 = "You are a helpful assistant. I am interested in building a " \
                        " causal graph to model a causal query. To help me build the graph, I will be asking you some" \
                        " questions. I need to eventually translate your text-based response to a graph. Hence, it " \
                        " will be immensely useful if your responses are concise. Further explanations and" \
                        " elaborations are not needed. Whenever possible, answer in one word. "\
                        "But this does not mean, you need to compromise the quality. Also, no need to end answer with periods"
        self.prompt1 = 'The question of interest is: {}?. Make '.format(query)
        self.prompt2 = "What is the treatment variable?"
        self.prompt3 = "Likewise, what is the outcome variable?"
        self.prompt4 = ("I want the causal graph to be informative. What would be other variables of interest? "
                        "Make sure to include forks, colliders, and mediators in addition to other variables. "
                        "Provide a comprehensive list. In the output only include the variable names.")
        self.prompt5 = ("List all the edges associated with the above variables in the graph. I want the graph to be as expressive as possible."
                        " Include edges for mediators, colliders, and forks "
                        "Use  -> to indicate to indicate an edge between two variables. Also no need to number the "
                       "edges. Include one set of edge in one line. Double check to ensure the edges do not form a cycle.")

        #self.prompt5 = "Provide me the list of edges between the above variables. Use "\
        #               " -> to indicate to indicate an edge between two variables. Also no need to number the "\
        #               "edges. Include one set of edge in one line. Double check to ensure the edges do not form a cycle"
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

class CausalQuery:
    """
    base class for representing the query and the graphical structure
    Attributes:
        query: (str) the original query from the user
        data: (df) the data on which we run the inference
    """

    def __init__(self, query, data=None):

        self.query = query
        self.prompt = Prompt(query)
        # if data is None:
        #     data = find_data(query) # retrieve the dataset that is best for the query
        self.data = data
        self.formalized_query = self.formalize_query()


        self.causal_graph = CausalGraph(self.formalized_query["treatment"], self.formalized_query["outcome"],
                                        self.formalized_query["other_vars"], self.formalized_query["edges"],
                                        self.formalized_query["unobserved_vars"], self.formalized_query["unobserved_edges"])
    def formalize_query(self):

        ## this is how raw response looks like when querying via GPT
        raw_response1 = ['Variables: Stress, Sleep', "Stress", "Sleep",
                        'Exercise, Diet, Work Hours, Age, Health Conditions, Family Status, Noise Levels, Light Levels, Bed Comfort',
                        "Stress -> Exercise \nExercise -> Sleep \nStress -> Work Hours \nWork Hours -> Sleep "
                        "\nAge -> Stress \nAge -> Sleep \nHealth Conditions -> Stress \nHealth Conditions -> Sleep "
                        "\nFamily Status -> Stress \nFamily Status -> Sleep \nStress -> Diet \nDiet -> Sleep "
                        "\nNoise Levels -> Sleep \nLight Levels -> Sleep \nBed Comfort -> Sleep",
                        "Genetics, Previous Sleep Disorders, Mental Health Status", "Genetics -> Stress \nGenetics -> Sleep"
                        " \nPrevious Sleep Disorders -> Stress \nPrevious Sleep Disorders -> Sleep "
                        "\nMental Health Status -> Stress \nMental Health Status -> Sleep"]
        raw_response2 = ['Yes', 'Stress', 'Sleep', 'Age, Diet, Physical Activity, Mental Health, Job Type, Lifestyle, '
                        'Caffeine Consumption, Alcohol Consumption',
                         'Stress -> Sleep\nAge -> Stress\nAge -> Physical Activity\nDiet -> Physical Activity'
                         '\nPhysical Activity -> Stress\nPhysical Activity -> Sleep\nMental Health -> Stress'
                         '\nMental Health -> Sleep\nJob Type -> Stress\nLifestyle -> Stress\nLifestyle -> Sleep'
                         '\nCaffeine Consumption -> Stress\nCaffeine Consumption -> Sleep\nAlcohol Consumption -> Stress'
                         '\nAlcohol Consumption -> Sleep']

        ## Uncomment this
        #raw_response = self.prompt.send_query_gpt(False)
        #print(raw_response2)

        return restructure_gpt_response(raw_response2)

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

    def estimate_causal_effect(self, data=None):
        """
        estimates the causal effect via the graph
        """

        self.causal_graph.estimate_treatment_effect(data)


