## This files defines classes to represent queries and prompt to handle queries

from gpt import restructure_gpt_response
from graph import CausalGraph
from prompt import Prompt

class CausalQuery:
    """
    base class for representing the query and the graphical structure
    Attributes:
        query: (str) the original query from the user
        data: (df) the data on which we run the inference
        hidden_vars: (bool) wehther to include hidden vars or not
    """

    def __init__(self, query, data=None, hidden_vars=False):

        self.query = query
        self.prompt = Prompt(query)
        # if data is None:
        #     data = find_data(query) # retrieve the dataset that is best for the query
        self.data = data
        self.hidden_vars = hidden_vars
        self.formalized_query = self.formalize_query()

        self.causal_graph = CausalGraph(self.formalized_query["treatment"], self.formalized_query["outcome"],
                                        self.formalized_query["other_vars"], self.formalized_query["edges"],
                                        self.formalized_query["unobserved_vars"], self.formalized_query["unobserved_edges"])
    def formalize_query(self):

        ## these are some examples. The responses from gpt are structured in this format before using them to build the
        ## graph.
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
        raw_response3 = ["Yes", "Stress", "Sleep", "Workload, Exercise, Mental Health, Diet, Age, Gender, "
                                           "Alcohol Consumption, Smoking, Screen Time",
                         "Stress -> Workload \nWorkload -> Exercise \nExercise -> Sleep \nStress -> Mental Health"
                         "\nMental Health -> Sleep \nStress -> Diet \nDiet -> Sleep \nAge -> Stress \nAge -> Sleep "
                         "\nGender -> Stress \nGender -> Sleep \nAlcohol Consumption -> Stress \nAlcohol Consumption -> Sleep"
                          "\nSmoking -> Stress \nScreen Time -> Stress \nScreen Time -> Sleep"]

        raw_response4 = ["Yes", "Stress", "Sleep", "Workload, Smoking, Screen Time, Diet",
                         "Workload -> Stress \nWorkload -> Sleep \nStress -> Sleep "
                         "\nDiet -> Stress \nDiet -> Sleep \nScreen Time -> Sleep \nScreen Time -> Stress "
                         "\nSleep -> Smoking \nStress -> Smoking", "Alcohol, Caffeine", "Stress -> Alcohol "
                         "\n Alcohol -> Sleep \nCaffeine -> Sleep \nCaffeine -> Stress"]
        raw_response5 = ["Yes", "Stress", "Sleep", "Alcohol", "Stress -> Alcohol"
                                                              "\nAlcohol -> Sleep"]
        raw_response6 = ["Yes", "Stress", "Sleep", "Alcohol", "Alcohol -> Sleep \nAlcohol -> Stress"]

        ## Uncomment this
        raw_response = self.prompt.send_query_gpt(self.hidden_vars)
        #print(raw_response1)

        return restructure_gpt_response(raw_response)

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
