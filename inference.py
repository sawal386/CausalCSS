import dowhy
from util import format_graph_DOT
from ananke import graphs
from ananke import identification
import matplotlib.pyplot as plt
from dowhy import gcm
import numpy as np
from ananke.models import LinearGaussianSEM
from ananke.estimation import CausalEffect


class Estimator:
    """
    a wrapper class for representing all estimators
    Attributes:
        method_name: (str) the name of the estimation method 
        ate: () the average treatment effect 
        test: () the refutation test 

    """

    def __init__(self, ate, test_result):

        self.ate = ate 
        self.test = test_result 
    
    def return_ate(self):
        """
        Returns:
            (float / None): the average treatment effect
        """

        return self.ate.value if self.ate is not None else None 
    
    def get_test_result(self):
        """
        Returns:
            (float / None)
        """

        return self.test_result 


## ToDo: For now we consider the case where there is only 1 treatment and 1 outcome variable
class Inference:
    """
    base class for performing the inference
    Attributes:
        causal graph: (CausalGraph)
        data: (pd.DataFrame)
    """

    def __init__(self, causal_graph, data, refute_method="bootstrap_refuter"):

        self.causal_graph = causal_graph
        self.data = data
        self.identifiable = None
        self.estimand = None
        self.treat_var = causal_graph.get_treatment_var()
        self.outcome_var = causal_graph.get_outcome_var()
        self.refute_method = refute_method 

    def full_inference_pipeline(self):
        """
        performs the full inference by invoking both identification and estimation
        """

        self.identification()

        return self.estimation()

    def identification(self, print_=False):
        """
        performs identification
        """

        pass 

    def estimation(self):
        """
        estimates the treatment effect

        Returns:
            (float) the estimated treatment effect
        """

        pass

    def is_identified(self):
        """
        whether the causal effect is identifiable or not 
        Returns:
            (bool)
        """

        return self.identifiable


class DowhyInference(Inference):
    """
    performs inference using the DoWhy package
    """

    def __init__(self, causal_graph, data):

        super().__init__(causal_graph, data)
        self.model = dowhy.CausalModel(data=data, treatment=self.treat_var, outcome=self.outcome_var,
                                       graph=format_graph_DOT(self.causal_graph.graph))

        self.model.view_model(layout="dot")  # This generates the graph
        #im_graph.draw("graph_plots/dowhy_model.png", prog="dot")


    def identification(self, print_=True):
        """
        Performs identification using DoWhy
        """

        self.estimand = self.model.identify_effect(proceed_when_unidentifiable=True)
        if print_:
            print(self.estimand)

        ## ToDo: Updates with instrumental variables
        self.identifiable = (len(self.estimand.get_backdoor_variables()) != 0 or
                             len(self.estimand.get_frontdoor_variables()) != 0)

    def estimation(self, adjustments=["backdoor", "frontdoor", "iv"], 
                  method_back="linear_regression", method_front="linear_regression",
                  method_iv="iv.instrumental_variable"):
        """
        returns the treatment effect for different methods 
        """

        estimates = {}
        for method in adjustments:
            try:
                if method == "backdoor":
                    estimates[method] = self.backdoor_estimation(method_back)
                elif method == "frontdoor":
                    estimates[method] = self.frontdoor_estimation(method_front)
                elif method == "iv":
                    estimates[method] = self.iv_estimation()
                else:
                    raise ValueError(f"{method} is not a valid adjustment crieria")
            except Exception as e:
                estimates[method] = None 
                raise Exception("Error")
        
        return estimates


    def backdoor_estimation(self, method="propensity_score_weighting"):
        """
        Estimates the causal effect using backdoor criterion
        Args:
            method: (str)
        Returns:
            (float / None)
        """

        method_name = "backdoor.{}".format(method)

        if len(self.estimand.get_backdoor_variables()) != 0:
            try:
                ate = self.model.estimate_effect(self.estimand, method_name=method_name)
                #refuter = self.model.refute_estimate(self.estimand, ate, 
                #                                     method_name=self.refute_method) 
                #estimator = Estimator(ate.value, refutation)

                return ate.value 

            except Exception as e:
                print("Got the following error: {}".format(e))
        else:
            return None

    def frontdoor_estimation(self, method="linear_regression"):
        """
        Estimates the causal effect using frontdoor criterion
        Args:
            method: (str)

        Returns:
            (float/None)
        """

        method_name = "frontdoor.{}".format(method)
        if len(self.estimand.get_frontdoor_variables()) != 0:
            try:
                ate = self.model.estimate_effect(self.estimand,
                                                 method_name=method_name)
                #refuter = self.model.refute_estimate(self.estimand, ate, 
                #                                     method_name=self.refute_method) 
                #estimator = Estimator(ate.value, refutation)

                return ate.value 

            except Exception as e:
                print("Got the following error: {}".format(e))
        else:
            return None

    ## ToDo: Add instrumental variable estimation

    def iv_estimation(self):
        """
        Estimates the causal effect using instrumental variable"

        Returns:
        (float/None)
        """

        method_name = "iv.instrumental_variable"
        if len(self.estimand.get_instrumental_variables()) != 0:
            try:
                ate = self.model.estimate_effect(self.estimand, 
                                                 method_name=method_name)
                #refuter = self.model.refute_estimate(self.estimand, ate, 
                #                                     method_name=self.refute_method) 
                #estimator = Estimator(ate.value, refutation)

                return ate.value 

            except Exception as e:
                print("Got the following error: {}".format(e))
        else:
            return None 
        


class AnankeInference(Inference):
    """
    base class for inference using Ananke. We will be using this for the cases where
    the usual identification fails. This can often happen in cases with unobserved confounding
    """

    def __init__(self, causal_graph, data):
        super().__init__(causal_graph, data)
        vertices, di_edges, bi_edges = self.causal_graph.create_ananke_inputs()
        bi_edges = [(self.causal_graph.get_treatment_var(), self.causal_graph.get_outcome_var())]

        self.model = graphs.ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
        g_draw = self.model.draw()
        g_draw.render(filename='my_graph', directory="graph_plots", cleanup=False)


    def identification(self, print_=False):

        one_line_id = identification.OneLineID(graph=self.model,
                                    treatments=[self.treat_var], outcomes=[self.outcome_var])
        self.identifiable = one_line_id.id()
        if self.identifiable:
            print("Identification works. The functinal form is: {}".format(one_line_id.functional()))

    def estimation(self, method="eff-aipw"):
        """
        performs treatment effect estimation using Arid graphs + SEMs
        Args:
            method: (str) what estimator method to use. For now, this is for identifiable cases only
        Returns:
            (float)
        """

        ate = {}
        if not self.identifiable:
            print("Usual identification failed. We will be using ARID graphs + SEMs")
            g_arid = self.model.maximal_arid_projection()
            g_arid_draw = g_arid.draw(direction="LR")
            g_arid_draw.render(filename='my_arid_graph', directory="graph_plots", cleanup=False)
            self.model = LinearGaussianSEM(g_arid)
            self.model.fit(self.data)
            model_draw = self.model.draw(direction="LR")
            model_draw.render(filename='model_arid_graph', directory="graph_plots", cleanup=False)
            ate["sem"] = self.model.total_effect([self.treat_var], [self.outcome_var])

        else:
            causal_effect = CausalEffect(graph=self.model, treatment=self.treat_var,
                                         outcome=self.outcome_var)
            print(self.data)
            print(method)
            ate["front/backdoor"] = causal_effect.compute_effect(self.data, method)

        return ate


def infer_causal_effect(graph, data=None, method="linear_regression"):
    """
    Infers the causal effect associated with the graph
    Args:
        graph: (CausalGraph)
        data: (pd.DataFrame)

    Returns:
        (Dict) the causal effect
    """

    if data is None:
        data = graph.generate_synthetic_data()
    tool = DowhyInference(graph, data)
    tool.identification(True)

    if tool.is_identified():
        print("Inference via DoWhy")
        return tool.estimation()
    else:
        print("Do Why based identification fails. We will now use Ananke based inference")
        tool = AnankeInference(graph, data)
        tool.identification()
        return tool.estimation()
