import dowhy
from util import format_graph_DOT
from ananke import graphs
from ananke import identification
import matplotlib.pyplot as plt
from dowhy import gcm
import numpy as np
from ananke.models import LinearGaussianSEM
from ananke.estimation import CausalEffect

## ToDo: For now we consider the case where there is only 1 treatment and 1 outcome variable
class Inference:
    """
    base class for performing the inference
    Attributes:
        causal graph: (CausalGraph)
        data: (pd.DataFrame)
    """

    def __init__(self, causal_graph, data):

        self.causal_graph = causal_graph

        self.data = data
        self.identifiable = None
        self.estimand = None
        self.treat_var = causal_graph.get_treatment_var()
        self.outcome_var = causal_graph.get_outcome_var()

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

        self.identification()

        return self.estimation()

    def estimation(self):
        """
        estimates the treatment effect

        Returns:
            (float) the estimated treatment effect
        """

        pass

    def is_identified(self):

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

    def estimation(self):
        """
        returns the treatment effect estimation
        """
        n_treat_var = np.unique(self.data[self.treat_var].to_numpy())
        estimate = {}
        ## assuming this is binary. later we can relax this to account for non-binary cases
        if len(n_treat_var) == 2:
            estimate["do_intervention"] = self.gcm_estimation()

        estimate["backdoor"] = self.backdoor_estimation()
        estimate["frontdoor"] = self.frontdoor_estimation()

        return estimate

    def identification(self, print_=False):

        self.estimand = self.model.identify_effect(proceed_when_unidentifiable=True)
        if print_:
            print(self.estimand)

        ## ToDo: Updates with instrumental variables
        self.identifiable = (len(self.estimand.get_backdoor_variables()) != 0 or
                             len(self.estimand.get_frontdoor_variables()) != 0)

    def backdoor_estimation(self, method="linear_regression"):
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
                return ate.value
            except Exception as e:
                print("Got the following error: {}".format(e))
        else:
            return None

    ## ToDo: Add instrumental variable estimation

    def gcm_estimation(self):
        """
        estimates the causal effect using do-interventions
        Returns:
            (float)
        """

        gcm_model = gcm.StructuralCausalModel(self.causal_graph.graph)
        gcm.auto.assign_causal_mechanisms(gcm_model, self.data)
        gcm.fit(gcm_model, self.data)

        causal_effect = gcm.average_causal_effect(gcm_model, self.outcome_var,
                                                  interventions_alternative={self.treat_var: lambda x:1},
                                                  interventions_reference={"T": lambda x: 0},
                                                  num_samples_to_draw=1000)
        return causal_effect

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


def infer_causal_effect(graph, data=None):
    """
    infers the causal effect associated with the graph
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
