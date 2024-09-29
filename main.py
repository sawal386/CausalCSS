
import statsmodels.formula.api as smf
import dowhy

from query import CausalQuery
from generator import generate_data_stress_sleep
from util import format_graph_DOT


if __name__ == "__main__":

    default_query = "Does high stress affect sleep?"
    cq = CausalQuery(default_query, factor_collider=True)
    g = cq.make_graph()

    data_df = generate_data_stress_sleep(500, cq.treatment_var, cq.outcome_var, cq.mediator,
                                         cq.confounders, cq.colliders)

    graph_dot = format_graph_DOT(g)

    cg = dowhy.CausalModel(data=data_df, treatment=cq.treatment_var, outcome=cq.outcome_var,
                           graph=graph_dot)
    identified_estimand = cg.identify_effect(proceed_when_unidentifiable=True)
    method = None
    if len(identified_estimand.get_backdoor_variables()) != 0:
        ate = cg.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        print("backdoor ate: ", ate)

    if len(identified_estimand.get_frontdoor_variables()) != 0:
        ate = cg.estimate_effect(identified_estimand, method_name='frontdoor.two_stage_regression')
        print("Frontdoor ate:", ate)


    # print("Test using regression model")
    # ols_alpha = smf.ols("{} ~ {} + {}".format(cq.outcome_var, cq.treatment_var, cq.mediator),
    #                    data=data_df).fit(cov_type="HC1")
    # ols_gamma = smf.ols("{}~{}".format(cq.mediator, cq.treatment_var),
    #                    data=data_df).fit(cov_type="HC1")
    # ate_ols = ols_alpha.params[cq.mediator] * ols_gamma.params[cq.treatment_var]

    # print("OLS: ", ate_ols)












