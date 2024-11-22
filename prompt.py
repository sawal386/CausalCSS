## This file contains classes / functions for representing prompts
from gpt import interface_gpt
import sys 

def construct_prompt_0():
    """
    Constructs the zeroth, which provides the preliminary instructions. 

    Returns:
        (str)
    """

    opening = "You are a helpful assistant. I need your help to build a causal graph to model a query."
    final_prompt = ("Please make your answer concise. Whenever possible, provide one-word or list-form "
                    "answers only. Avoid explanations or full sentences.")
    prompt_combined = opening + final_prompt

    return prompt_combined


def construct_prompt_1(query, data, additional_info):
    """
    Constructs the first prompt, which puts forward the query and other relevant information 

    Args:
        query: (str) The query of interest 
        data: (pd.DataFrame / None) the available data 
        additional_info: (str) Additional information that can give GPT more context
    
    Return:
       (str)
    """

    query_prompt = "The query of interest is: {}\n".format(query)

    data_prompt = ""
    if data is not None:
        data_prompt = ("I have a dataset that contains the following variables: {} \n".format(", ".join(list(data.columns))))
    additional_prompt = ""
    if len(additional_info) != 0:
        additional_prompt = " Here is some additional information: {} ".format(additional_info)

    closing = ("I am now going to ask you some questions. Base your responses on the information I have provided so far."
               "Don't respond to this prompt")

    return query_prompt +  data_prompt + additional_prompt + closing

def ask_treatment(data, other_info=""):
    """
    Creates the prompt that asks about the treatment variable

    Args:
        data: (pd.DataFrame) the input data 
        other_info: (str) Additional instruction to GPT specific to the treatment. 

    Returns:
        (str)
    """

    prompt = "What is the treatment variable?" + other_info
    if data is not None:
        prompt =  "What is the treatment variable based on the available data variables? " + other_info
    
    return prompt 


def ask_outcome(data, other_info):
    """
    Creates the prompt that asks about the outcome variable 

    Args:
        data: (pd.DataFrame) the input data 
        other_info: (str) Additional instruction to GPT specific to the treatment. 

    Returns:
        (str)
    """

    prompt = "What is the outcome variable? " + other_info 
    if data is not None:
        prompt = "What is the outcome variable based on the available data variables?" + other_info 
    
    return prompt 


def ask_covariates(data, other_info):
    """
    Creates the prompt that asks about other variables in the model

    Args:
        data: (pd.DataFrame) the input data 
        other_info: (str) Additional instruction to GPT specific to the treatment. 

    Returns:
        (str)
    """

 
    prompt = ("I want the causal graph to be informative. What are the other variables that we need to consider for this model? " 
              "Make sure to include forks, colliders, and mediators in addition to other variables. Provide a comprehensive list") + other_info 
    if data is not None:
        prompt = "What are the other variables in the model based the data set?" + other_info 
    
    return prompt 


def ask_edges():
    """
    Creates the prompt that asks bout the edges in the model
    """

    prompt =("List the plausible edges for the causal graph consisting based on the information so far. "
            "\nWrite output in the format node1 -> node2. Include one set of edge in one line. Avoid cycles. " )
    
    return prompt 



class CausalPrompt:

    """
    Base class to represent the general structure and order of prompts to ChatGPT
    for answering a causal query.

    Attributes:
        query: (str) the natural language query
        other_info: (str) Additional information associated with the query 
        prompt0: (str) the opening prompt 
    """

    def __init__(self, query, data, additional_info=""):

        instruction1 = "Respond with only the variable name. Avoid full sentences"
        instruction2 = "Respond with only the variable names, separated by commas"

        self.query = query
        self.other_info = additional_info

        self.prompt0 = construct_prompt_0()
        self.prompt_query = construct_prompt_1(query, data, additional_info)
        self.prompt_treat = ask_treatment(data, instruction1)
        self.prompt_out = ask_outcome(data, instruction1)
        self.prompt_covar = ask_covariates(data, instruction2)
        self.prompt_edge = ask_edges()

        self.all_query_prompts = {"query": self.prompt_query, "covar":self.prompt_covar, 
                                 "treat":self.prompt_treat, "edges":self.prompt_edge, "outcome":self.prompt_out}


    def send_query_gpt(self, include_confounder=False):
        """
        Sends a sequence of queries to GPT and collects the responses

        Args:
            include_confounder: (bool) whether to include the confounder or not 
        Returns:
            (List[str]) the response from GPT to the prompts. The indices correspond to the following information
                0: Direct answer of the causal query (Not Required. IGNORE)
                1. Treatment Variable
                2. Outcome Variable
                3. Other variables
                4. Edges between the variables
        """

        #if include_confounder:
        #    all_prompts = all_prompts + [self.prompt6] + [self.prompt7]
        answers = {}
        print("Asking GPT to help answer the query: {}".format(self.query))
        print("------------------------------------------------------")
        all_history = [ {"role": "system", "content": "Clear memory. Start fresh."},
                        {"role":"system", "content": self.prompt0}]
        order = ["query", "treat", "outcome", "covar", "edges"]
        for key in order:
            q = self.all_query_prompts[key]
            answer = interface_gpt(all_history, q)
            print(f"Q: {q}\nA: {answer}\n")
            all_history.append({"role": "assistant", "content": answer})
            answers[key] = answer
        print("-------------------Done----------------\n")
        #sys.exit()

        return answers