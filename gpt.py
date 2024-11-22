## This file contains functions for interfacing with GPT

import openai
import os
from util import filter_str

def interface_gpt(messages, question, temperature=1, top_p=0.001):
    """
    Interfaces with GPT model to generate answer to a query via OpenAI API

    Args:
        messages: (list[dict]) past history of user prompts and GPT responses
        question: (str) the question of interest
        temperature: (float) the temperature to control the randomness
        top_p: (float) 0.5

    Returns:
        (str) response to the prompt
    """
    openai.api_key = os.getenv('OPENAI_API_KEY')

    messages.append({"role":"user", "content":question})
    try:
        response = openai.ChatCompletion.create(model="gpt-4o", messages=messages, temperature=temperature,
                                                top_p=top_p)
        answer = response['choices'][0]['message']['content'].strip()

        return answer
    except Exception as e:
        print(f"Error interfacing with GPT: {e}")


def extract_edges(edge_list):
    """
    extracts the edges and store it a format compatible with network X
    Args:
        edge_list: (List[str])
    Returns:
        List[(str, str)]
    """
    new_edge_list = []
    for edge in edge_list:
        edge_sp = edge.strip().split("->")
        if len(edge_sp) != 1:
            if len(edge_sp) == 2:
                new_edge_list.append((filter_str(edge_sp[0].strip()), filter_str(edge_sp[1].strip())))
            else:
                raise ValueError(f"{edge} is not in right format")

    return new_edge_list

def restructure_gpt_response(response):
    """
    restructures the GPT response so that it can be readily converted into a causal graph
    Args:
        response: (List[str]) list of gpt responses to prompts. For details on response structure
                   see Prompt.send_query_gpt() method.
                   ToDo: For now, we are catering this to a specific structure. Later, we hope to make this more flexible.
    Returns:
        (dict)
    """


    dict_graph = {"treatment": filter_str(response["treat"].strip()), "outcome": filter_str(response["outcome"].strip()),
                  "other_vars": [filter_str(var.strip()) for var in response["covar"].split(",")],
                  "unobserved_vars":None, "unobserved_edges":None}
    str_edge_list = response["edges"].split("\n")
    dict_graph["edges"] = extract_edges(str_edge_list)
    if len(response) > 5:
        print("Unobserved variables are also included in the model")
        dict_graph["unobserved_vars"] = [filter_str(var.strip()) for var in response[5].split(",")]
        try:
            str_conf_edge_list = response[6].split("\n")
            edges_unobserved = extract_edges(str_conf_edge_list)
            dict_graph["unobserved_edges"] = edges_unobserved
        except IndexError:
            print("No edges associated with confounding variables")


    return dict_graph
