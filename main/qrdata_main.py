## this tests our pipeline on IHDP dataset found in QRdata.


import os 
import pandas as pd
from pathlib import Path
import json
import argparse
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query import CausalQuery 
from inference import DowhyInference

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", help="name of the dataset")
    parser.add_argument("--query", help="the causal query", default="")
    parser.add_argument("--data_folder", help="folder containing the data")
    parser.add_argument("--json_filepath", help="json files containing the necessary information")
    parser.add_argument("--output_folder", help="location where output is saved")
    parser.add_argument("--method", help="what method to use for estimation",
                        default="linear_regression")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    output_folder = Path(args.output_folder)
    output_graphs = output_folder / "graphs"
    output_folder.mkdir(exist_ok=True, parents=True)
    output_graphs.mkdir(exist_ok=True, parents=True)

    result_dict = {"data_name":[], "true":[], "predicted_backdoor":[],
                   "predicted_frontdoor":[]}

    with open(args.json_filepath, "r") as f:
        json_info = json.load(f)
    query = args.query

    count = 0
    for q in json_info:
        print("Testing data: {}".format(q["data_files"]))
        data = pd.read_csv(Path(args.data_folder) / q["data_files"][0])
        info = q['data_description']
        if len(query) != 0:
            cq = CausalQuery(query, data=data, additional_info=info)
        else:
            cq = CausalQuery(q["question"], data=data, additional_info=info)
        graph = cq.get_graph()
        infer = DowhyInference(graph, data)
        infer.identification()

        if "method" in q:
            estim = infer.backdoor_estimation(q["method"])
        else:
            estim = infer.backdoor_estimation(args.method)

        frontdoor_estim = infer.frontdoor_estimation()


        result_dict['data_name'].append(q["data_files"][0])
        result_dict['true'].append(float(q['answer']))
        result_dict['predicted_backdoor'].append(estim)
        result_dict["predicted_frontdoor"].append(frontdoor_estim)
        print("true:{}, predicted:{}, frontdoor: {}".format(q['answer'], estim, frontdoor_estim))
        print('xxxxxxxxxxxxxxxxxxxxxx')

    df = pd.DataFrame(result_dict)
    df.to_csv(output_folder/"{}.csv".format(args.data_name))
