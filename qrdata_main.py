## this tests our pipeline on IHDP dataset found in QRdata.

from query import CausalQuery
from inference import DowhyInference
import pandas as pd
from pathlib import Path
import json
import argparse

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", help="name of the dataset")
    parser.add_argument("--query", help="the causal query")
    parser.add_argument("--data_folder", help="folder containing the data")
    parser.add_argument("--json_filepath", help="json files containing the necessary information")
    parser.add_argument("--output_folder", help="location where output is saved")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()


    output_folder = Path(args.output_folder)
    output_graphs = output_folder / "graphs"
    output_folder.mkdir(exist_ok=True, parents=True)
    output_graphs.mkdir(exist_ok=True, parents=True)

    result_dict = {"data_name":[], "true":[], "predicted":[]}

    with open(args.json_filepath, "r") as f:
        json_info = json.load(f)
    query = args.query

    count = 0
    for q in json_info:
        print("Testing data: {}".format(q["data_files"]))
        data = pd.read_csv(Path(args.data_folder) / q["data_files"][0])
        info = q['data_description']
        cq = CausalQuery(query, data=data, other_info=info)
        graph = cq.get_graph()
        infer = DowhyInference(graph, data)
        infer.identification()
        estim = infer.backdoor_estimation("propensity_score_weighting")
        result_dict['data_name'].append(q["data_files"][0])
        result_dict['true'].append(float(q['answer']))
        result_dict['predicted'].append(float(estim))
        print("true:{}, predicted:{}".format(q['answer'], estim))
        print('xxxxxxxxxxxxxxxxxxxxxx')
        #count += 1
        #if count == 3:
        #    break
        #sys.exit()

    df = pd.DataFrame(result_dict)
    df.to_csv(output_folder/"{}.csv".format(args.data_name))




