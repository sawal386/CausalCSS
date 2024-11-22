python -m main.qrdata_main \
    --data_name  "ihdp" \
    --query "What is the effect of home visit from specialist doctors on the cognitive scores of premature infants?" \
    --data_folder "/Users/sawal/Desktop/research/causal_LLM/CausalCSS/benchmark/qrdata/data" \
    --json_filepath "/Users/sawal/Desktop/research/causal_LLM/CausalCSS/benchmark/qrdata/info/ihdp_ate.json" \
    --output_folder "/Users/sawal/Desktop/research/causal_LLM/CausalCSS/output/qrdata" \
    --method "propensity_score_weighting" 




