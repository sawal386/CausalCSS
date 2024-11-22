## Test on Benchmark Dataset 

### IHDP dataset
The IHDP dat is one the benchmark data in the QRDATA paper(https://arxiv.org/pdf/2402.17644) for estimating the ATE. To test our pipeline on the dataset, run the following 
```
bash scripts/qrdata_test.sh
```
The output file is a csv file named ihdp, which is saved in the folder: output/qrdata. The filde contains the true and predicted estimates of ATE. 
