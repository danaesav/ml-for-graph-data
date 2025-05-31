import glob
import re
import statistics

# Get a list of all files on the dataset

#################################################### Our Model HomoLevel #################################################
# data_names = ["tmp_hyper_HomoLevel_02_linear_clean", "tmp_hyper_HomoLevel_02_MLP_clean", "tmp_hyper_HomoLevel_02_MLP-1_clean", 
#              "tmp_hyper_HomoLevel_04_linear_clean", "tmp_hyper_HomoLevel_04_MLP_clean", "tmp_hyper_HomoLevel_04_MLP-1_clean",
#              "tmp_hyper_HomoLevel_06_linear_clean", "tmp_hyper_HomoLevel_06_MLP_clean", "tmp_hyper_HomoLevel_06_MLP-1_clean",
#              "tmp_hyper_HomoLevel_08_linear_clean", "tmp_hyper_HomoLevel_08_MLP_clean", "tmp_hyper_HomoLevel_08_MLP-1_clean",
#              "tmp_hyper_HomoLevel_10_linear_clean", "tmp_hyper_HomoLevel_10_MLP_clean", "tmp_hyper_HomoLevel_10_MLP-1_clean",]


########################################### HomoLevel Baselines #######################################################
# data_names = [#"GCN_hyper_homo02", "GAT_hyper_homo02", "SAGE_hyper_homo02", "H2GCN_hyper_homo02", 
#               #"GCNLPA_hyper_homo02", "MLP_hyper_homo02", "LANC_hyper_homo02",
#               "IDGNN_hyper_homo02",
             
#             # "GCN_hyper_homo04", "GAT_hyper_homo04", "SAGE_hyper_homo04", "H2GCN_hyper_homo04", 
#             #"GCNLPA_hyper_homo04", "MLP_hyper_homo04", "LANC_hyper_homo04",
#              "IDGNN_hyper_homo04",
             
#             #"GCN_hyper_homo06", "GAT_hyper_homo06", "SAGE_hyper_homo06", "H2GCN_hyper_homo06", 
#             #"GCNLPA_hyper_homo06", "MLP_hyper_homo06", "LANC_hyper_homo06",
#              "IDGNN_hyper_homo06",
             
#             #"GCN_hyper_homo08", "GAT_hyper_homo08", "SAGE_hyper_homo08", "H2GCN_hyper_homo08", 
#             #"GCNLPA_hyper_homo08", "MLP_hyper_homo08", "LANC_hyper_homo08",
#             "IDGNN_hyper_homo08",
             
#             #"GCN_hyper_homo10", "GAT_hyper_homo10", "SAGE_hyper_homo10", "H2GCN_hyper_homo10",
#             #"GCNLPA_hyper_homo10", "MLP_hyper_homo10", "LANC_hyper_homo10",
#             "IDGNN_hyper_homo10"]

##################################################### Baseline FeatNoise #################################################
#data_names = [#"GCN_hyper_FeatureNoise_0.0", "GAT_hyper_FeatureNoise_0.0", "SAGE_hyper_FeatureNoise_0.0", 
             #"H2GCN_hyper_FeatureNoise_0.0", "GCNLPA_hyper_FeatureNoise_0.0", "MLP_hyper_FeatureNoise_0.0", 
             #"LANC_hyper_FeatureNoise_0.0", "IDGNN_hyper_FN0.0",
             
             #"GCN_hyper_FeatureNoise_0.2", "GAT_hyper_FeatureNoise_0.2", "SAGE_hyper_FeatureNoise_0.2", 
             #"H2GCN_hyper_FeatureNoise_0.2", "GCNLPA_hyper_FeatureNoise_0.2", "MLP_hyper_FeatureNoise_0.2", 
             #"LANC_hyper_FeatureNoise_0.2", "IDGNN_hyper_FN0.2",
             
             #"GCN_hyper_FeatureNoise_0.5", "GAT_hyper_FeatureNoise_0.5", "SAGE_hyper_FeatureNoise_0.5", 
             #"H2GCN_hyper_FeatureNoise_0.5", "GCNLPA_hyper_FeatureNoise_0.5", "MLP_hyper_FeatureNoise_0.5", 
             #"LANC_hyper_FeatureNoise_0.5", "IDGNN_hyper_FN0.5",
             
            #"GCN_hyper_FeatureNoise_0.8", "GAT_hyper_FeatureNoise_0.8", "SAGE_hyper_FeatureNoise_0.8", 
            #"H2GCN_hyper_FeatureNoise_0.8", "GCNLPA_hyper_FeatureNoise_0.8", "MLP_hyper_FeatureNoise_0.8", 
            #"LANC_hyper_FeatureNoise_0.8", "IDGNN_hyper_FN0.8",
             
            #"GCN_hyper_FeatureNoise_1.0", "GAT_hyper_FeatureNoise_1.0", "SAGE_hyper_FeatureNoise_1.0", 
            #"H2GCN_hyper_FeatureNoise_1.0", "GCNLPA_hyper_FeatureNoise_1.0", "MLP_hyper_FeatureNoise_1.0", 
            #"LANC_hyper_FeatureNoise_1.0", "IDGNN_hyper_FN1.0",
            #]

##################################################### Our Model FeatNoise #################################################
# data_names = ["tmp_hyper_FeatureNoise_0.0_linear", "tmp_hyper_FeatureNoise_0.0_MLP-1", "tmp_hyper_FeatureNoise_0.0",
#               "tmp_hyper_FeatureNoise_0.2_linear", "tmp_hyper_FeatureNoise_0.2_MLP-1", "tmp_hyper_FeatureNoise_0.2",
#               "tmp_hyper_FeatureNoise_0.5_linear", "tmp_hyper_FeatureNoise_0.5_MLP-1", "tmp_hyper_FeatureNoise_0.5",
#               "tmp_hyper_FeatureNoise_0.8_linear", "tmp_hyper_FeatureNoise_0.8_MLP-1", "tmp_hyper_FeatureNoise_0.8",
#               "tmp_hyper_FeatureNoise_1.0_linear", "tmp_hyper_FeatureNoise_1.0_MLP-1", "tmp_hyper_FeatureNoise_1.0"]


################################################# Real-world ML Graphs ###################################
data_names = ["blog_MLP-1", "blog", "blogcatalog_linear",
              "DBLP_MLP-1", "DBLP_linear", "DBLP",
              "pcg", "pcg_MLP-1", "pcg_linear",
              "yelp", "yelp_linear", "yelp_MLP-1",
              #"IDGNN_yelp", "IDGNN_pcg"
              #  "LSPE_pcg", "LSPE_dblp"
             ]

################################################# FSGNN on all datasets ###################################
# data_names = ["FSGNN_blog", "FSGNN_dblp", "FSGNN_yelp", "FSGNN_pcg",
#               "FSGNN_hyper_FeatureNoise_0.0", "FSGNN_hyper_FeatureNoise_0.2", "FSGNN_hyper_FeatureNoise_0.5", "FSGNN_hyper_FeatureNoise_0.8", "FSGNN_hyper_FeatureNoise_1.0",
#               "FSGNN_hyper_homo02", "FSGNN_hyper_homo04", "FSGNN_hyper_homo06", "FSGNN_hyper_homo08", "FSGNN_hyper_homo10"]


for data_name in data_names:
    # majority vote
    if "cora" in data_name or "citeseer" in data_name:
        pattern = r"Average precision: (\d+\.\d+)"
    else:
        pattern = r"Test Average Precision Score: (\d+\.\d+)"

    #files = glob.glob("vary_homo_clean_features/" + data_name + '_S*.out')
    #files = glob.glob("" + data_name + '_S*.out')
    files = glob.glob("../Split_train_nodes_results/" + data_name + '_S*.out')
    results = []

    for file in files:
        with open(file, 'r') as f:
            last_line = f.readlines()[-1]
            match = re.search(pattern, last_line)
            print("check if find the file", file, last_line)
            if match:
                
                score = float(match.group(1))
                print(f"File: {file}, Score: {score}")
                print("\n")
                results.append(score)
    print(results)

    # Calculate average and standard deviation
    average = round(statistics.mean(results), 4)
    std_dev = round(statistics.stdev(results), 4)

    # Write the results into a file
    #with open("vary_homo_clean_features/" + data_name + "_report.txt", "w") as file:
    with open("../Split_train_nodes_results/" + data_name + "_report.txt", "w") as file:
    #with open("" + data_name + "_report.txt", "w") as file:
        file.write(f"Results on splits: {results}\n")
        file.write(f"Average: {average}\n")
        file.write(f"Standard Deviation: {std_dev}\n")

