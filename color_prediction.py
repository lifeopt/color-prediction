import data_analysis
data_location = "./data_project"
five_raw_data_names = ["log_spectrodetail.csv", "log_spectrok_pkpaper.csv", "mst_ink.csv", "mst_paper.csv", "udt_imxbom.csv"]
# data_analysis.data_integration(data_location, "new_spectrodetail.csv", "new_spectrok_inkmixsample.csv", "new_udt_inkmixbarcode.csv")
data_analysis.data_integration(data_location, five_raw_data_names)
data_analysis.data_analysis(data_location, "20210302_add_paper.csv", "analysis_results.csv")