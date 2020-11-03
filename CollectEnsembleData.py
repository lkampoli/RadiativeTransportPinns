from CollectUtils import *

np.random.seed(42)

base_path_list = ["EnsRadFreq3"]

for base_path in base_path_list:
    print("#################################################")
    print(base_path)

    b = False
    compute_std = False
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sensitivity_df = pd.DataFrame(columns=["batch_size",
                                           "regularization_parameter",
                                           "kernel_regularizer",
                                           "neurons",
                                           "hidden_layers",
                                           "residual_parameter",
                                           "L2_norm_test",
                                           "error_train",
                                           "error_val",
                                           "error_test"])
    # print(sensitivity_df)
    selection_criterion = "error_train"

    Nu_list = []
    Nf_list = []
    t_0 = 0
    t_f = 1
    x_0 = -1
    x_f = 1

    L2_norm = []
    criterion = []
    best_retrain_list = []
    list_models_setup = list()

    for subdirec in directories_model:
        model_path = base_path

        sample_path = model_path + "/" + subdirec
        retrainings_fold = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]

        retr_to_check_file = None
        for ret in retrainings_fold:
            if os.path.isfile(sample_path + "/" + ret + "/TrainedModel/Information.csv"):
                retr_to_check_file = ret
                break

        setup_num = int(subdirec.split("_")[1])
        if retr_to_check_file is not None:
            info_model = pd.read_csv(sample_path + "/" + retr_to_check_file + "/TrainedModel/Information.csv", header=0, sep=",")
            best_retrain = select_over_retrainings(sample_path, selection=selection_criterion, mode="min", compute_std=compute_std, compute_val=False, rs_val=0)
            info_model["error_train"] = best_retrain["error_train"]
            info_model["train_time"] = best_retrain["train_time"]
            info_model["error_val"] = 0
            info_model["error_test"] = 0
            info_model["L2_norm_test"] = best_retrain["L2_norm_test"]
            info_model["rel_L2_norm"] = best_retrain["rel_L2_norm"]
            if os.path.isfile(sample_path + "/" + retr_to_check_file + "/Images/errors.txt"):
                # info_model["u_rel_err"] = best_retrain["u_rel_err"]
                # info_model["k_rel_err"] = best_retrain["k_rel_err"]
                info_model["err_1"] = best_retrain["err_1"]
                info_model["err_2"] = best_retrain["err_2"]
            if os.path.isfile(sample_path + "/" + retr_to_check_file + "/Images/errors_inv.txt"):
                info_model["l2_glob"] = best_retrain["l2_glob"]
                info_model["l2_glob_rel"] = best_retrain["l2_glob_rel"]
                info_model["l2_om_big"] = best_retrain["l2_om_big"]
                info_model["l2_om_big_rel"] = best_retrain["l2_om_big_rel"]
                info_model["h1_glob"] = best_retrain["h1_glob"]
                info_model["h1_glob_rel"] = best_retrain["h1_glob_rel"]
            info_model["setup"] = setup_num
            info_model["retraining"] = best_retrain["retraining"]

            if info_model["batch_size"].values[0] == "full":
                info_model["batch_size"] = best_retrain["Nu_train"] + best_retrain["Nf_train"]
            sensitivity_df = sensitivity_df.append(info_model, ignore_index=True)
        else:
            print(sample_path + "/TrainedModel/Information.csv not found")

    sensitivity_df = sensitivity_df.sort_values(selection_criterion)
    sensitivity_df = sensitivity_df.rename(columns={'L2_norm_test': 'L2'})
    best_setup = sensitivity_df.iloc[0]
    best_setup.to_csv(base_path + "best.csv", header=0, index=False)
    # print(sensitivity_df)
    print("Best Setup:", best_setup["setup"])
    print(best_setup)

    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.scatter(sensitivity_df[selection_criterion], sensitivity_df["L2"])
    plt.xlabel(r'$\varepsilon_T$')
    plt.ylabel(r'$\varepsilon_G$')
    plt.savefig(base_path + "/et_vs_eg_" + base_path + ".png", dpi=400)
