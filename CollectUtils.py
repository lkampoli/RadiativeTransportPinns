from ImportFile import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def select_over_retrainings(folder_path, selection="error_train", mode="min", compute_std=False, compute_val=False, rs_val=0):
    retrain_models = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    models_list = list()
    for retraining in retrain_models:
        rs = int(folder_path.split("_")[-1])
        retrain_path = folder_path + "/" + retraining
        number_of_ret = retraining.split("_")[-1]

        if os.path.isfile(retrain_path + "/InfoModel.txt"):
            models = pd.read_csv(retrain_path + "/InfoModel.txt", header=0, sep=",")
            models["retraining"] = number_of_ret
            if os.path.isfile(retrain_path + "/Images/errors.txt"):
                errors = pd.read_csv(retrain_path + "/Images/errors.txt", header=0, sep=",")
                # models["u_rel_err"] = errors["u_rel_err"]
                # models["k_rel_err"] = errors["k_rel_err"]
                # models["g_rel_err"] = errors["g_rel_err"]
                models["err_1"] = errors["err_1"]
                models["err_2"] = errors["err_2"]
            if os.path.isfile(retrain_path + "/Images/errors_inv.txt"):
                errors = pd.read_csv(retrain_path + "/Images/errors_inv.txt", header=0, sep=",")
                models["l2_glob"] = errors["l2_glob"]
                models["l2_glob_rel"] = errors["l2_glob_rel"]
                models["l2_om_big"] = errors["l2_om_big"]
                models["l2_om_big_rel"] = errors["l2_om_big_rel"]
                models["h1_glob"] = errors["h1_glob"]
                models["h1_glob_rel"] = errors["h1_glob_rel"]
                if 'l2_p_rel' in errors.columns:
                    models["l2_p_rel"] = errors["l2_p_rel"]
                if 'h1_p_rel' in errors.columns:
                    models["h1_p_rel"] = errors["h1_p_rel"]

            if os.path.isfile(retrain_path + "/TrainedModel/model.pkl"):
                trained_model = torch.load(retrain_path + "/TrainedModel/model.pkl", map_location=torch.device('cpu'))
                trained_model.eval()
            models_list.append(models)
        else:
            print("No File Found")

    retraining_prop = pd.concat(models_list, ignore_index=True)
    retraining_prop = retraining_prop.sort_values(selection)
    if mode == "min":
        return retraining_prop.iloc[0]
    else:
        retraining = retraining_prop["retraining"].iloc[0]
        retraining_prop = retraining_prop.mean()
        retraining_prop["retraining"] = retraining
        return retraining_prop

