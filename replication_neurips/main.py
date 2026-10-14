# =====================================================
#                DATA GENERATION PHASE
# =====================================================

CHOICES = [
    "detection",
    "detection_cc",
    "detection_cc_strong",
    "detection_cc_mid",
    "detection_cc_2",
    "detection_sachs",
    "detection_hyper",
    "rca_hyper",
    "causal_discovery",
    "classification",
    "classification_regression_compare",
    "classification_priors_small_default",
    "classification_priors_small_single",
    "classification_priors_small_multi",
    "classification_cc",
    "classification_cc_2",
    "classification_cc_strong",
    "classification_cc_mid",
    "classification_hyper",
    "classification_rc_count_sensitivity",
    "classification_correlated_measurement",
    "classification_dag_misspecification",
    "classification_heteroscedasticity",
    "classification_soft_interventions",
    "classification_confounding",
    "rca",
    "rca_regression_compare",
    "rca_priors_small_default",
    "rca_priors_small_single",
    "rca_priors_small_multi",
    "rca_only_mech",
    "rca_only_meas",
    "rca_cc",
    "rca_cc_strong",
    "rca_cc_mid",
    "rca_cc_2",
    "rca_cc_2000",
    "rca_cc_only_mech",
    "rca_cc_only_meas",
    "rca_cc_only_mech_2",
    "rca_cc_only_meas_2",
    "rca_sample_sensitivity",
    "rca_feature_sensitivity",
    "rca_edge_prob_sensitivity",
    "rca_rc_count_sensitivity",
    "rca_correlated_measurement",
    "rca_dag_misspecification",
    "rca_heteroscedasticity",
    "rca_soft_interventions",
    "rca_confounding",
    "sachs",
    "sample_classification",
]

def generate_data(experiment: str):
    
    
    
    config = {
        "N_RUNS":20,
        "N": 2000,
        "D": 15,
        "EDGE_DENSITY": 0.3,
        "SAMPLE_RATIO": 0.1,
        "MEAS_RATIO": 0,
        "GEN_RATIO": 0,
        "CORRUPT_RATIO": 0,
        "OUTLIER_SETTING": "MEAN_SHIFT",
        "MEAN_SHIFT": 3,
        "OUTLIER_TYPE":"both",
        "DAG_MISSPECIFICATION_TYPE": None,
        "DAG_MISSPECIFICATION_RATIO": 0.0,
        "SOFT_INTERVENTION_TYPE": None,
        "HIDDEN_VAR_RATIO": 0.0,
        "HIDDEN_VAR_COUNT": 0,
        "CORRELATED_MEASUREMENT_GROUP_SIZE": 1,
        "HETEROSCEDASTICITY_STRENGTH": 0.0,
    }


    N_RUNS = config["N_RUNS"]
    initial_seed = CHOICES.index(experiment) * N_RUNS

    if experiment in ["detection_hyper","detection"]:
        
        for mean_shift in [0, 1, 2, 3, 4, 5]:
            
            config["MEAN_SHIFT"] = mean_shift
            
            directory_path =  f"src/datasets/{experiment}_strength_{mean_shift}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):

                generate_experiment_data(
                    f"src/datasets/{experiment}_strength_{mean_shift}",
                    experiment, N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config
                )
            else:
                print("DATA ALREADY GENERATED")

    elif experiment in ["rca_hyper","rca","rca_only_mech","rca_only_meas","classification","rca_realisitc","causal_discovery","classification_hyper","sample_classification"]:
        
    
        for mean_shift in [6,7,8]:

            directory_path =  f"src/datasets/{experiment}_strength_{mean_shift}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                config["MEAN_SHIFT"] = mean_shift
                generate_experiment_data(
                    f"src/datasets/{experiment}_strength_{mean_shift}",
                    experiment, N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config
            )

    elif experiment == "classification_regression_compare":

        config.update({
            "N_RUNS": 3,
            "N": 1000,
            "D": 5,
            "EDGE_DENSITY": 0.35,
            "SAMPLE_RATIO": 0.1,
            "MEAN_SHIFT": 6,
            "OUTLIER_TYPE": "both",
        })
        directory_path = f"src/datasets/{experiment}_strength_{config['MEAN_SHIFT']}"

        if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
            generate_experiment_data(
                directory_path,
                "classification",
                N_RUNS=config["N_RUNS"],
                seeds=[initial_seed + i for i in range(config["N_RUNS"])],
                config=config,
            )

    elif experiment == "rca_regression_compare":

        config.update({
            "N_RUNS": 3,
            "N": 1000,
            "D": 5,
            "EDGE_DENSITY": 0.35,
            "SAMPLE_RATIO": 0.1,
            "MEAN_SHIFT": 6,
            "OUTLIER_TYPE": "both",
        })
        directory_path = f"src/datasets/{experiment}_strength_{config['MEAN_SHIFT']}"

        if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
            generate_experiment_data(
                directory_path,
                experiment,
                N_RUNS=config["N_RUNS"],
                seeds=[initial_seed + i for i in range(config["N_RUNS"])],
                config=config,
            )

    elif experiment == "rca_correlated_measurement":

        config["OUTLIER_TYPE"] = "both"
        config["CORRELATED_MEASUREMENT_GROUP_SIZE"] = 2
        for mean_shift in [3, 4, 5]:
            config["MEAN_SHIFT"] = mean_shift
            directory_path = f"src/datasets/{experiment}_strength_{mean_shift}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                generate_experiment_data(
                    directory_path,
                    experiment,
                    N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config
                )

    elif experiment == "rca_heteroscedasticity":

        config["MEAN_SHIFT"] = 3
        for heteroscedasticity_strength in [0.0,0.25, 0.5,0.75, 1.0]:
            config["HETEROSCEDASTICITY_STRENGTH"] = heteroscedasticity_strength
            directory_path = f"src/datasets/{experiment}_hetero_{heteroscedasticity_strength}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                generate_experiment_data(
                    directory_path,
                    experiment,
                    N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config
                )

    elif experiment == "rca_dag_misspecification":

        for corruption_type in ["add", "remove", "reverse"]:
            for corruption_ratio in [0.1, 0.3, 0.5]:
                config["DAG_MISSPECIFICATION_TYPE"] = corruption_type
                config["DAG_MISSPECIFICATION_RATIO"] = corruption_ratio
                directory_path = f"src/datasets/{experiment}_{corruption_type}_{corruption_ratio}"

                if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                    generate_experiment_data(
                        directory_path,
                        experiment,
                        N_RUNS=N_RUNS,
                        seeds=[initial_seed + i for i in range(N_RUNS)],
                        config=config
                    )

    elif experiment == "rca_soft_interventions":

        config["OUTLIER_TYPE"] = "both"
        for soft_intervention_type in ["soft_noise", "mechanism"]:
            for mean_shift in [3, 4, 5]:
                config["SOFT_INTERVENTION_TYPE"] = soft_intervention_type
                config["MEAN_SHIFT"] = mean_shift
                directory_path = f"src/datasets/{experiment}_{soft_intervention_type}_strength_{mean_shift}"

                if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                    generate_experiment_data(
                        directory_path,
                        experiment,
                        N_RUNS=N_RUNS,
                        seeds=[initial_seed + i for i in range(N_RUNS)],
                        config=config
                    )

    elif experiment == "classification_correlated_measurement":

        config["OUTLIER_TYPE"] = "both"
        config["CORRELATED_MEASUREMENT_GROUP_SIZE"] = 2
        for mean_shift in [3, 4, 5]:
            config["MEAN_SHIFT"] = mean_shift
            directory_path = f"src/datasets/{experiment}_strength_{mean_shift}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                generate_experiment_data(
                    directory_path,
                    experiment,
                    N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config
                )

    elif experiment == "classification_heteroscedasticity":

        config["OUTLIER_TYPE"] = "both"
        config["MEAN_SHIFT"] = 3
        for heteroscedasticity_strength in [0.0,0.25, 0.5,0.75, 1.0]:
            config["HETEROSCEDASTICITY_STRENGTH"] = heteroscedasticity_strength
            directory_path = f"src/datasets/{experiment}_hetero_{heteroscedasticity_strength}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                generate_experiment_data(
                    directory_path,
                    experiment,
                    N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config
                )

    elif experiment == "classification_dag_misspecification":

        for corruption_type in ["add", "remove", "reverse"]:
            for corruption_ratio in [0.1, 0.3, 0.5]:
                config["DAG_MISSPECIFICATION_TYPE"] = corruption_type
                config["DAG_MISSPECIFICATION_RATIO"] = corruption_ratio
                directory_path = f"src/datasets/{experiment}_{corruption_type}_{corruption_ratio}"

                if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                    generate_experiment_data(
                        directory_path,
                        experiment,
                        N_RUNS=N_RUNS,
                        seeds=[initial_seed + i for i in range(N_RUNS)],
                        config=config
                    )

    elif experiment == "classification_soft_interventions":

        config["OUTLIER_TYPE"] = "both"
        for soft_intervention_type in ["soft_noise", "mechanism"]:
            for mean_shift in [3, 4, 5]:
                config["SOFT_INTERVENTION_TYPE"] = soft_intervention_type
                config["MEAN_SHIFT"] = mean_shift
                directory_path = f"src/datasets/{experiment}_{soft_intervention_type}_strength_{mean_shift}"

                if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                    generate_experiment_data(
                        directory_path,
                        experiment,
                        N_RUNS=N_RUNS,
                        seeds=[initial_seed + i for i in range(N_RUNS)],
                        config=config
                    )

    elif experiment in ["rca_confounding", "classification_confounding"]:

        target_observed_d = config["D"]
        for hidden_var_ratio in [0.1,0.25, 0.5]:
            hidden_var_count = max(1, int(round(target_observed_d * hidden_var_ratio / (1 - hidden_var_ratio))))
            config["D"] = target_observed_d + hidden_var_count
            config["HIDDEN_VAR_RATIO"] = hidden_var_ratio
            config["HIDDEN_VAR_COUNT"] = hidden_var_count
            directory_path = f"src/datasets/{experiment}_hidden_{hidden_var_ratio}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                generate_experiment_data(
                    directory_path,
                    experiment,
                    N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config
                )
   

    elif experiment in ["sample_real", "rca_cc","classification_cc",  "classification_cc_2","rca_real_realistic", "rca_cc_only_meas", "rca_cc_only_mech","detection_cc","rca_cc_2","rca_cc_only_mech_2","rca_cc_only_meas_2","detection_cc_2"]:
        
        directory_path =  f"src/datasets/{experiment}"

        if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
            
            outlier_type = "both"
            if experiment in["rca_cc_only_meas","rca_cc_only_meas_2"]:
               outlier_type = "meas"
            elif experiment in ["rca_cc_only_mech","rca_cc_only_mech_2"]:
                outlier_type = "mech"

            process_real_data(
                DOWNLOAD_DIR=f"src/datasets/{experiment}",
                DATA_DIR=f"src/datasets/{experiment}",
                SAMPLE_ANOM_RATIO=0.05,
                N_RUNS=N_RUNS, MEAN_SHIFT=3,dataset_name=experiment, seed=initial_seed,outlier_type=outlier_type
            )
    
    elif experiment in ["rca_cc_mid","classification_cc_mid","detection_cc_mid"]:
        
        directory_path =  f"src/datasets/{experiment}/{experiment}_5.pkl"

        if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
            
            outlier_type = "both"
            # if experiment in["rca_cc_only_meas","rca_cc_only_meas_2"]:
            #    outlier_type = "meas"
            # elif experiment in ["rca_cc_only_mech","rca_cc_only_mech_2"]:
            #     outlier_type = "mech"

            process_real_data(
                DOWNLOAD_DIR=f"src/datasets/{experiment}",
                DATA_DIR=f"src/datasets/{experiment}",
                SAMPLE_ANOM_RATIO=0.05,
                N_RUNS=N_RUNS, MEAN_SHIFT=3,dataset_name=experiment, seed=initial_seed,outlier_type=outlier_type,strength="mid"
            )

    elif experiment in ["rca_cc_strong","classification_cc_strong","detection_cc_strong"]:
        
        directory_path =  f"src/datasets/{experiment}/{experiment}_5.pkl"

        if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
            
            outlier_type = "both"
            # if experiment in["rca_cc_only_meas","rca_cc_only_meas_2"]:
            #    outlier_type = "meas"
            # elif experiment in ["rca_cc_only_mech","rca_cc_only_mech_2"]:
            #     outlier_type = "mech"

            process_real_data(
                DOWNLOAD_DIR=f"src/datasets/{experiment}",
                DATA_DIR=f"src/datasets/{experiment}",
                SAMPLE_ANOM_RATIO=0.05,
                N_RUNS=N_RUNS, MEAN_SHIFT=3,dataset_name=experiment, seed=initial_seed,outlier_type=outlier_type,strength="strong",
            )


    elif experiment in ["classification_rc_count_sensitivity","rca_rc_count_sensitivity"]:
      

        for rc_count in [1,3,5,8]:

            directory_path = f"src/datasets/{experiment}_rc_count_{rc_count}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                
                config["RC_COUNT"] = rc_count

                generate_experiment_data(
                    directory_path,
                    experiment, N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config,specific="fixed_rc_count"
                )

    elif experiment in [
        "rca_priors_small_single",
        "rca_priors_small_multi",
        "classification_priors_small_single",
        "classification_priors_small_multi",
    ]:
        config["N"] = 500
        config["D"] = 3
        config["RC_COUNT"] = 1 if "single" in experiment else 2

        for mean_shift in [3, 4, 5]:
            config["MEAN_SHIFT"] = mean_shift
            directory_path = f"src/datasets/{experiment}_strength_{mean_shift}"
            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                generate_experiment_data(
                    directory_path,
                    experiment,
                    N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config,
                    specific="fixed_rc_count",
                )

    elif experiment in [
        "rca_priors_small_default",
        "classification_priors_small_default",
    ]:
        config["N"] = 500
        config["D"] = 3
        shared_seed_base = 9000

        for mean_shift in [3, 4, 5]:
            config["MEAN_SHIFT"] = mean_shift
            directory_path = f"src/datasets/{experiment}_strength_{mean_shift}"
            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                generate_experiment_data(
                    directory_path,
                    experiment,
                    N_RUNS=N_RUNS,
                    seeds=[shared_seed_base + i for i in range(N_RUNS)],
                    config=config,
                )

    elif experiment == "rca_sample_sensitivity":

        for sample_size in [500,2000,8000]:
            config["N"] = sample_size

            directory_path = f"src/datasets/{experiment}_N_{sample_size}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                


                generate_experiment_data(
                    directory_path,
                    experiment, N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config
                )


    elif experiment == "rca_feature_sensitivity":

        for feature_size in [5,15,30]:

            config["D"] = feature_size

            directory_path = f"src/datasets/{experiment}_D_{feature_size}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                


                generate_experiment_data(
                    directory_path,
                    experiment, N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config
                )

    elif experiment == "rca_edge_prob_sensitivity":

        for p_e in [0.1,0.3,0.6]:

            config["EDGE_DENSITY"] = p_e

            directory_path = f"src/datasets/{experiment}_edge_prob_{p_e}"

            if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
                


                generate_experiment_data(
                    directory_path,
                    experiment, N_RUNS=N_RUNS,
                    seeds=[initial_seed + i for i in range(N_RUNS)],
                    config=config
                )


    elif experiment == "sachs":
        N_RUNS = 20
        directory_path = "src/datasets/sachs/0.pkl"

        if not (os.path.exists(directory_path)):
            process_sachs_data(
                data_dir="src/datasets/sachs",
                seed=initial_seed ,
                N_RUNS=N_RUNS
            )


            
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")


# =====================================================
#                EXPERIMENT EXECUTION PHASE
# =====================================================
# import os
# import argparse
# from functools import partial

# # limit threading to avoid overuse of CPU
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"







def run_experiment(experiment: str):
    """
    Execute a chosen experiment type (oracle or realistic).
    """
    
    if experiment == "detection":
        from src.experiments.evaluate_competitors import (
            run_sample_ad,
            evaluate_sample_ad
        )
        
        methods = {
            "CALI_TRUE_NEW":cali,
            "CALI_LEARN_NEW":cali,
            "AE":ae_wrapper,
            "NF":nf_wrapper,
            "LOF":lof_wrapper,
            "IFOREST": iforest_wrapper,
            "DEEPSVDD":deepsvdd_wrapper,
            "COPOD":copod_wrapper
        }
        
        
        for mean_shift in [0, 1, 2, 3, 4, 5]:
            
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
            
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"

            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"


            run_sample_ad(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_sample_ad(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)









    # elif experiment == "rca_realistic":
        
    #     from src.experiments.evaluate_competitors_realistic import (
    #         run_feature_ad,
    #         evaluate_feature_ad
    #     )

    #     methods = {
    #         "CALI_TRUE":cali,
    #         "CALI_LEARN":cali,
    #         "AE-SHAP":partial(shap_feature_explainer,realistic=True,model_type="AE"),
    #         "M2OE":run_m2oe_method_realistic,
    #         "ROOTCLAM":run_vaca_realistic,
    #         "DICE":run_dice_realistic,
    #         # "QGAM_SCM_SHAP":shap_qgam_scm
    #     }

    #     for mean_shift in [3, 4, 5]:
            
            
    #         DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
            
    #         RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"

    #         PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"

    #         run_feature_ad(methods,DATA_DIR,experiment,RESULT_DIR)
    #         evaluate_feature_ad(RESULT_DIR,PLOT_DIR)
    #         plot_all(RESULT_DIR,PLOT_DIR)





    elif experiment in [
        "rca_priors_small_default",
        "rca_priors_small_single",
        "rca_priors_small_multi",
    ]:
        
        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE": {"method": cali, "params": {}},
            "CALI_PRIOR_EXHAUSTIVE": {
                "method": cali_with_priors_exhaustive_oracle,
                "params": {"pass_labels": True},
            },
            "CALI_PRIOR_SINGLE_ROOT": {
                "method": cali_with_priors_single_root_oracle,
                "params": {"pass_labels": True},
            },
        }

        for mean_shift in [3, 4, 5]:
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"
            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"

            run_rca(methods, DATA_DIR, experiment, RESULT_DIR)
            evaluate_rca(RESULT_DIR, PLOT_DIR)
            plot_all(RESULT_DIR, PLOT_DIR)

    elif experiment in ["rca","rca_only_mech","rca_only_meas"]:
        
        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
            # "SCORE_ORDER_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            # "SMOOTH_TRAVERSALREALISTIC":partial(full_realistic_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            # "AE_RECREALISTIC":partial(full_realistic_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            # "AE_SHAP_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            # "SCM_RESID_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
        }

        # if experiment == "rca":
        #     methods[ "M2OE"] = partial(full_rca_run,rca_explainer=M2OE,ad_model=AutoEncoder)
        #     methods["DICE"] = partial(full_rca_run,rca_explainer=DICE,ad_model=AutoEncoder)

        
            # methods[ "M2OE_REALISTIC"] = partial(full_realistic_rca_run,rca_explainer=M2OE,ad_model=AutoEncoder),
            # methods["DICE_REALISTIC"] = partial(full_realistic_rca_run,rca_explainer=DICE,ad_model=AutoEncoder)

        for mean_shift in [6,7,8]:
            
            
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
            
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"

            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"

            run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_rca(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "rca_regression_compare":

        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        base_settings = {
            "mode": "default",
            "scale": False,
            "loss": "L1",
            "discrete": False,
            "qu": 0.5,
        }

        methods = {
            "CALI_QGAM": {
                "method": cali,
                "params": {
                    "regression_alg": "QGAM",
                    "regression_settings": {
                        **base_settings,
                        "backend": "r_qgam_strict",
                    },
                },
            },
            "CALI_SKLEARN_QUANTILE_SPLINE": {
                "method": cali,
                "params": {
                    "regression_alg": "SKLEARN_QUANTILE_SPLINE",
                    "regression_settings": base_settings,
                },
            },
            "CALI_PYGAM_EXPECTILE": {
                "method": cali,
                "params": {
                    "regression_alg": "PYGAM_EXPECTILE",
                    "regression_settings": base_settings,
                },
            },
        }

        mean_shift = 6
        DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
        RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"
        PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"

        run_rca(methods, DATA_DIR, experiment, RESULT_DIR, max_workers=1)
        evaluate_rca(RESULT_DIR, PLOT_DIR)
        plot_all(RESULT_DIR, PLOT_DIR)

    elif experiment == "rca_correlated_measurement":

        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
        }

        for mean_shift in [3, 4, 5]:
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"
            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"

            run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_rca(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "rca_heteroscedasticity":

        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
        }

        for heteroscedasticity_strength in [0.0,0.25, 0.5,0.75, 1.0]:
            DATA_DIR = f"src/datasets/{experiment}_hetero_{heteroscedasticity_strength}"
            RESULT_DIR = f"src/results/{experiment}_hetero_{heteroscedasticity_strength}"
            PLOT_DIR = f"src/plots/{experiment}_hetero_{heteroscedasticity_strength}"

            run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_rca(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "rca_dag_misspecification":

        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
        }

        for corruption_type in ["add", "remove", "reverse"]:
            for corruption_ratio in [0.1, 0.3, 0.5]:
                DATA_DIR = f"src/datasets/{experiment}_{corruption_type}_{corruption_ratio}"
                RESULT_DIR = f"src/results/{experiment}_{corruption_type}_{corruption_ratio}"
                PLOT_DIR = f"src/plots/{experiment}_{corruption_type}_{corruption_ratio}"

                run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
                evaluate_rca(RESULT_DIR,PLOT_DIR)
                plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "rca_confounding":

        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
        }

        for hidden_var_ratio in [0.1, 0.25, 0.5]:
            DATA_DIR = f"src/datasets/{experiment}_hidden_{hidden_var_ratio}"
            RESULT_DIR = f"src/results/{experiment}_hidden_{hidden_var_ratio}"
            PLOT_DIR = f"src/plots/{experiment}_hidden_{hidden_var_ratio}"

            run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_rca(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "rca_soft_interventions":

        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
        }

        for soft_intervention_type in ["soft_noise", "mechanism"]:
            for mean_shift in [3, 4, 5]:
                DATA_DIR = f"src/datasets/{experiment}_{soft_intervention_type}_strength_{mean_shift}"
                RESULT_DIR = f"src/results/{experiment}_{soft_intervention_type}_strength_{mean_shift}"
                PLOT_DIR = f"src/plots/{experiment}_{soft_intervention_type}_strength_{mean_shift}"

                run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
                evaluate_rca(RESULT_DIR,PLOT_DIR)
                plot_all(RESULT_DIR,PLOT_DIR)


    elif experiment == "rca_sample_sensitivity":

        
        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
            # "M2OE":partial(full_rca_run,rca_explainer=M2OE,ad_model=AutoEncoder)

        }

        for N in [500, 2000, 8000]:
            
            
            DATA_DIR = f"src/datasets/{experiment}_N_{N}"
            
            RESULT_DIR = f"src/results/{experiment}_N_{N}"

            PLOT_DIR = f"src/plots/{experiment}_N_{N}"

            run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_rca(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)


    elif experiment == "rca_feature_sensitivity":

        
        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE_NEW":cali,
            "CALI_LEARN_NEW":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
            # "M2OE":partial(full_rca_run,rca_explainer=M2OE,ad_model=AutoEncoder)

        }

        for D in [5,15, 30]:
            
            
            DATA_DIR = f"src/datasets/{experiment}_D_{D}"
            
            RESULT_DIR = f"src/results/{experiment}_D_{D}"

            PLOT_DIR = f"src/plots/{experiment}_D_{D}"

            run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_rca(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)


    elif experiment == "rca_edge_prob_sensitivity":

        
        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE_NEW":cali,
            "CALI_LEARN_NEW":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
            # "M2OE":partial(full_rca_run,rca_explainer=M2OE,ad_model=AutoEncoder)

        }

        for p_e in [0.1, 0.3, 0.6]:
            
            
            DATA_DIR = f"src/datasets/{experiment}_edge_prob_{p_e}"
            
            RESULT_DIR = f"src/results/{experiment}_edge_prob_{p_e}"

            PLOT_DIR = f"src/plots/{experiment}_edge_prob_{p_e}"

            run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_rca(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    
    elif experiment == "rca_rc_count_sensitivity":

        
        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

        methods = {
            "CALI_TRUE_NEW":cali,
            "CALI_LEARN_NEW":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=AutoEncoder),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=AutoEncoder),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=AutoEncoder,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=AutoEncoder),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=AutoEncoder),
            # "M2OE":partial(full_rca_run,rca_explainer=M2OE,ad_model=AutoEncoder)

        }

        for rc_count in [1, 3, 5,8]:
            
            
            DATA_DIR = f"src/datasets/{experiment}_rc_count_{rc_count}"
            
            RESULT_DIR = f"src/results/{experiment}_rc_count_{rc_count}"

            PLOT_DIR = f"src/plots/{experiment}_rc_count_{rc_count}"

            run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_rca(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)




    # elif experiment == "sample_real":
        
    #     from src.experiments.evaluate_competitors_real_data import (
    #         run_sample_ad,
    #         evaluate_sample_ad
    #     )


    #     methods = {
    #         "CALI_TRUE":cali,
    #         "CALI_LEARN":cali,
    #         "AE":ae_wrapper,
    #         "NF":nf_wrapper,
    #         "LOF":lof_wrapper,
    #         "IFOREST": iforest_wrapper,
    #         "DEEPSVDD":deepsvdd_wrapper,
    #         "COPOD":copod_wrapper
    #     }

    #     DATA_DIR = f"src/datasets/{experiment}"
    #     RESULT_DIR = f"src/results/{experiment}"
    #     PLOT_DIR = f"src/plots/{experiment}"

    #     run_sample_ad(methods,DATA_DIR,experiment,RESULT_DIR)
    #     evaluate_sample_ad(RESULT_DIR,PLOT_DIR)
    #     plot_all(RESULT_DIR,PLOT_DIR)
        

    # elif experiment == "rca_realistic_real":
        
    #     from src.experiments.evaluate_competitors_real_data import (
    #         run_feature_ad_realistic,
    #         evaluate_feature_ad_realistic
    #     )
        
    #     methods = {
    #         "CALI_TRUE":cali,
    #         "CALI_LEARN":cali,
    #         "AE-SHAP":partial(shap_feature_explainer,realistic=True,model_type="AE"),
    #         "NF-SHAP":partial(shap_feature_explainer,realistic=True,model_type="NF"),
    #         # "M2OE":run_m2oe_method_realistic,
    #         "ROOTCLAM":run_vaca_realistic,
    #         "DICE":run_dice_realistic,
    #         "QGAM_SCM_SHAP":shap_qgam_scm
    #     }

    #     DATA_DIR = f"src/datasets/{experiment}"
    #     RESULT_DIR = f"src/results/{experiment}"
    #     PLOT_DIR = f"src/plots/{experiment}"

    #     run_feature_ad_realistic(methods,DATA_DIR,experiment,RESULT_DIR)
    #     evaluate_feature_ad_realistic(RESULT_DIR,PLOT_DIR)
    #     plot_all(RESULT_DIR,PLOT_DIR)

    # elif experiment == "rca_cc":
    #     from src.experiments.evaluate_rca_cc import (
    #         run_rca,
    #         evaluate_rca 
    #     )

    #     methods = {
    #         "CALI_TRUE":cali,
    #         "CALI_LEARN_NEW_ALPHA":cali,
    #         "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=NF_WRAPPER),
    #         "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=NF_WRAPPER),
    #         "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=NF_WRAPPER,rca_params={"epochs":10,"batch_size":64}),
    #         "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=NF_WRAPPER),
    #         "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=NF_WRAPPER),
    #         "SCORE_ORDER_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SCORE_ORDER,ad_model=NF_WRAPPER),
    #         "SMOOTH_TRAVERSAL_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=NF_WRAPPER),
    #         "AE_REC_REALISTIC":partial(full_realistic_rca_run,rca_explainer=AE_REC,ad_model=NF_WRAPPER,rca_params={"epochs":10,"batch_size":64}),
    #         "AE_SHAP_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SHAPExplainer,ad_model=NF_WRAPPER),
    #         "SCM_RESID_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=NF_WRAPPER),
    #         # "M2OE":partial(full_rca_run,rca_explainer=M2OE,ad_model=NF_WRAPPER)
    #     }


    #     DATA_DIR = f"src/datasets/{experiment}"
    #     RESULT_DIR = f"src/results/{experiment}"
    #     PLOT_DIR = f"src/plots/{experiment}"

    #     run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
    #     evaluate_rca(RESULT_DIR,experiment,PLOT_DIR)
    #     plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "rca_cc_2":
        from src.experiments.evaluate_rca_cc import (
            run_rca,
            evaluate_rca 
        )

        methods = {
            "CALI_TRUE_NEW":cali,
            "CALI_LEARN_NEW":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=NF_WRAPPER),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=NF_WRAPPER),
            # "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=NF_WRAPPER,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=NF_WRAPPER),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=NF_WRAPPER),
            # "SCORE_ORDER_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SCORE_ORDER,ad_model=NF_WRAPPER),
            # "SMOOTH_TRAVERSAL_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=NF_WRAPPER),
            # "AE_REC_REALISTIC":partial(full_realistic_rca_run,rca_explainer=AE_REC,ad_model=NF_WRAPPER,rca_params={"epochs":10,"batch_size":64}),
            # "AE_SHAP_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SHAPExplainer,ad_model=NF_WRAPPER),
            # "SCM_RESID_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=NF_WRAPPER),
            # "M2OE":partial(full_rca_run,rca_explainer=M2OE,ad_model=NF_WRAPPER)
        }


        DATA_DIR = f"src/datasets/{experiment}"
        RESULT_DIR = f"src/results/{experiment}"
        PLOT_DIR = f"src/plots/{experiment}"

        # run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
        evaluate_rca(RESULT_DIR,experiment,PLOT_DIR)
        plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment in ["rca_cc_strong","rca_cc_mid"]:
        from src.experiments.evaluate_rca_cc import (
            run_rca,
            evaluate_rca 
        )

        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=NF_WRAPPER),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=NF_WRAPPER),
            # "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=NF_WRAPPER,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=NF_WRAPPER),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=NF_WRAPPER),
            "SCORE_ORDER_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SCORE_ORDER,ad_model=NF_WRAPPER),
            "SMOOTH_TRAVERSAL_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=NF_WRAPPER),
            # "AE_REC_REALISTIC":partial(full_realistic_rca_run,rca_explainer=AE_REC,ad_model=NF_WRAPPER,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SHAPExplainer,ad_model=NF_WRAPPER),
            "SCM_RESID_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=NF_WRAPPER),
            # "M2OE":partial(full_rca_run,rca_explainer=M2OE,ad_model=NF_WRAPPER)
        }


        DATA_DIR = f"src/datasets/{experiment}"
        RESULT_DIR = f"src/results/{experiment}"
        PLOT_DIR = f"src/plots/{experiment}"

        run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
        evaluate_rca(RESULT_DIR,experiment,PLOT_DIR)
        plot_all(RESULT_DIR,PLOT_DIR)
    # elif experiment == "rca_cc_2000":
    #     from src.experiments.evaluate_rca_cc import (
    #         run_rca,
    #         evaluate_rca 
    #     )

    #     methods = {
    #         "CALI_TRUE":cali,
    #         # "CALI_LEARN":cali,
    #         # "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=NF_WRAPPER),
    #         # "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=NF_WRAPPER),
    #         # # "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=NF_WRAPPER,rca_params={"epochs":10,"batch_size":64}),
    #         # "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=NF_WRAPPER),
    #         # "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=NF_WRAPPER),
    #         # "SCORE_ORDER_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SCORE_ORDER,ad_model=NF_WRAPPER),
    #         # "SMOOTH_TRAVERSAL_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=NF_WRAPPER),
    #         # "AE_REC_REALISTIC":partial(full_realistic_rca_run,rca_explainer=AE_REC,ad_model=NF_WRAPPER,rca_params={"epochs":10,"batch_size":64}),
    #         # "AE_SHAP_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SHAPExplainer,ad_model=NF_WRAPPER),
    #         # "SCM_RESID_REALISTIC":partial(full_realistic_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=NF_WRAPPER),
    #         # "M2OE":partial(full_rca_run,rca_explainer=M2OE,ad_model=NF_WRAPPER)
    #     }


    #     DATA_DIR = f"src/datasets/{experiment}"
    #     RESULT_DIR = f"src/results/{experiment}"
    #     PLOT_DIR = f"src/plots/{experiment}"

    #     run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
    #     evaluate_rca(RESULT_DIR,experiment,PLOT_DIR)
    #     plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment in ["rca_cc_only_meas","rca_cc_only_mech","rca_cc_only_meas_2","rca_cc_only_mech_2"]:
            
            from src.experiments.evaluate_rca_cc import (
                run_rca,
                evaluate_rca 
            )

            methods = {
                "CALI_TRUE_NEW":cali,
                "CALI_LEARN_NEW":cali,
                "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=NF_WRAPPER),
                "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=NF_WRAPPER),
                "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=NF_WRAPPER,rca_params={"epochs":10,"batch_size":64}),
                "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=NF_WRAPPER),
                "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=NF_WRAPPER),
            }


            DATA_DIR = f"src/datasets/{experiment}"
            RESULT_DIR = f"src/results/{experiment}"
            PLOT_DIR = f"src/plots/{experiment}"

            run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_rca(RESULT_DIR,experiment,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment in ["detection_cc","detection_cc_2","detection_cc_mid","detection_cc_strong"]:
        from src.experiments.evaluate_detection_cc import (
            run_detection,
            evaluate_detection 
        )

        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "AE":ae_wrapper,
            "NF":nf_wrapper,
            "LOF":lof_wrapper,
            "IFOREST": iforest_wrapper,
            "DEEPSVDD":deepsvdd_wrapper,
            "COPOD":copod_wrapper
        }


        DATA_DIR = f"src/datasets/{experiment}"
        RESULT_DIR = f"src/results/{experiment}"
        PLOT_DIR = f"src/plots/{experiment}"

        run_detection(methods,DATA_DIR,experiment,RESULT_DIR)
        evaluate_detection(RESULT_DIR,experiment,PLOT_DIR)
        plot_all(RESULT_DIR,PLOT_DIR)


   


    elif experiment == "detection_hyper":
        from src.experiments.evaluate_competitors import (
            run_sample_ad,
            evaluate_sample_ad
        )


        EPOCHS = [10,100,500]
        BATCH_SIZES = [32,64,128]

        N_ESTIMATORS = [100,200,300]

        N_NEIGHBORS = [10,20,40]


        methods = {
        f"AE_bs_{bs}_epochs_{e}":partial(ae_wrapper,batch_size=bs,epochs=e) for e in EPOCHS for bs in BATCH_SIZES
        }

        methods.update({
        f"NF_bs_{bs}_epochs_{e}":partial(nf_wrapper,batch_size=bs,epochs=e) for e in EPOCHS for bs in BATCH_SIZES
        })

        methods.update({
        f"DEEPSVDD_bs_{bs}_epochs_{e}":partial(deepsvdd_wrapper,batch_size=bs,epochs=e) for e in EPOCHS for bs in BATCH_SIZES
        })



        methods.update({
        f"IForest_n_estimators_{n_e}":partial(iforest_wrapper,n_estimators=n_e) for n_e in N_ESTIMATORS
        })


        methods.update({
        f"LOF_n_estimators_{n_n}":partial(lof_wrapper,n_neighbors=n_n) for n_n in N_NEIGHBORS
        })

        for mean_shift in [0, 1, 2, 3, 4, 5]:
            
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
            
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"

            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"


            run_sample_ad(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_sample_ad(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)



    elif experiment == "rca_hyper":
        
        EPOCHS = [10,100,500]
        BATCH_SIZES = [32,64,128]
        N_NEIGHBORS = [10,50,100]
        
        run_rca = partial(full_rca_run,ad_model=AutoEncoder)
        

        methods = {}
        # methods.update({
        #     f"ROOTCLAM_bs_{bs}_epochs_{e}":partial(run_vaca,batch_size=bs,max_epochs=e) for e in EPOCHS for bs in BATCH_SIZES
        # })

        methods.update({
            f"M2OE_bs_{bs}_epochs_{e}_n{n_n}":partial(run_rca,rca_explainer=M2OE,rca_params={"batch_size":bs,"epochs":e,"k_neighbors":n_n}) for e in EPOCHS for bs in BATCH_SIZES for n_n in N_NEIGHBORS
        })

        density_scorer = lambda: ITAnomalyScorer(InverseDensityScorer())
        median_scorer = RescaledMedianCDFQuantileScorer 

        it_anom_scores = [density_scorer, median_scorer]

  


        methods.update({
            f"SCORE_ORDER_it_scorer_{i}":partial(run_rca,rca_explainer=SCORE_ORDER,rca_params={"it_anom_scorer":anom_scorer}) for i,anom_scorer in enumerate(it_anom_scores)
        })

        methods.update({
            f"SMOOTH_TRAVERSAL_it_scorer_{i}":partial(run_rca,rca_explainer=SMOOTH_TRAVERSAL,rca_params={"it_anom_scorer":anom_scorer}) for i,anom_scorer in enumerate(it_anom_scores)
        })


        methods.update({
            f"AE_RECON_bs_{bs}_epochs_{e}":partial(run_rca,rca_explainer=AE_REC,rca_params={"epochs":e,"batch_size":bs},ad_model=AutoEncoder) for e in EPOCHS for bs in BATCH_SIZES 
        })
        
        


        from src.experiments.evaluate_rca import (
            run_rca,
            evaluate_rca
        )

      

        for mean_shift in [3, 4, 5]:
            
            
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
            
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"

            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"

            run_rca(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_rca(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

       
    
    elif experiment == "classification_hyper":
        
        from src.experiments.evaluate_classification import (
            run_classification,
            evaluate_classification
        )

        from src.cali.classifier_baseline import classify

        ods = ["kde","med"]
        alphas = [0.1,0.05,0.01,0.001]
        correction = [True,False]

        methods = {
            f"BASELINE_alpha_{alpha}_od_{od_name}_correction_{c}":{"method":partial(classify,alpha=alpha,detector=od_name,correction=c),"params":{}} for alpha in alphas for od_name in ods for c in correction
        }

        methods["CALI"] = {"method":cali,"params":{}}

        for mean_shift in [3,4,5]:
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
                
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"

            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"
            run_classification(methods,DATA_DIR,RESULT_DIR)
            evaluate_classification(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "classification_regression_compare":
        
        from src.experiments.evaluate_classification import (
            run_classification,
            evaluate_classification
        )

        base_settings = {
            "mode": "default",
            "scale": False,
            "loss": "L1",
            "discrete": False,
            "qu": 0.5,
        }

        methods = {
            "CALI_QGAM": {
                "method": cali,
                "params": {
                    "regression_alg": "QGAM",
                    "regression_settings": {
                        **base_settings,
                        "backend": "r_qgam_strict",
                    },
                },
            },
            "CALI_SKLEARN_QUANTILE_SPLINE": {
                "method": cali,
                "params": {
                    "regression_alg": "SKLEARN_QUANTILE_SPLINE",
                    "regression_settings": base_settings,
                },
            },
            "CALI_PYGAM_EXPECTILE": {
                "method": cali,
                "params": {
                    "regression_alg": "PYGAM_EXPECTILE",
                    "regression_settings": base_settings,
                },
            },
        }

        mean_shift = 6
        DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
        RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"
        PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"

        run_classification(methods, DATA_DIR, RESULT_DIR)
        evaluate_classification(RESULT_DIR, PLOT_DIR)
        plot_all(RESULT_DIR, PLOT_DIR)


    elif experiment == "classification_rc_count_sensitivity":

        from src.experiments.evaluate_classification import (
            run_classification,
            evaluate_classification
        )
        from src.cali.classifier_baseline import classify

        methods =  {
            "CALI": {
                "method":cali,
                "params":{
                    
                }
            },
            "CALI_LEARN": {
                "method":cali,
                "params":{
                   
                    }
                
                
                },
            "BASLINE":{"method":partial(classify,alpha=0.01,detector="kde",),"params":{}}
            }
        

        for rc_count in [1,3,5,8]:
            
            DATA_DIR = f"src/datasets/{experiment}_rc_count_{rc_count}"
            
            RESULT_DIR = f"src/results/{experiment}_rc_count_{rc_count}"

            PLOT_DIR = f"src/plots/{experiment}_rc_count_{rc_count}"
            run_classification(methods,DATA_DIR,RESULT_DIR)
            evaluate_classification(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)



    elif experiment in [
        "classification_priors_small_default",
        "classification_priors_small_single",
        "classification_priors_small_multi",
    ]:

        from src.experiments.evaluate_classification import (
            run_classification,
            evaluate_classification
        )

        methods = {
            "CALI_TRUE": {"method": cali, "params": {}},
            "CALI_PRIOR_EXHAUSTIVE": {
                "method": cali_with_priors_exhaustive_oracle,
                "params": {"pass_labels": True},
            },
            "CALI_PRIOR_SINGLE_ROOT": {
                "method": cali_with_priors_single_root_oracle,
                "params": {"pass_labels": True},
            },
        }

        for mean_shift in [3, 4, 5]:
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"
            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"
            run_classification(methods, DATA_DIR, RESULT_DIR)
            evaluate_classification(
                RESULT_DIR,
                PLOT_DIR,
                common_tp_only=(experiment == "classification_priors_small_default"),
            )
            plot_all(RESULT_DIR, PLOT_DIR)

    elif experiment == "classification":

       

        from src.experiments.evaluate_classification import (
            run_classification,
            evaluate_classification
        )

        from src.cali.classifier_baseline import classify

        od = "kde"
        alpha = 0.1
        correction = True

        methods = {
            f"BASELINE_TRUE":{"method":partial(classify,alpha=alpha,detector=od,correction=correction),"params":{}} ,
            f"BASELINE_LEARN":{"method":partial(classify,alpha=alpha,detector=od,correction=correction),"params":{}} ,

        }

        methods["CALI_LEARN"] = {"method":cali,"params":{}}
        methods["CALI_TRUE"] = {"method":cali,"params":{}}
        

        for mean_shift in [6,7,8]:
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
                
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"

            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"
            run_classification(methods,DATA_DIR,RESULT_DIR)
            evaluate_classification(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "classification_correlated_measurement":

        from src.experiments.evaluate_classification import (
            run_classification,
            evaluate_classification
        )

        from src.cali.classifier_baseline import classify

        methods = {
            "BASELINE_TRUE":{"method":partial(classify,alpha=0.1,detector="kde",correction=True),"params":{}} ,
            # "BASELINE_LEARN":{"method":partial(classify,alpha=0.1,detector="kde",correction=True),"params":{}} ,
            "CALI_TRUE":{"method":cali,"params":{}},
            # "CALI_LEARN":{"method":cali,"params":{}},
        }

        for mean_shift in [3, 4, 5]:
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"
            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"
            run_classification(methods,DATA_DIR,RESULT_DIR)
            evaluate_classification(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "classification_heteroscedasticity":

        from src.experiments.evaluate_classification import (
            run_classification,
            evaluate_classification
        )

        from src.cali.classifier_baseline import classify

        methods = {
            "BASELINE_TRUE":{"method":partial(classify,alpha=0.1,detector="kde",correction=True),"params":{}} ,
            # "BASELINE_LEARN":{"method":partial(classify,alpha=0.1,detector="kde",correction=True),"params":{}} ,
            "CALI_TRUE":{"method":cali,"params":{}},
            # "CALI_LEARN":{"method":cali,"params":{}},
        }

        for heteroscedasticity_strength in [0.0,0.25, 0.5,0.75, 1.0]:
            DATA_DIR = f"src/datasets/{experiment}_hetero_{heteroscedasticity_strength}"
            RESULT_DIR = f"src/results/{experiment}_hetero_{heteroscedasticity_strength}"
            PLOT_DIR = f"src/plots/{experiment}_hetero_{heteroscedasticity_strength}"
            run_classification(methods,DATA_DIR,RESULT_DIR)
            evaluate_classification(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "classification_dag_misspecification":

        from src.experiments.evaluate_classification import (
            run_classification,
            evaluate_classification
        )

        from src.cali.classifier_baseline import classify

        methods = {
            "BASELINE":{"method":partial(classify,alpha=0.1,detector="kde",correction=True),"params":{}} ,
            "BASELINE_LEARN":{"method":partial(classify,alpha=0.1,detector="kde",correction=True),"params":{}} ,
            "CALI_TRUE":{"method":cali,"params":{}},
            "CALI_LEARN":{"method":cali,"params":{}},
        }

        for corruption_type in ["add", "remove", "reverse"]:
            for corruption_ratio in [0.1, 0.3, 0.5]:
                DATA_DIR = f"src/datasets/{experiment}_{corruption_type}_{corruption_ratio}"
                RESULT_DIR = f"src/results/{experiment}_{corruption_type}_{corruption_ratio}"
                PLOT_DIR = f"src/plots/{experiment}_{corruption_type}_{corruption_ratio}"
                run_classification(methods,DATA_DIR,RESULT_DIR)
                evaluate_classification(RESULT_DIR,PLOT_DIR)
                plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "classification_soft_interventions":

        from src.experiments.evaluate_classification import (
            run_classification,
            evaluate_classification
        )

        from src.cali.classifier_baseline import classify

        methods = {
            "BASELINE_NEW":{"method":partial(classify,alpha=0.1,detector="kde",correction=True),"params":{}} ,
            "BASELINE_LEARN_NEW":{"method":partial(classify,alpha=0.1,detector="kde",correction=True),"params":{}} ,
            "CALI_NEW_2.0":{"method":cali,"params":{}},
            "CALI_LEARN_NEW":{"method":cali,"params":{}},
        }

        for soft_intervention_type in ["soft_noise", "mechanism"]:
            for mean_shift in [3, 4, 5]:
                DATA_DIR = f"src/datasets/{experiment}_{soft_intervention_type}_strength_{mean_shift}"
                RESULT_DIR = f"src/results/{experiment}_{soft_intervention_type}_strength_{mean_shift}"
                PLOT_DIR = f"src/plots/{experiment}_{soft_intervention_type}_strength_{mean_shift}"
                run_classification(methods,DATA_DIR,RESULT_DIR)
                evaluate_classification(RESULT_DIR,PLOT_DIR)
                plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "classification_confounding":

        from src.experiments.evaluate_classification import (
            run_classification,
            evaluate_classification
        )

        from src.cali.classifier_baseline import classify

        methods = {
            "BASELINE_NEW":{"method":partial(classify,alpha=0.1,detector="kde",correction=True),"params":{}} ,
            "BASELINE_LEARN_NEW":{"method":partial(classify,alpha=0.1,detector="kde",correction=True),"params":{}} ,
            "CALI_NEW_2.0":{"method":cali,"params":{}},
            "CALI_LEARN_NEW":{"method":cali,"params":{}},
        }

        for hidden_var_ratio in [0.1, 0.25, 0.5]:
            DATA_DIR = f"src/datasets/{experiment}_hidden_{hidden_var_ratio}"
            RESULT_DIR = f"src/results/{experiment}_hidden_{hidden_var_ratio}"
            PLOT_DIR = f"src/plots/{experiment}_hidden_{hidden_var_ratio}"
            run_classification(methods,DATA_DIR,RESULT_DIR)
            evaluate_classification(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "sample_classification":
        from src.cali.classifier_baseline import classify_lasosa
        methods = {
            f"LAROSA_alpha_{alpha}":{"method":partial(classify_lasosa,alpha=alpha),"params":{}} for alpha in [0.0001,0.001,0.01,0.05,0.1]
            }


        methods["CALI_NEW"] = {"method":cali,"params":{}}
        methods["CALI_LEARN_NEW"] = {"method":cali,"params":{}} 


        from src.experiments.evaluate_sample_classification import (
            run_classification,
            evaluate_classification
        )

        for mean_shift in [3,4,5]:
            
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
                
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"

            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"
            
            run_classification(methods,DATA_DIR,RESULT_DIR)
            evaluate_classification(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)


    elif experiment == "classification_cc":
        
        from src.experiments.evaluate_classification_cc import (
            run_classification,
            evaluate_classification
        )

        from src.cali.classifier_baseline import classify
        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "BASELINE":partial(classify,alpha=0.1,correction=True,detector="kde"),
            "BASELINE_LEARN":partial(classify,alpha=0.1,correction=True,detector="kde")
            # "BASE_LINE":
        }

        DATA_DIR = f"src/datasets/{experiment}"
        RESULT_DIR = f"src/results/{experiment}"
        PLOT_DIR = f"src/plots/{experiment}"

        run_classification(methods,DATA_DIR,experiment,RESULT_DIR)
        evaluate_classification(RESULT_DIR,PLOT_DIR)
        plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment == "classification_cc_2":
        
        from src.experiments.evaluate_classification_cc import (
            run_classification,
            evaluate_classification
        )

        from src.cali.classifier_baseline import classify
        methods = {
            "CALI_TRUE_NEW":cali,
            "CALI_LEARN_NEW":cali,
            "BASELINE_NEW":partial(classify,alpha=0.1,correction=True,detector="kde"),
            "BASELINE_LEARN_NEW":partial(classify,alpha=0.1,correction=True,detector="kde")
            # "BASE_LINE":
        }

        DATA_DIR = f"src/datasets/{experiment}"
        RESULT_DIR = f"src/results/{experiment}"
        PLOT_DIR = f"src/plots/{experiment}"

        # run_classification(methods,DATA_DIR,experiment,RESULT_DIR)
        evaluate_classification(RESULT_DIR,PLOT_DIR)
        plot_all(RESULT_DIR,PLOT_DIR)

    elif experiment in ["classification_cc_strong","classification_cc_mid"]:
    
        from src.experiments.evaluate_classification_cc import (
            run_classification,
            evaluate_classification
        )

        from src.cali.classifier_baseline import classify
        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "BASELINE":partial(classify,alpha=0.1,correction=True,detector="kde"),
            "BASELINE_LEARN":partial(classify,alpha=0.1,correction=True,detector="kde")
            # "BASE_LINE":
        }

        DATA_DIR = f"src/datasets/{experiment}"
        RESULT_DIR = f"src/results/{experiment}"
        PLOT_DIR = f"src/plots/{experiment}"

        run_classification(methods,DATA_DIR,experiment,RESULT_DIR)
        evaluate_classification(RESULT_DIR,PLOT_DIR)
        plot_all(RESULT_DIR,PLOT_DIR)
 
                
    elif experiment == "causal_discovery":
        
        from src.experiments.evaluate_causal_discovery import (
            run_causal_discovery,
            evaluate_causal_discovery
        )

        methods = {

            "SCORE":score_wrapper,
            "CAM":cam_wrapper,
            "nogam":nogam_wrapper,
            "topic":topic_wrapper,
            "notears_mlp":notears_mlp_wrapper,

        }
        
        for mean_shift in range(3,6):
            DATA_DIR = f"src/datasets/{experiment}_strength_{mean_shift}"
            
            RESULT_DIR = f"src/results/{experiment}_strength_{mean_shift}"

            PLOT_DIR = f"src/plots/{experiment}_strength_{mean_shift}"
            run_causal_discovery(methods,DATA_DIR,experiment,RESULT_DIR)
            evaluate_causal_discovery(RESULT_DIR,PLOT_DIR)
            plot_all(RESULT_DIR,PLOT_DIR)
    
    elif experiment == "sachs":

        from src.experiments.evaluate_rca_sachs import (
            run_rca,
            evaluate_rca
        )

        from src.cali.classifier_baseline import classify


        DATA_DIR = "src/datasets/sachs"
        RESULT_DIR = f"src/results/{experiment}"
        PLOT_DIR = f"src/plots/{experiment}"

        methods = {
            "CALI_TRUE":cali,
            "CALI_LEARN":cali,
            "SCORE_ORDER":partial(full_rca_run,rca_explainer=SCORE_ORDER,ad_model=NF_WRAPPER),
            "SMOOTH_TRAVERSAL":partial(full_rca_run,rca_explainer=SMOOTH_TRAVERSAL,ad_model=NF_WRAPPER),
            "AE_REC":partial(full_rca_run,rca_explainer=AE_REC,ad_model=NF_WRAPPER,rca_params={"epochs":10,"batch_size":64}),
            "AE_SHAP":partial(full_rca_run,rca_explainer=SHAPExplainer,ad_model=NF_WRAPPER),
            "BASELINE":partial(classify,alpha=0.1,correction=True,detector="kde"),
            "BASELINE_LEARN":partial(classify,alpha=0.1,correction=True,detector="kde"),
            "SCM_RESID":partial(full_rca_run,rca_explainer=SCM_LOCALIZER,ad_model=NF_WRAPPER)
            # "M2OE":partial(full_rca_run,rca_explainer=M2OE,ad_model=NF_WRAPPER)
        }

        run_rca(methods,DATA_DIR,RESULT_DIR)
        evaluate_rca(RESULT_DIR,PLOT_DIR)
        plot_all(RESULT_DIR,PLOT_DIR)


    #
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")



# =====================================================
#                      MAIN
# =====================================================
if __name__ == "__main__":
    # import torch.multiprocessing as mp
    # import multiprocessing as mp
    # mp.set_start_method('spawn', force=True)  # or skip if you do Option A
    import os
    import numpy
    import scipy
    import torch
    import dice_ml
    import sklearn
    import tensorflow
    import tensorflow.keras

    import pyod 
    import pzflow

    import torch

    torch.set_default_dtype(torch.float32)
    from dowhy.gcm import RescaledMedianCDFQuantileScorer, ITAnomalyScorer, InverseDensityScorer
    from pyod.models.auto_encoder import AutoEncoder
    
    from src.cali.cali import cali
    from src.cali.cali_with_priors import (
        cali_with_priors_exhaustive_oracle,
        cali_with_priors_single_root_oracle,
    )

# from code.src.cali.scm_shap import shap_qgam_scm



    from src.plots.plot import plot_all
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"




    CHOICES = [
            
            "detection", 
            "detection_cc",
            "detection_cc_strong",
            "detection_cc_mid",
            "detection_cc_2",
            "detection_sachs",
            "detection_hyper", 
            "rca_hyper", 
            "causal_discovery",
            "classification",
            "classification_regression_compare",
            "classification_priors_small_default",
            "classification_priors_small_single",
            "classification_priors_small_multi",
            "classification_cc",
            "classification_cc_2",
            "classification_cc_strong",
            "classification_cc_mid",
            "classification_hyper",
            "classification_rc_count_sensitivity",
            "classification_correlated_measurement",
            "classification_dag_misspecification",
            "classification_heteroscedasticity",
            "classification_soft_interventions",
            "classification_confounding",
            "rca",
            "rca_regression_compare",
            "rca_priors_small_default",
            "rca_priors_small_single",
            "rca_priors_small_multi",
            "rca_only_mech",
            "rca_only_meas",
            "rca_cc",
            "rca_cc_strong",
            "rca_cc_mid",
            "rca_cc_2",
            "rca_cc_2000",
            "rca_cc_only_mech",
            "rca_cc_only_meas",
            "rca_cc_only_mech_2",
            "rca_cc_only_meas_2",
            "rca_sample_sensitivity",
            "rca_feature_sensitivity",
            "rca_edge_prob_sensitivity",
            "rca_rc_count_sensitivity",
            "rca_correlated_measurement",
            "rca_dag_misspecification",
            "rca_heteroscedasticity",
            "rca_soft_interventions",
            "rca_confounding",
            "sachs",
            "sample_classification"
            
            ]


    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    import argparse
    import os
    from functools import partial

    

    # ====== IMPORT DATA GENERATION ======
    from src.datasets.generate_synthetic_data import generate_experiment_data
    from src.datasets.process_real_data_cc import process_real_data

    # ====== IMPORT EXPERIMENT EXECUTION ======
   
    from src.cali.cali import cali
    from src.cali.cali_with_priors import (
        cali_with_priors_exhaustive_oracle,
        cali_with_priors_single_root_oracle,
    )
    # from code.src.cali.scm_shap import shap_qgam_scm
    from src.plots.plot import plot_all


    import faulthandler, sys
    faulthandler.enable(file=sys.stderr, all_threads=True)
    from src.competitors.feature_attribution import (
        run_dice,
        run_dice_realistic,
        run_m2oe_method,
        run_m2oe_method_realistic,
        run_vaca,
        run_vaca_realistic,
        shap_feature_explainer
    )
    from src.competitors.sample_level_ad import (
        ae_wrapper,
        nf_wrapper,
        copod_wrapper,
        deepsvdd_wrapper,
        iforest_wrapper,
        lof_wrapper,
        NF_WRAPPER
    )
    from src.competitors.causal_discovery import (
        score_wrapper,
        cam_wrapper,
        topic_wrapper,
        nogam_wrapper,
        notears_mlp_wrapper
    )


    from src.competitors.root_cause_localization import (
        SCORE_ORDER,
        SMOOTH_TRAVERSAL,
        SHAPExplainer,
        SCM_LOCALIZER,
        AE_REC,
        M2OE,
        DICE,
        full_rca_run,
        full_realistic_rca_run
    )



    from src.datasets.process_real_data_sachs import process_sachs_data

    parser = argparse.ArgumentParser(description="Generate data and run experiments.")
   
    parser.add_argument("-e", "--experiment", type=str, required=True, choices=CHOICES)
    args = parser.parse_args()

    print(f"🔧 Generating data for experiment: {args.experiment}")
    generate_data(args.experiment)

    print(f"🚀 Running experiment: {args.experiment}")
    run_experiment(args.experiment)

    print("✅ Done!")
