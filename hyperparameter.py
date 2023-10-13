def model_param_list():
    model_params = {}
    model_params['out_path'] = ''
    model_params['emb_size'] = 64
    model_params['dropout_rate'] = 0.001
    model_params['max_drug_seq'] = 1000
    model_params['max_protein_seq'] = 15000
    model_params['input_dim_drug'] = 23532
    model_params['input_dim_target'] = 16693

    # training
    model_params['training_batch_size'] = 128
    model_params['validation_batch_size'] = 128
    model_params['num_filters'] = 128
    model_params['lr'] = 0.0001
    model_params['early_stopping'] = 20
    model_params['max_epoch'] = 50
    model_params['thresh'] = 0.5
    model_params['stopping_met'] = "loss"

    # DenseNet

    model_params["feature_size"] = model_params['emb_size']  
    model_params["kernel_size"] = 5
    model_params["stride"] = 1
    model_params["n_heads"] = 6
    model_params["d_dim"] = 64
    model_params["feature"] = 128
    model_params["pooling_dropout"] = 0.1
    model_params["linear_dropout"] = 0.1

    # 我加的
    model_params['smi_dict_path'] = 'dictionary/smiles_dict.pickle'
    model_params['fas_dict_path'] = 'dictionary/fasta_dict.pickle'
    model_params['smiles_max_len'] = 1500
    model_params['fasta_max_len'] = 15000
    # model_params['smi_dict_len'] = 61  # The length of dictionary
    # model_params['fas_dict_len'] = 8083  # The length of dictionary
    model_params['smi_dict_len'] = 64  # The length of dictionary
    model_params['fas_dict_len'] = 8107  # The length of dictionary
    model_params['smi_ngram'] = 1
    model_params['fas_ngram'] = 3
    model_params['protein_kernel'] = [4, 8, 12]
    model_params['drug_kernel'] = [4, 6, 8]
    model_params['conv'] = 40

    return model_params


