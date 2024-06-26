class RegressionConfig:
    """ Config class for regression"""
    """ Dataset parameters"""
    data_path = './data/Intra_CLS1.xlsx'
    test_data_path = './data/Intra-extra.xlsx'
    ratio = 0.2  # train:test

    """ Training parameters """
    batch_size = 9
    learning_rate = 1e-5
    momentum = 1e-2
    weight_decay = 1e-3
    num_epochs = 300

    """ Model parameters """
    seed = 666666    # 随机种子
    feature_size = 6  # 输入特征
    hidden_size = 142  # 隐藏层维度  (160)
    num_layers = 3  # CNN层数
    output_size = 1
    dropout_prob = 0.3

    """ Output parameters """
    model_name = 'reg'
    save_path = './results/{}.pth'.format(model_name)


class ClsConfig:
    """ Config Class for CLS """

    """ Dataset parameters"""
    data_path = './data/Intra_CLS.xlsx'
    ratio = 0.3  # train:test
    x_index = 3
    y_index = 0
    classes = [2, 5, 10]
    idx_to_class = {i: j for i, j in enumerate(classes)}
    class_to_idx = {value: key for key, value in idx_to_class.items()}

    """ Model parameters """
    num_classes = 3
    feature_size = 6  # 输入特征
    hidden_size = 64  # 隐藏层维度
    dropout_prob = 0.3

    """ Train parameters """
    batch_size = 16
    learning_rate = 1e-6
    num_epochs = 100

    """ OutPut """
    model_name = 'cls'
    save_path = './results/{}.pth'.format(model_name)
