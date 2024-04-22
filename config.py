class RegressionConfig:
    train_data_path = './data/Intra_train.xlsx'
    test_data_path = './data/Intra_test.xlsx'
    timestep = 1  # 时间步长
    batch_size = 16
    learning_rate = 1e-6
    feature_size = 6  # 输入特征
    hidden_size = 196  # 隐藏层维度
    output_size = 1
    num_layers = 2  # GRU层数
    dropout_prob = 0.3
    num_epochs = 600
    best_loss = float('inf')
    model_name = 'cnn-lstm'
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
