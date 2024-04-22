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
    train_data_path = './data/Intra_train.xlsx'
    test_data_path = './data/Intra_test.xlsx'
    x_index = 3
    y_index = 0

    """ Model parameters """
    num_classes = 3
    feature_size = 6  # 输入特征
    hidden_size = 64  # 隐藏层维度
    dropout_prob = 0.3

    """ Train parameters """
    batch_size = 16
    learning_rate = 1e-6
    num_epochs = 600

    """ OutPut """
    model_name = 'cls'
    save_path = './results/{}.pth'.format(model_name)
