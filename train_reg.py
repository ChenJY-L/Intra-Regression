"""
                  _ooOoo_
                 o8888888o
                 88" . "88
                 (| -_- |)
                  O\ = /O
              ____/`---'\____
            .   ' \\| |// `.
             / \\||| : |||// \
           / _||||| -:- |||||- \
             | | \\\ - /// | |
           | \_| ''\---/'' | |
            \ .-\__ `-` ___/-. /
         ___`. .' /--.--\ `. . __
      ."" '< `.___\_<|>_/___.' >'"".
     | | : `- \`.;`\ _ /`;.`/ - ` : | |
         \ \ `-. \_ __\ /__ _/ .-` / /
 ======`-.____`-.___\_____/___.-`____.-'======
                    `=---='

  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             佛祖保佑       永无BUG
"""

# Encoding utf-8
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import RegressionConfig
from models_reg import *
from utilis.evaluate import *
from datasets import *


def train():
    # Load config
    config = RegressionConfig()

    # 1. 创建数据集
    # 读取文件
    df = read_data(config.data_path)
    train_df, test_df = train_test_split(df, test_size=config.ratio, random_state=42)
    del df

    x_scaler = MinMaxScaler()
    train_df.iloc[:, 3:] = x_scaler.fit_transform(train_df.iloc[:, 3:])
    test_df.iloc[:, 3:] = x_scaler.transform(test_df.iloc[:, 3:])

    y_scaler = MinMaxScaler()
    train_df.iloc[:, 2] = y_scaler.fit_transform(train_df.iloc[:, 2].values.reshape(-1, 1))
    test_df.iloc[:, 2] = y_scaler.transform(test_df.iloc[:, 2].values.reshape(-1, 1))

    train_data = IntraDataset(train_df)
    test_data = IntraDataset(test_df)

    # 2. 创建迭代器
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)

    # 3. 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # model = MyGRU(config.feature_size, config.hidden_size, config.num_layers, config.output_size, config.dropout_prob)
    # model = CNN1DRegression(config.timestep, config.output_size)
    # model = NNModel(config.feature_size, config.hidden_size, config.output_size)
    model = ResNet1D(config.feature_size, config.hidden_size, config.output_size, config.num_layers,
                     config.dropout_prob)
    model = model.to(device)
    loss_fn = nn.MSELoss()
    current_optim = "RMS"
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=1e-2, weight_decay=1e-3)
    # optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=1e-2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.num_epochs)

    train_losses = []
    test_losses = []

    best_loss = float('inf')

    # 4. 训练模型
    for epoch in range(config.num_epochs):
        model.train()

        running_loss = 0.0
        # train_bar = tqdm(train_loader)

        for data in train_loader:
            x, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # train_bar.set_description('[{}/{}] Train Epoch: loss:{:.5f} '.format(epoch + 1, config.num_epochs, loss))
        train_losses.append(running_loss / len(train_loader))
        scheduler.step()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            # test_bar = tqdm(test_loader)
            for data in test_loader:
                x, y = data[0].to(device), data[1].to(device)
                y_pred = model(x)
                test_loss += loss_fn(y_pred, y)
                # test_bar.set_description('Test: loss:{:.5f}'.format(test_loss / len(test_loader)))

        test_loss /= len(test_loader)
        test_losses.append(test_loss.cpu())

        # Print training progress
        print(
            f"Epoch {epoch + 1}/{config.num_epochs}, "
            f"Train Loss: {running_loss / len(train_loader):.4f},"
            f"Test Loss: {test_loss / len(test_loader):.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            try:
                torch.save(model.state_dict(), config.save_path)
                print('\nBest loss: {}\n'.format(best_loss))
            finally:
                pass

        if epoch >= config.num_epochs * 2 / 3 and current_optim == "Adam":
            print("Use SGD ")
            current_optim = "SGD"
            optimizer = optim.SGD(model.parameters(), lr=scheduler.get_last_lr()[-1], momentum=1e-2, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs // 3)
        if epoch >= config.num_epochs / 3 and current_optim == "RMS":
            current_optim = "Adam"
            print("Use Adam")
            optimizer = optim.Adam(model.parameters(), lr=scheduler.get_last_lr()[-1])

    print('Finished')
    # torch.save(model.state_dict(), config.save_path + "last")
    # 5. 评估模型
    plot_loss(train_losses, test_losses)

    # 获取所有预测值和目标值，用于绘制预测结果对比图
    all_train_predictions, all_train_targets = evaluate_model(model, train_loader, y_scaler)
    plot_predictions(all_train_predictions, all_train_targets)

    all_test_predictions, all_test_targets = evaluate_model(model, test_loader, y_scaler)
    plot_predictions(all_test_predictions, all_test_targets)
    # print(all_predictions)

    # print(all_targets)


if __name__ == '__main__':
    train()
