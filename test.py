# test.py
from data_processing import load_and_process_data

def test_load_and_process_data():
    # 加载并处理数据
    train_data, valid_data = load_and_process_data()

    # 打印一些数据
    print("Train Data (first 5 samples):")
    print(train_data[:1])  # 输出训练数据的前5个样本

    print("\nValidation Data (first 5 samples):")
    print(valid_data[:1])  # 输出验证数据的前5个样本

if __name__ == "__main__":
    test_load_and_process_data()
