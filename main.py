from data_processing import load_and_process_data

# 加载并处理数据
train_data, valid_data = load_and_process_data()

# 如果你想进行更多的验证，可以继续在主程序中处理数据
print(f"Train Data Length: {len(train_data)}")
print(f"Validation Data Length: {len(valid_data)}")
