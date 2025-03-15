# data_processing.py
from datasets import load_dataset
from transformers import MarianTokenizer

def load_and_process_data():
    # 加载数据集
    dataset = load_dataset('wmt14', 'de-en')

    # 加载分词器
    model_name = 'Helsinki-NLP/opus-mt-en-de'
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    def clean_text(text):
        # 文本清洗：去除标点符号，空格处理等
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s,\.!?\'\"-]', '', text)
        text = text.lower()
        return text

    def tokenize_function(examples):
        # 这里更新了列名的访问方式
        model_inputs = tokenizer(examples['en'], padding='max_length', truncation=True, max_length=128)
        labels = tokenizer(examples['de'], padding='max_length', truncation=True, max_length=128)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # 清洗并分词数据集
    train_data = dataset['train'].map(lambda x: {'en': clean_text(x['translation']['en']), 'de': clean_text(x['translation']['de'])})
    valid_data = dataset['validation'].map(lambda x: {'en': clean_text(x['translation']['en']), 'de': clean_text(x['translation']['de'])})
    
    train_data = train_data.map(tokenize_function, batched=True)
    valid_data = valid_data.map(tokenize_function, batched=True)

    return train_data, valid_data
