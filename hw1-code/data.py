import pandas as pd
import json

# 读取CSV文件
csv_data = pd.read_csv("e:/hw/align/align/hw1-code/finetune/data/train.csv", sep=",")

# 转换为JSON格式并保存
with open("e:/hw/align/align/hw1-code/finetune/data/train.json", 'w', encoding='utf-8') as f:
    json.dump(csv_data.to_dict('records'), f, ensure_ascii=False, indent=4)