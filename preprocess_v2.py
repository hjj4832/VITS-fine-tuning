"""数据处理脚本
"""
import os
import argparse
import json
import sys
sys.setrecursionlimit(500000)  # Fix the error message of RecursionError: maximum recursion depth exceeded while calling a Python object.  You can change the number as you want.

def load_annos():
    """读取 short_character_anno.txt 和 long_character_anno.txt"""
    new_annos = []

    if os.path.exists("./short_character_anno.txt"):
        with open("./short_character_anno.txt", "r", encoding="utf-8") as f:
            new_annos += f.readlines()

    if os.path.exists("./long_character_anno.txt"):
        with open("./long_character_anno.txt", "r", encoding="utf-8") as f:
            new_annos += f.readlines()

    return new_annos

def get_speakers(annos):
    """从原始标注中收集说话人 ID"""
    speakers = []

    for line in annos:
        parsed = parse_line(line)
        if parsed is None:
            continue

        _, speaker, _ = parsed

        if speaker not in speakers:
            speakers.append(speaker)

    assert len(speakers) != 0, "No audio file found. Please check your uploaded file structure."

    return speakers

def update_hps_for_train(hps, speakers):
    """
    更新最终训练配置。

    注意：
    - 预处理阶段使用 ce_cleaners
    - 训练阶段读取的是已经清洗好的 IPA，所以 text_cleaners 置空
    - n_languages=3，对应：
        0 = PAD / blank
        1 = 中文 / cmn
        2 = 英文 / en
    """

    speaker2id = {}
    for i, speaker in enumerate(speakers):
        speaker2id[speaker] = i

    hps["data"]["n_speakers"] = len(speakers)
    hps["speakers"] = speaker2id

    hps["train"]["log_interval"] = 10
    hps["train"]["eval_interval"] = 100
    hps["train"]["batch_size"] = 16

    hps["data"]["training_files"] = "final_annotation_train.txt"
    hps["data"]["validation_files"] = "final_annotation_val.txt"

    # 训练阶段读取 final_annotation_train.txt，里面已经是 IPA
    hps["data"]["cleaned_text"] = True

    # 训练阶段不再做中文/英文转 IPA
    hps["data"]["text_cleaners"] = []

    # 中英文语言嵌入
    hps["model"]["n_languages"] = 3

    return hps, speaker2id

def clean_annos(annos, speaker2id, cleaner_names, max_text_len=150):
    """
    将原始文本清洗为 IPA，并把 speaker 名称替换成 speaker id。

    输入：
        ./xxx.wav|0001|[ZH]你好[ZH]

    输出：
        ./xxx.wav|0|ni2 χɑu2.
    """
    import text

    cleaned_annos = []

    for line in annos:
        path, speaker, txt = line.split("|")

        # 这里按原始文本长度过滤，避免特别长的句子
        if len(txt) > max_text_len:
            continue

        cleaned_text = text._clean_text(txt, cleaner_names)

        # 防止空文本进入训练集
        if cleaned_text is None or len(cleaned_text.strip()) == 0:
            continue

        cleaned_text = cleaned_text.strip()
        cleaned_text += "\n"

        cleaned_annos.append(
            path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text
        )

    return cleaned_annos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_auxiliary_data", type=bool, help="Whether to add extra data as fine-tuning helper")
    parser.add_argument("--languages", default="CE", choices=["C", "E", "CE"], help="C=中文, E=英文, CE=中英文")
    args = parser.parse_args()
    if args.languages == "CE":
        langs = ["[ZH]", "[EN]"]
    elif args.languages == "C":
        langs = ["[ZH]"]
    elif args.languages == "E":
        langs = ["[EN]"]
    
    # 1. 读取数据
    new_annos = load_annos()
    speakers = get_speakers(new_annos)

    # 2. 读取并筛选辅助数据（可选）
    old_annos = []
    if args.add_auxiliary_data:
        if not os.path.exists("./sampled_audio4ft.txt"):
            raise FileNotFoundError("sampled_audio4ft.txt not found.")
        
        with open("./sampled_audio4ft.txt", 'r', encoding='utf-8') as f:
            old_annos = f.readlines()
        
        # 根据支持语言筛选辅助数据
        filtered_old_annos = []
        for line in old_annos:
            if any(lang in line for lang in langs):
                filtered_old_annos.append(line)
        old_annos = filtered_old_annos
        for line in old_annos:
            _, speaker, _ = line.split("|")
            if speaker not in speakers:
                speakers.append(speaker)

    # 3. 修改配置文件
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    
    # 4. 保存预处理阶段用的 cleaner
    cleaner_names = hps["data"]["text_cleaners"]
    
    # 5. 生成 speaker2id，并更新最终训练配置
    hps, speaker2id = update_hps_for_train(hps, speakers)
    
    # 6. 清洗新数据：原始文本 -> IPA
    cleaned_new_annos = clean_annos(
        annos=new_annos,
        speaker2id=speaker2id,
        cleaner_names=cleaner_names,
        max_text_len=150
    )
    
    # 7. 清洗辅助数据
    cleaned_old_annos = []
    if args.add_auxiliary_data:
        cleaned_old_annos = clean_annos(
            annos=old_annos,
            speaker2id=speaker2id,
            cleaner_names=cleaner_names,
            max_text_len=150
        )

    # 8. 合并训练集
    if args.add_auxiliary_data and len(cleaned_old_annos) > 0:
        num_old_voices = len(cleaned_old_annos)
        num_new_voices = len(cleaned_new_annos)

        cc_duplicate = num_old_voices // max(num_new_voices, 1)
        if cc_duplicate == 0:
            cc_duplicate = 1

        final_annos = cleaned_old_annos + cc_duplicate * cleaned_new_annos
    else:
        final_annos = cleaned_new_annos
        
    # 9. 保存训练集
    with open("./final_annotation_train.txt", "w", encoding="utf-8") as f:
        for line in final_annos:
            f.write(line)

    # 10. 保存验证集
    # 验证集使用新数据
    with open("./final_annotation_val.txt", "w", encoding="utf-8") as f:
        for line in cleaned_new_annos:
            f.write(line)

    # 11. 保存最终训练配置
    with open("./configs/modified_finetune_speaker.json", "w", encoding="utf-8") as f:
        json.dump(hps, f, indent=2, ensure_ascii=False)

    print("===== finished =====")
    print(f"languages: {args.languages}")
    print(f"num speakers: {len(speakers)}")
    print(f"num train annos: {len(final_annos)}")
    print(f"num val annos: {len(cleaned_new_annos)}")
    print("saved: final_annotation_train.txt")
    print("saved: final_annotation_val.txt")
    print("saved: configs/modified_finetune_speaker.json")
