from pathlib import Path
import shutil
import random
import re
import argparse

random.seed(1234)

def parser_line(line):
    '''分割句子'''
    line = line.strip()
    file_name, _ = line.split(" ", 1)
    txt, emo = _.rsplit(" ", 1)
    
    return file_name, txt, emo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="../语音包/Emotion Speech Dataset", help="源数据集目录")
    parser.add_argument("--tar_dir", default="../语音包/ESDv2", help="目标目录")
    parser.add_argument("--parent_dir", default="./custom_character_voice/", help="存放音频文件的根目录")
    args = parser.parse_args()
    # 支持中英文
    lang2token = {
        'zh': "[ZH]",
        'en': "[EN]"
    }
    # 区分情绪
    emo2tokne = {
        '中立': "Neutral",
        '快乐': "Happy",
        '生气': "Angry",
        '伤心': "Sad",
        '惊喜': "Surprise"
    }
    
    root_dir = args.root_dir
    tar_dir = args.tar_dir
    parent_dir = args.parent_dir
    root_path = Path(root_dir)
    
    speaker_annos = []
    processed_files = 0
    total_files = 300 * 20
    for numdir in root_path.iterdir():
        # 不是目录则跳过
        if not numdir.is_dir():
            continue
        
        # 获取说话人名字
        speaker_name = numdir.name
        
        txt_path = numdir / f"{speaker_name}.txt"
        with txt_path.open("r", encoding='utf-8') as f:
            anno_lines = [line for line in f if line.strip()]
        
        # 随机挑选360条数据
        # selected_annos = random.sample(anno_lines, min(360, len(anno_lines)))
        selected_annos = []
        for i in range(0, len(anno_lines), 350):
            block = anno_lines[i:i+350]
            if not block:
                continue
            selected_annos.extend(random.sample(block, min(60, len(block))))
        for anno in selected_annos:
            file_name, txt, emo = anno.strip().split("\t")
            if int(speaker_name) <= 10:
                lang = 'zh'
                emo = emo2tokne[emo]
            else:
                lang = 'en'
            # 复制被选中的音频到目标文件夹下
            save_path = parent_dir + speaker_name + "/" + f"{file_name}.wav"
            src_path = root_path / speaker_name / emo / f"{file_name}.wav"
            tar_path = Path(tar_dir) / speaker_name / f"{file_name}.wav"
            shutil.copy2(src_path, tar_path)
            # 被选中的音频按照格式保存
            txt = lang2token[lang] + txt + lang2token[lang] + "\n"
            speaker_annos.append(f"{save_path}|{speaker_name}|{txt}")
            # 记录处理文件数
            processed_files += 1
            print(f"Processed: {processed_files}/{total_files}")
    
    # 写入标记文本
    if len(speaker_annos) == 0:
        print("WARING, THIS IS NOT ANNOS!!!")
    with open("short_character_anno.txt", "w", encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
    