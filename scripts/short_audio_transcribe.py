"""短音频转录
1. 读取 ./custom_character_voice/ 目录下每个说话人子文件夹里的音频文件
2. 使用 whisper 模型识别音频语言并生成文本
3. 将结果保存到 short_character_anno.txt，每行格式为：音频路径|说话人|带语言标签的文本
"""
import whisper
import os
import json
import torchaudio
import argparse
import torch
import opencc

lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
            "en": "[EN]",
        }
cc = opencc.OpenCC('t2s')
def transcribe_one(audio_path):
    try:
        # 加载音频，并将其填充/修剪至30秒
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # 制作对数梅尔频谱并且将数据移到GPU上
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # 检测说话语言
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        lang = max(probs, key=probs.get)
        
        # Mel 频谱解码成文本
        options = whisper.DecodingOptions(beam_size=5)
        result = whisper.decode(model, mel, options)
        # 繁体转简体
        if lang == "zh":
            result.text = cc.convert(result.text)
        # 打印被识别的文本
        print(result.text)
        return lang, result.text

    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJE")
    parser.add_argument("--whisper_size", default="medium")
    args = parser.parse_args()
    if args.languages == "CJE":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
            "en": "[EN]",
        }
    elif args.languages == "CJ":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
        }
    elif args.languages == "C":
        lang2token = {
            'zh': "[ZH]",
        }
    assert (torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"
    model = whisper.load_model(args.whisper_size)
    parent_dir = "./custom_character_voice/"
    speaker_names = list(os.walk(parent_dir))[0][1]
    speaker_annos = []
    total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])
    
    # 重采样 & 转录
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    processed_files = 0
    for speaker in speaker_names:
        for i, wavfile in enumerate(list(os.walk(parent_dir + speaker))[0][2]):
            # 跳过已经处理过的文件
            if wavfile.startswith("processed_"):
                continue
            try:
                wav, sr = torchaudio.load(parent_dir + speaker + "/" + wavfile, frame_offset=0, num_frames=-1, normalize=True,
                                          channels_first=True)
                wav = wav.mean(dim=0).unsqueeze(0)
                
                # 重采样
                if sr != target_sr:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
                if wav.shape[1] / sr > 20:
                    print(f"{wavfile} too long, ignoring\n")
                save_path = parent_dir + speaker + "/" + f"processed_{i}.wav"
                torchaudio.save(save_path, wav, target_sr, channels_first=True)
                
                # 转录成文本
                lang, text = transcribe_one(save_path)
                if lang not in list(lang2token.keys()):
                    print(f"{lang} not supported, ignoring\n")
                    continue
                text = lang2token[lang] + text + lang2token[lang] + "\n"
                speaker_annos.append(save_path + "|" + speaker + "|" + text)
                # 记录处理文件数
                processed_files += 1
                print(f"Processed: {processed_files}/{total_files}")
            except:
                continue


    # 写入标记文本
    if len(speaker_annos) == 0:
        print("Warning: no short audios found, this iS expected if you have only uploaded long audios, videos or video links.")
        print("this is not expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported.")
    with open("short_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)