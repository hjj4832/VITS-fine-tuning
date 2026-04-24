"""长音频转录
1. 使用whipser模型对长音频进行语音识别
2. 按时间段拆分长音频为多个子片段
3. 切分音频段保存到./segmented_character_voice/<character_name>/ 目录
4. 把每段音频片段路径、角色名和文本注释写入 ./long_character_anno.txt
"""
from moviepy.editor import AudioFileClip
import whisper
import os
import json
import torchaudio
import librosa
import torch
import argparse
parent_dir = "./denoised_audio/"
filelist = list(os.walk(parent_dir))[0][2]
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
    assert(torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"
    
    # 重采样 & 转录
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    model = whisper.load_model(args.whisper_size)
    speaker_annos = []
    for file in filelist:
        audio_path = os.path.join(parent_dir, file)
        print(f"Transcribing {audio_path}...\n")

        options = dict(beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)

        result = model.transcribe(audio_path, word_timestamps=True, **transcribe_options)
        segments = result["segments"]
        lang = result['language']
        if lang not in lang2token:
            print(f"{lang} not supported, ignoring...\n")
            continue

        character_name = file.rstrip(".wav").split("_")[0]
        code = file.rstrip(".wav").split("_")[1]
        outdir = os.path.join("./segmented_character_voice", character_name)
        os.makedirs(outdir, exist_ok=True)

        wav, sr = torchaudio.load(
            audio_path,
            frame_offset=0,
            num_frames=-1,
            normalize=True,
            channels_first=True
        )

        for i, seg in enumerate(segments):
            start_time = seg['start']
            end_time = seg['end']
            text = seg['text']
            text_tokened = lang2token[lang] + text.replace("\n", "") + lang2token[lang] + "\n"
            start_idx = int(start_time * sr)
            end_idx = int(end_time * sr)
            num_samples = end_idx - start_idx
            if num_samples <= 0:
                print(f"Skipping zero-length segment: start={start_time}, end={end_time}")
                continue
            wav_seg = wav[:, start_idx:end_idx]
            if wav_seg.shape[1] == 0:
                print(f"Skipping empty segment i={i}, shape={wav_seg.shape}")
                continue
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                wav_seg = resampler(wav_seg)

            wav_seg_name = f"{character_name}_{code}_{i}.wav"
            savepth = os.path.join(outdir, wav_seg_name)
            speaker_annos.append(savepth + "|" + character_name + "|" + text_tokened)
            print(f"Transcribed segment: {speaker_annos[-1]}")
            torchaudio.save(savepth, wav_seg, target_sr, channels_first=True)
            
    if len(speaker_annos) == 0:
        print("Warning: no long audios & videos found, this iS expected if you have only uploaded short audios")
        print("this iS not expected if you have uploaded any long audios, videos or video links. Please check your file structure or make sure your audio/video language is supported.")
    with open("./long_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
