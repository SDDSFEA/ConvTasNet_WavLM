import torch
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.dataloader import default_collate
from AudioReader import AudioReader
import torch.nn.functional as F
import random
import pandas as pd

import pandas as pd

def TranscriptReader(text_path):
    """
    从单个 CSV 文件中读取转写文本
    Args:
        text_path (str): 包含转写文本的 CSV 文件路径
    Returns:
        dict: {audio_id: "speaker_text"}
    """
    try:
        df = pd.read_csv(text_path, encoding='utf-8')
        text_dict = {}
        
        for _, row in df.iterrows():
            audio_id = str(row['ID']).strip() if pd.notna(row['ID']) else ""
            speaker_text = str(row['Speaker']) if pd.notna(row['Speaker']) else ""
            
            if audio_id:  # 确保ID不为空
                text_dict[audio_id] = speaker_text.strip()
        
        print(f"成功处理 {len(text_dict)} 条音频文本数据")
        return text_dict
    except FileNotFoundError as e:
        print(f"错误：文件不存在 {e}")
        return {}
    except KeyError as e:
        print(f"错误：CSV 文件中缺少必要的列 {e}")
        print("请确保 CSV 文件包含列：ID, Speaker")
        return {}
    except Exception as e:
        print(f"处理文件时出错：{e}")
        return {}


def pad_collate(batch):
    """
    batch: List[dict]
        dict keys: 'mix', 'ref'
        mix: Tensor [T]
        ref: List[Tensor [T]]
    """
    # ===== mix =====
    mix_list = [b['mix'] for b in batch]
    mix_lens = torch.tensor([m.shape[-1] for m in mix_list], dtype=torch.long)
    max_mix_len = max(mix_lens)

    mix_pad = torch.stack([
        F.pad(m, (0, max_mix_len - m.shape[-1]))
        for m in mix_list
    ])

    # ===== ref =====
    num_spk = len(batch[0]['ref'])
    ref_pad = []

    for spk in range(num_spk):
        spk_list = [b['ref'][spk] for b in batch]
        spk_lens = [r.shape[-1] for r in spk_list]
        max_len = max(spk_lens)

        spk_pad = torch.stack([
            F.pad(r, (0, max_len - r.shape[-1]))
            for r in spk_list
        ])
        ref_pad.append(spk_pad)
    
    # ===== transcript =====
    transcript_list1 = [b['transcript1'] for b in batch]
    transcript_list2 = [b['transcript2'] for b in batch]

    return {
        'mix': mix_pad,              # [B, T]
        'ref': ref_pad,              # List([B, T])
        'mix_len': mix_lens,           # [B]
        'transcript1': transcript_list1,  # List[str]
        'transcript2': transcript_list2
    }


def make_dataloader(is_train=True,
                    data_kwargs=None,
                    num_workers=4,
                    # chunk_size=32000,
                    batch_size=16):
    dataset = Datasets(**data_kwargs)
    return DataLoader(dataset,
                      shuffle=is_train,
                    #   chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      collate_fn=pad_collate)


class Datasets(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    '''

    def __init__(self, mix_scp=None, ref_scp=None, text_path1=None,text_path2=None, sr=8000):
        super(Datasets, self).__init__()
        self.mix_audio = AudioReader(mix_scp, sample_rate=sr)
        self.ref_audio = [AudioReader(r, sample_rate=sr) for r in ref_scp]
        self.transcript1 = TranscriptReader(text_path1)
        self.transcript2 = TranscriptReader(text_path2)

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        key = self.mix_audio.keys[index]
        mix = self.mix_audio[key]
        ref = [r[key] for r in self.ref_audio]
        transcript1 = self.transcript1.get(key, "")
        transcript2 = self.transcript2.get(key, "")
        return {
            'mix': mix,
            'ref': ref,
            'transcript1': transcript1,
            'transcript2': transcript2
        }



if __name__ == "__main__":
    datasets = Datasets('/home/likai/data1/create_scp/cv_mix.scp',
                        ['/home/likai/data1/create_scp/cv_s1.scp', '/home/likai/data1/create_scp/cv_s2.scp'])
    dataloaders = DataLoader(datasets, num_workers=0,
                              batch_size=10, is_train=False)
    for eg in dataloaders:
        print(eg)
        import pdb
        pdb.set_trace()
