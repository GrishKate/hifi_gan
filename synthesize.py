import warnings

import hydra
import torch
from hydra.utils import instantiate
import os
import torchaudio
import numpy as np
import torch
from omegaconf import OmegaConf
from pathlib import Path

from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils import MelSpectrogram, MelSpectrogramConfig
from src.datasets.data_utils import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)


def text_to_mel(dir, device):
    data_dir = Path(dir).absolute() / "transcriptions"
    tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
    tacotron2 = tacotron2.to(device)
    tacotron2.eval()
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
    if not os.path.exists(str(data_dir / 'mel')):
        os.mkdir(str(data_dir / 'mel'))
    if data_dir.exists():
        for trans in os.listdir(str(data_dir)):
            if trans.endswith((".txt")):
                id = trans[:-4]
                with open(os.path.join(data_dir, trans), 'r') as f:
                    text = f.read()
                sequences, lengths = utils.prepare_input_sequence([text])
                lst = []
                n = lengths.item()
                M = 100
                K = n//M
                if n%M>0:
                    K+=1
                with torch.no_grad():
                    for k in range(K):
                        s = sequences[:, M*k:M*(k+1)]
                        el = torch.tensor([s.shape[1]], device=device)
                        mel, _, _ = tacotron2.infer(s, el)
                        lst.append(mel.detach().cpu().squeeze())
                mel = np.hstack(lst)
                np.save(str(data_dir / 'mel') +'/' + id + '.npy', mel)


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    set_random_seed(config.inferencer.seed)
    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device
    if not config.datasets.test.resynthesize:
        text_to_mel(config.datasets.test.data_dir, device)

    if config.inferencer.log:
        project_config = OmegaConf.to_container(config)
        #logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, None, project_config)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)
    dataloader = dataloaders['test']

    # build model architecture, then print to console
    model = instantiate(config.generator).to(device)
    checkpoint = torch.load(config.inferencer.from_pretrained, device)
    model.load_state_dict(checkpoint["gen_state_dict"])

    # save_path for model predictions
    save_path = ROOT_PATH / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    mel_config = MelSpectrogramConfig()
    make_mel = MelSpectrogram(mel_config)
    sr = mel_config.sr
    cnt = 0

    for batch in dataloader:
        batch['real_mel'] = batch['real_mel'].to(device)
        audios = model(batch['real_mel'])['fake'].detach().cpu()
        for i, (audio, id) in enumerate(zip(audios, batch['id'])):
            output_f = os.path.join(save_path, id + '.wav')
            torchaudio.save(output_f, audio, sr)
            if config.inferencer.log:
                writer.set_step(cnt)
                cnt += 1
                if 'real' in batch.keys():
                    writer.add_audio("real_audio", batch['real'][i], sr)
                writer.add_audio("fake_audio", audio, sr)
                fake_mel = make_mel(audio).squeeze()
                writer.add_image("fake_mel", fake_mel)
                writer.add_image("real_mel", batch['real_mel'][i])
                writer.add_text("text", batch['text'][i])
                writer.add_text("id", id)


if __name__ == "__main__":
    main()
