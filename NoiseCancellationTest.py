import os
import torch
import argparse
from AudioReader import write_wav, read_wav
from DINC_Net import DINC_Net
from utils import get_logger
import scipy.signal as ss


class Cancellation():
    def __init__(self, mix_path, noise_path, model, gpu_id, frame_length=4096, savgol_length=100):
        super(Cancellation, self).__init__()
        self.mix = read_wav(mix_path)
        self.noise = read_wav(noise_path)
        self.frame_length = frame_length
        net = DINC_Net()
        dicts = torch.load(model, map_location='cpu')
        net.load_state_dict(dicts["model_state_dict"])
        self.logger = get_logger(__name__)
        self.net = net.cuda()
        self.device = torch.device('cuda:{}'.format(
            gpu_id[0]) if len(gpu_id) > 0 else 'cpu')
        self.gpu_id = tuple(gpu_id)
        self.savgol_length = savgol_length

    def inference(self, save_path):
        with torch.no_grad():
            frame_length = self.frame_length
            savgol_length = self.savgol_length   # Length of interval for interpolation repair
            for index in range(int(self.mix.shape[1] / frame_length)):
                split_mix = self.mix[0, frame_length * index:frame_length * (index + 1)]
                split_noise = self.noise[0, frame_length * index:frame_length * (index + 1)]
                split_mix = torch.unsqueeze(split_mix, dim=0)
                split_noise = torch.unsqueeze(split_noise, dim=0)

                mix = split_mix.to(self.device)
                noise = split_noise.to(self.device)
                norm = torch.norm(mix, float('inf')).cpu()
                egs = [mix, noise]

                if len(self.gpu_id) != 0:
                    ests = self.net(egs)
                    split_restore = torch.squeeze(ests.detach().cpu())
                else:
                    ests = self.net(egs)
                    split_restore = torch.squeeze(ests.detach())

                split_mix = torch.squeeze(split_mix)
                split_noise = torch.squeeze(split_noise)
                split_restore = split_restore * norm / torch.max(torch.abs(split_restore))
                if index == 0:
                    restore = split_restore
                    speech = split_mix
                    noise0 = split_noise
                    continue

                speech = torch.cat((speech, split_mix), dim=0)
                noise0 = torch.cat((noise0, split_noise), dim=0)
                restore = torch.cat((restore, split_restore), dim=0)

                # Here is the interpolation repair, you can also use other interpolation methods.
                if index != 0:
                    y = restore[index * frame_length - savgol_length:index*frame_length + savgol_length]
                    tmp_smooth = ss.savgol_filter(y, 23, 3)
                    restore[index * frame_length - savgol_length:index * frame_length + savgol_length] = torch.from_numpy(tmp_smooth)

            restore = torch.unsqueeze(restore, dim=0)
            os.makedirs(save_path, exist_ok=True)
            # Save the output of the model
            filename_restore = save_path + '/' + 'restore.wav'
            write_wav(filename_restore, restore, 8000)
        self.logger.info("Compute over {:d} utterances".format(len(self.mix)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-mix_path', type=str, default="./normal_NLMS/mix.wav", help='Path to mix file.')
    parser.add_argument(
        '-noise_path', type=str, default='./normal_NLMS/noise.wav', help='Path to noise file.')
    parser.add_argument(
        '-model', type=str, default='./model/best_model.pt', help="Path to model file.")
    parser.add_argument(
        '-gpu_id', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='./result', help='save result path')
    args = parser.parse_args()
    gpu_id = [int(i) for i in args.gpu_id.split(',')]
    cancellation = Cancellation(args.mix_path, args.noise_path, args.model, gpu_id)
    cancellation.inference(args.save_path)


if __name__ == "__main__":
    main()