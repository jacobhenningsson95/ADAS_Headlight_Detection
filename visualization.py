from model import PoseNet
import config as c
import torch
from dataset import Dataset
import argparse
import os
from utils.vis_utils import initialize_visualizations, visualize_iteration
from utils.loss import calc_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_ordinal', default=0, type=int, metavar='N', help='Specify which graphics card to use')
    parser.add_argument('--model_filename', default="model_epoch1.pth", type=str, metavar='N', help='Filename of model to visualize')
    parser.add_argument('--image_name', default="002580.png", type=str, metavar='N', help='filename of image to visualize')
    parser.add_argument('--dataset_type', default="test", type=str, metavar='N', help='test/val or train')
    args = parser.parse_args()
    return args


def main():
    config = c.__config__
    args = parse_args()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_ordinal)

    model_path = "Trained model/" + args.model_filename
    checkpoint = torch.load(model_path, map_location=device)

    saved_config = checkpoint["config"]

    model = PoseNet(saved_config['nstack'], saved_config['inp_dim'], saved_config['oup_dim'], bn=saved_config['bn'],
                    increase=saved_config['increase'])
    model = model.to(device)
    model.load_state_dict(checkpoint["model"])
    config['nstack'] = saved_config['nstack']

    model.eval()
    print('Model loaded successfully')

    plot_dict = initialize_visualizations(config)
    criterion = calc_loss(config)

    test_dataset = Dataset(config["dataset_path"], args.dataset_type + "/images/", args.dataset_type +  "/labels/keypoints", max_num_car=config['max_num_car'],
                           max_num_light=config['max_num_light'], output_res=config['output_res'], negative_samples=True)

    img, keypoints_ref, keypoints, img_name, heatmaps = test_dataset.generate_data_by_name(args.image_name)
    img = torch.Tensor(img)
    keypoints_ref = torch.Tensor(keypoints_ref)
    heatmaps = torch.Tensor(heatmaps)
    keypoints = [keypoints]
    img = img.to(device)
    keypoints_ref = keypoints_ref.to(device)
    heatmaps = heatmaps.to(device)
    img = img[None,:,:,:]
    keypoints_ref = keypoints_ref[None,:,:,:]
    heatmaps = heatmaps[None,:,:,:]

    output = model(img)
    result = criterion(output, keypoints_ref, heatmaps)

    losses = {i[0]: result[-len(config['loss']) + idx] * i[1] for idx, i in enumerate(config['loss'])}
    loss = 0
    for i in losses:
        loss = loss + torch.mean(losses[i])

    visualize_iteration(output, img, keypoints, heatmaps, losses, config, plot_dict)


if __name__ == '__main__':
    main()
