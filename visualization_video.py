from model import PoseNet
import config as c
import torch
from tqdm import tqdm
from dataset import Dataset
import argparse
import os
from utils.vis_utils import tags_test
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_ordinal', default=0, type=int, metavar='N', help='Specify which graphics card to use')
    parser.add_argument('--model_filename', default="model_epoch1.pth", type=str, metavar='N', help='Filename of model to visualize')
    parser.add_argument('--sequence_folder', default=None, type=str, metavar='N', help='specific sequence folder')
    parser.add_argument('--dataset_type', default="test", type=str, metavar='N', help='test/val or train')
    args = parser.parse_args()
    return args

def main():
    config = c.__config__
    args = parse_args()

    if not os.path.exists('Video Output'):
        os.makedirs('Video Output')

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

    print("Loading dataset...")
    test_dataset = Dataset(config["dataset_path"], args.dataset_type + "/images", args.dataset_type +  "/labels/keypoints", max_num_car=config['max_num_car'],
                           max_num_light=config['max_num_light'], output_res=config['output_res'], negative_samples=True)
    print("Dataset loaded!")

    if args.sequence_folder != None:
        test_dataset.find_sequence_indices(args.sequence_folder)
        output_video_path = os.path.join("Video Output",  args.model_filename.split("/")[-1] + "_" + args.sequence_folder +'_output_video.mp4')
    else:
        output_video_path = os.path.join("Video Output", args.model_filename.split("/")[-1] + "_" + args.dataset_type + '_output_video.mp4')


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(output_video_path)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 24, (1280, 960))

    colors = [
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 128, 0),  # Olive
        (0, 128, 128)  # Teal
    ]
    # Iterate over the dataset and visualize predictions
    for i in tqdm(range(test_dataset.start_index, test_dataset.end_index)):
        img_dataset, keypoints_ref, keypoints, img_name, heatmaps = test_dataset.__getitem__(i)

        img_path = test_dataset.get_image_path(i)

        img = cv2.imread(img_path)
        img_dataset = torch.Tensor(img_dataset)
        img_dataset = img_dataset.to(device)
        img_dataset = img_dataset[None, :, :, :]
        keypoints = [keypoints]
        output = model(img_dataset)

        val_keypoints_batch = []
        nr_val_keypoints_batch = []
        for keypoint_batch in keypoints:
            val_keypoints = []
            nr_val_keypoints = [0, 0]
            for keypoint_car in keypoints[0]:
                for i, keypoint_light in enumerate(keypoint_car):
                    if (keypoint_light[2] != -1):
                        nr_val_keypoints[i] += 1
                        val_keypoints.append(keypoint_light[1])
                        val_keypoints.append(keypoint_light[0])
            val_keypoints_batch.append(val_keypoints)
            nr_val_keypoints_batch.append(nr_val_keypoints)

        batch_tags_results = tags_test(config, output)

        if len(batch_tags_results) != 0:

            batch_tags_results[0] = sorted(batch_tags_results[0], key=lambda x: x[0][1])

            for i, coordinates in enumerate(batch_tags_results[0]):
                for coordinate in coordinates:
                    if coordinate != None:
                        try:
                            cv2.circle(img, (int(coordinate[1]*4/0.4), int(coordinate[0]*4/0.53333)) ,7, colors[i], -1)
                        except NameError:
                            print(coordinate)

        video_writer.write(img)
    video_writer.release()

if __name__ == '__main__':
    main()