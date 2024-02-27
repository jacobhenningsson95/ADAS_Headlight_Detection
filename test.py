import config as c
from torch.cuda.amp import autocast, GradScaler
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from utils.eval_method import gaussian_method
import torch.backends.cudnn as cudnn
from model import PoseNet
from tqdm import tqdm
from dataset import Dataset
import argparse
import warnings
import os
import matplotlib
from utils.vis_utils import initialize_visualizations, non_max, tags_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_ordinal', default=0, type=int, metavar='N', help='Specify which graphics card to use')
    parser.add_argument('--cuda', default=True, type=int, metavar='N', help='Specify if cuda')
    parser.add_argument('--visualizations', default=False, type=bool, metavar='N',
                        help='Enables visualisations in the testing loop')
    parser.add_argument('--model_filename', default="model_epoch1.pth", type=str, metavar='N',
                        help='model path')
    parser.add_argument('--eval_method', default="Gaussian", type=str, metavar='N',
                        help='eval_method')
    parser.add_argument('--Gaussian_scalar', default=[16*16, 32*32, 64*64, 128*128], type=list, metavar='N',
                        help='eval_method')
    args = parser.parse_args()
    return args


def data_gen(config):

    test_dataset = Dataset(config["dataset_path"], "test/images", "test/labels/keypoints",
                           size=config['input_res'],
                           max_num_car=config['max_num_car'],
                           max_num_light=config['max_num_light'],
                           output_res=config['output_res'],
                           negative_samples=True, day_samples=False, test=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True)

    return test_loader


def visualize_iteration_test(output, img, keypoints, config, plot_dict):
    sum_output = torch.sum(output, dim=1)
    det = sum_output[:, :config['max_num_light'], :, :]  # torch.Size([16, 5, 128, 128])
    tag = sum_output[:, config['max_num_light']:, :, :]
    val_keypoints_batch = []
    nr_val_keypoints_batch = []
    for keypoint_batch in keypoints:
        val_keypoints = []
        nr_val_keypoints = [0, 0]
        for keypoint_car in keypoints[0]:
            for i, keypoint_light in enumerate(keypoint_car):
                if keypoint_light[2] != -1:
                    nr_val_keypoints[i] += 1
                    val_keypoints.append(keypoint_light[1])
                    val_keypoints.append(keypoint_light[0])
        val_keypoints_batch.append(val_keypoints)
        nr_val_keypoints_batch.append(nr_val_keypoints)
    det_keypoints = non_max(det=det, nr_keypoint=nr_val_keypoints_batch[0])

    batch_tags_results = tags_test(config, output)

    plot_dict["img_plot"].set_data(((img[0] * 255).cpu().detach().numpy()))
    plot_dict["img"].set_data(((img[0] * 255).cpu().detach().numpy()))

    if 0 < len(val_keypoints):
        plot_dict["img_det_keypoints"].set_offsets(list(zip(det_keypoints[0][1::2], det_keypoints[0][::2])))
        plot_dict["img_val_keypoints"].set_offsets(list(zip(val_keypoints_batch[0][1::2], val_keypoints_batch[0][::2])))
    else:
        plot_dict["img_det_keypoints"].set_offsets(np.empty((0, 2)))
        plot_dict["img_val_keypoints"].set_offsets(np.empty((0, 2)))

    for i, coordinates in enumerate(batch_tags_results[0]):
        values_list = []
        if coordinates[1] is not None:
            for tup in coordinates:
                    values_list.extend(tup)
        else:
            values_list.extend(coordinates[0])

        plot_dict["clustering_det_keypoints"][i].set_offsets(list(zip(values_list[1::2], values_list[::2])))

    for i in range(len(batch_tags_results[0]), config['max_num_car']*2):
        plot_dict["clustering_det_keypoints"][i].set_offsets(np.empty((0, 2)))

    plot_dict["det_heatmap_L_plot"].set_data((det[0][1] * 255).cpu().detach().numpy().astype(np.int32))
    plot_dict["det_heatmap_R_plot"].set_data((det[0][0] * 255).cpu().detach().numpy().astype(np.int32))

    plot_dict["det_tag_L_plot"].set_data((tag[0][1] * 255).cpu().detach().numpy().astype(np.int32))
    plot_dict["det_tag_R_plot"].set_data((tag[0][0] * 255).cpu().detach().numpy().astype(np.int32))
    plot_dict["fig"].canvas.draw()
    plot_dict["fig"].canvas.flush_events()
    plt.pause(0.1)


def eval_method_func(batch_tags_results, keypoints, args, batch_size=16):
    eval_method = args.eval_method
    if not eval_method:
        print('None eval method')
        return None
    else:
        cases_counting_batch = None
        if eval_method == 'Gaussian':
            eval_value, cases_counting_batch = \
                gaussian_method(batch_tags_results, keypoints, batch_size, scalars=args.Gaussian_scalar)
            # eval_value is dic {scalar1: score1, scalar2: score2....}
        elif eval_method == 'other_method_TBA':
            eval_value = -1  # TBA
        else:
            eval_value = -1  # TBA
        return eval_value, cases_counting_batch


def output_to_keypoint(output, config):
    batch_tags_results = tags_test(config, output)
    return batch_tags_results


def test_model(config, model, device, args):
    """run model in test dataset"""
    print('Chosen', args.eval_method, 'evaluation method...')
    if args.eval_method == 'Gaussian':
        print(f'Chosen covariance scalar={args.Gaussian_scalar}; '
              f'Sigma={[math.sqrt(value) for value in args.Gaussian_scalar]}...')

    if args.visualizations:
        matplotlib.use("TkAgg")
        plot_dict = initialize_visualizations(config)
    else:
        plot_dict = None

    config['batch_size'] = config['batch_size'] * 2  # # val's batch size is 2 * batch_size
    test_loader = data_gen(config)
    model.eval()

    if args.eval_method == 'Gaussian':
        eval_matrix = {}
        for scalar in args.Gaussian_scalar:
            eval_matrix[scalar] = 0
    else:
        eval_matrix = []
    cases_counting_sum = np.array([0, 0, 0])
    with torch.no_grad():
        with autocast(enabled=config['autocast']):
            for img, keypoints, img_name in tqdm(test_loader):
                img = img.to(device)
                output = model(img)
                results = output_to_keypoint(output, config)
                eval_result, cases_counting_batch = \
                    eval_method_func(results, keypoints, args, batch_size=img.size()[0])
                if isinstance(eval_result, dict):
                    for scalar in eval_result:
                        eval_matrix[scalar] += eval_result[scalar]
                    cases_counting_sum += cases_counting_batch
                if plot_dict is not None:
                    visualize_iteration_test(output, img, keypoints, config, plot_dict)
    if isinstance(eval_matrix, dict):
        for scalar in eval_matrix:
            eval_matrix[scalar] = eval_matrix[scalar]/len(test_loader)
        return eval_matrix, cases_counting_sum
    else:
        return np.mean(np.array(eval_matrix)), None


def seed_and_settings(seed_value=46, args=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = c.__config__
    warnings.filterwarnings("ignore")
    matplotlib.use("TkAgg")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # seed everything
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_ordinal)
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    return device, config


def model_load(config, args, device):
    """create and load model"""
    model_path = "Trained model/" + args.model_filename
    checkpoint = torch.load(model_path, map_location=device)

    saved_config = checkpoint["config"]

    model = PoseNet(saved_config['nstack'], saved_config['inp_dim'], saved_config['oup_dim'], bn=saved_config['bn'],
                    increase=saved_config['increase'])
    model = model.to(device)
    model.load_state_dict(checkpoint["model"])
    config['nstack'] = saved_config['nstack']
    
    return model


def print_matrix(args, eval_value, cases_counting_sum=None):
    """
    print final matrix
    """
    if args.eval_method == 'Gaussian':
        from tabulate import tabulate
        headers = ["Sigma", "Score"]
        matrix = []
        for value in args.Gaussian_scalar:
            matrix.append((math.sqrt(value), eval_value[value]))
        table = tabulate(matrix, headers, tablefmt="grid")
        print(table)
    else:
        print(eval_value)

    if cases_counting_sum is not None:
        total_case = np.sum(cases_counting_sum)
        words_list = ['an equal', 'a greater', 'a fewer']  # eval_method.py line 48-53
        for i in range(3):
            print(f'There is a {round(100*cases_counting_sum[i]/total_case, 3)}%({cases_counting_sum[i]}/{total_case}) '
                  f'likelihood that model predicts {words_list[i]} number of vehicles compared to the ground truth.')


def main():
    args = parse_args()
    device, config = seed_and_settings(args=args)
    model = model_load(config, args, device)
    eval_value, cases_counting_sum = test_model(config, model, device, args)
    print_matrix(args, eval_value, cases_counting_sum)


if __name__ == '__main__':
    main()