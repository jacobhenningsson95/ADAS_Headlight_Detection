import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


# nonmaximum suppression
def non_max(det, nr_keypoint):
    """
    det: predicted map
    nr_keypoint : number of jeypoints
    """
    # threshold = 0
    batch_sizes, num_channels, height, width = det.size()
    batch_sizes_keypoints = []
    for batch_size in range(batch_sizes):
        prediction_coordinates = []
        for channel in range(num_channels):
            heatmap = det[batch_size, channel, :, :]
            heatmap = torch.squeeze(heatmap)  # torch.Size([output_res, output_res])
            topk_values, topk_indices = torch.topk(heatmap.view(-1), nr_keypoint[channel])
            topk_row_indices = topk_indices // heatmap.shape[1]
            topk_col_indices = topk_indices % heatmap.shape[1]
            prediction_coordinates.append(torch.stack((topk_row_indices, topk_col_indices), dim=1))  # get the top 8 coordinates

        img_pred_keypoints = []
        for prediction in prediction_coordinates:
            for coordinate in prediction:
                img_pred_keypoints.append(int(coordinate[0]*4))
                img_pred_keypoints.append(int(coordinate[1]*4))
        img_pred_keypoints = torch.tensor(img_pred_keypoints, dtype=torch.float32)
        batch_sizes_keypoints.append(img_pred_keypoints)
    return batch_sizes_keypoints  # dic  len(dic)=batch_sizes


class HeadLightCoordinate:
    """
    Class to keep track of data during clustering
    """

    def __init__(self, coordinates, tag_value):
        self.coordinates = coordinates # Coordinates of headlight keypoint
        self.tag_value = tag_value # Associated tag value used for clustering
        self.pair = None # Pair of the opposite class
        self.distance_to_pair = np.inf # Distance to the pair
        self.distances = None # Distance all the keypoints of the opposite class.

    def set_distances(self, distances):
        self.distances = distances


def tags_test(config, output):
    """
    Takes the raw output of the model and outputs clustered keypoints.

    :param config: configuration dict
    :param output: raw output of the model
    :return: List of clustered keypoints
    """
    sum_output = torch.sum(output, dim=1)  # torch.Size([batch_size, n_stack, output_dim, 128, 128])
    batch_tags_results = []
    for i in range(output.size()[0]):  # for each imgs
        dets = sum_output[i, :config['max_num_light'], :, :]
        tags = sum_output[i, config['max_num_light']:, :, :]

        channel_tags = []
        channel_coordinates = []
        for channel in range(config['max_num_light']):  # assume first channel is left light, second is right light
            heatmap = dets[channel, :, :]
            heatmap = torch.squeeze(heatmap)
            # Apply dynamic threshold
            if config['min_threshold']*config['nstack'] <= torch.max(heatmap) and torch.max(heatmap) < config['threshold']*config['nstack']:
                threshold = torch.max(heatmap) - 0.01 * torch.std(heatmap)
            else:
                threshold = config['threshold']*config['nstack']

            mask = heatmap > threshold # Boolean mask where heatmap values exceed threshold
            indices = torch.nonzero(mask, as_tuple=False)
            topk_coordinates = indices.tolist()  # Filtered coordinates based on threshold
            tag = tags[channel, :, :]
            tag = torch.squeeze(tag)

            tag_value = []
            for coordinate in topk_coordinates:
                tag_value.append(tag[coordinate[0], coordinate[1]])
            channel_tags.append(tag_value)  # save coordinates and corresponding values in tags matrix
            channel_coordinates.append(topk_coordinates)

        # left tags and right tags
        left_light_tags = channel_tags[0]
        right_light_tags = channel_tags[1]

        left_light_coordinates = channel_coordinates[0]
        right_light_coordinates = channel_coordinates[1]

        left_headlights = []
        right_headlights = []

        #Sort the coordinate lists
        for idx, left_light_coordinate in enumerate(left_light_coordinates):
            left_headlights.append(HeadLightCoordinate(coordinates=left_light_coordinate, tag_value=left_light_tags[idx]))

        for idx, right_light_coordinate in enumerate(right_light_coordinates):
            right_headlights.append(
                HeadLightCoordinate(coordinates=right_light_coordinate, tag_value=right_light_tags[idx]))


        # Find which class has the most headlights
        longer_headlights_lists = max([right_headlights, left_headlights], key=len)
        shorter_headlights_lists = min([left_headlights, right_headlights], key=len)

        # Calculate distances between the classes
        for longer_headlight in longer_headlights_lists:
            distances = []
            for shorter_headlight in shorter_headlights_lists:
                distances.append(abs(longer_headlight.tag_value - shorter_headlight.tag_value))
            distances.sort()
            longer_headlight.set_distances(distances)

        # Cluster the keypoints based on distance
        while True:
            continue_loop = False
            for longer_headlight in longer_headlights_lists:
                for idx, distance in enumerate(longer_headlight.distances):

                    if longer_headlight.pair == None and distance < shorter_headlights_lists[idx].distance_to_pair:

                        if shorter_headlights_lists[idx].pair != None:
                            shorter_headlights_lists[idx].pair.distance_to_pair = np.inf
                            shorter_headlights_lists[idx].pair.pair = None

                        longer_headlight.pair = shorter_headlights_lists[idx]
                        longer_headlight.distance_to_pair = distance

                        shorter_headlights_lists[idx].pair = longer_headlight
                        shorter_headlights_lists[idx].distance_to_pair = distance
                        continue_loop = True

                        break

            if not continue_loop:
                break

        # Format the output before returning
        clusterings = []
        for longer_headlight in longer_headlights_lists:
            if longer_headlight.pair != None:
                clusterings.append([longer_headlight.coordinates, longer_headlight.pair.coordinates])
            else:
                clusterings.append([longer_headlight.coordinates, None]) # For all the headlights that don't have a pair

        batch_tags_results.append(clusterings)

    return batch_tags_results


def visualize_iteration(output, img, keypoints, heatmaps, loss, config, plot_dict):
    """
    Uses the initialized plot dict to visualize output from the model and ground truths.

    :param output: Raw output of the model
    :param img: Original image
    :param keypoints: True keypoint
    :param heatmaps: True heatmaps
    :param loss: Loss value for the current output
    :param config: Configuration for model
    :param plot_dict: Initialized plot dict
    """
    sum_output = torch.sum(output, dim=1)
    det = sum_output[:, :config['max_num_light'], :, :]  # torch.Size([16, 5, 128, 128])
    tag = sum_output[:, config['max_num_light']:, :, :]

    # Plot true keypoints
    val_keypoints = []
    nr_val_keypoints = [0,0]
    for keypoint_car in keypoints[0]:
        for i, keypoint_light in enumerate(keypoint_car):
            if (keypoint_light[2] != -1):
                nr_val_keypoints[i] += 1
                val_keypoints.append(keypoint_light[1].item())
                val_keypoints.append(keypoint_light[0].item())

    plot_dict["img_plot"].set_data(((img[0] * 255).cpu().detach().numpy()))
    plot_dict["img"].set_data(((img[0] * 255).cpu().detach().numpy()))


    batch_tags_results = tags_test(config, output[0:1, :, :, :, :])

    keypoints_values_list = []
    if len(batch_tags_results[0]) != 0:
        for i, coordinates in enumerate(batch_tags_results[0]):
            if i <= config['max_num_car']:
                values_list = []
                if coordinates[1] is not None:
                    for tup in coordinates:
                        values_list.extend(tup)
                        keypoints_values_list.append(tup[0]*4)
                        keypoints_values_list.append(tup[1]*4)
                else:
                    values_list.extend(coordinates[0])
                    keypoints_values_list.append(coordinates[0][0]*4)
                    keypoints_values_list.append(coordinates[0][1]*4)

                plot_dict["clustering_det_keypoints"][i].set_offsets(list(zip(values_list[1::2], values_list[::2])))
        for i in range(len(batch_tags_results[0]), config['max_num_car'] * 2):
            plot_dict["clustering_det_keypoints"][i].set_offsets(np.empty((0, 2)))

        plot_dict["img_det_keypoints"].set_offsets(list(zip(keypoints_values_list[1::2], keypoints_values_list[::2])))

    else:
        plot_dict["img_det_keypoints"].set_offsets(np.empty((0, 2)))
        for i in range(0, config['max_num_car'] * 2):
            plot_dict["clustering_det_keypoints"][i].set_offsets(np.empty((0, 2)))

    if len(val_keypoints) != 0:
        plot_dict["img_val_keypoints"].set_offsets(list(zip(val_keypoints[1::2], val_keypoints[::2])))
    else:
        plot_dict["img_val_keypoints"].set_offsets(np.empty((0, 2)))

    plot_dict["det_heatmap_L_plot"].set_data((det[0][1] * 255).cpu().detach().numpy().astype(np.int32))
    plot_dict["det_heatmap_R_plot"].set_data((det[0][0] * 255).cpu().detach().numpy().astype(np.int32))

    plot_dict["det_tag_L_plot"].set_data((tag[0][1] * 255).cpu().detach().numpy().astype(np.int32))
    plot_dict["det_tag_R_plot"].set_data((tag[0][0] * 255).cpu().detach().numpy().astype(np.int32))
    plot_dict["val_heatmap_L_plot"].set_data((heatmaps[0][1] * 255).cpu().detach().numpy())
    plot_dict["val_heatmap_R_plot"].set_data((heatmaps[0][0] * 255).cpu().detach().numpy())

    plot_dict["fig"].suptitle("Loss: " + str(torch.mean(loss["detection_loss"][0]).cpu().detach().numpy()) + ", Nr true keypoints: " + str(int(len(val_keypoints)/2)) + ", Nr predicted keypoints: " + str(int(len(keypoints_values_list)/2)))
    plot_dict["fig"].canvas.draw()
    plot_dict["fig"].canvas.flush_events()
    plt.pause(0.1)


def initialize_visualizations(config):
    """
    Initializes a dict for visualization.

    :param config: Configuration for the current model
    :return: Initialized visualization dict.
    """
    plt.ion()

    plot_dict = {}

    fig, axes = plt.subplots(2, 5, figsize=(12, 4))
    plot_dict["fig"] = fig
    plt.subplots_adjust(hspace=0.4)
    plt.subplots_adjust(right=0.945)

    plot_dict["img"] = axes[0][0].imshow(np.zeros((512, 512)), cmap='gray', vmin=0, vmax=255)
    axes[0][0].set_title("Image")

    plot_dict["img_plot"] = axes[1][0].imshow(np.zeros((512, 512)), cmap='gray', vmin=0, vmax=255)
    plot_dict["img_det_keypoints"] = axes[1][0].scatter(None, None, color="blue")
    axes[1][0].set_title("Image Keypoints")

    axes[0][1].imshow(np.ones((128, 128, 3)))
    plot_dict["img_val_keypoints"] = axes[0][1].scatter(None, None, color="orange")
    axes[0][1].set_title("True Keypoints")

    plot_dict["clustering_det_keypoints"] = []

    axes[1][1].set_title("Pred Clustering")
    axes[1][1].imshow(np.ones((128, 128, 3)))
    for x in range(config['max_num_car']*2):
        plot_dict["clustering_det_keypoints"].append(axes[1][1].scatter(None, None))

    plot_dict["val_heatmap_L_plot"]  = axes[0][2].imshow(np.zeros((128, 128)), vmin=0, vmax=255)
    axes[0][2].set_title("True heatmap L")
    plot_dict["val_heatmap_R_plot"] = axes[1][2].imshow(np.zeros((128, 128)), vmin=0, vmax=255)
    axes[1][2].set_title("True heatmap R")

    plot_dict["det_heatmap_L_plot"] = axes[0][3].imshow(np.zeros((128, 128)), vmin=0, vmax=255)
    axes[0][3].set_title("Pred Heatmap L")

    plot_dict["det_heatmap_R_plot"] = axes[1][3].imshow(np.zeros((128, 128)), vmin=0, vmax=255)
    axes[1][3].set_title("Pred Heatmap R")

    plot_dict["det_tag_L_plot"] = axes[0][4].imshow(np.zeros((128, 128)), vmin=0, vmax=255)
    axes[0][4].set_title("Pred tag L")

    plot_dict["det_tag_R_plot"] = axes[1][4].imshow(np.zeros((128, 128)), vmin=0, vmax=255)
    axes[1][4].set_title("Pred tag R")

    fig.legend(handles=[plot_dict["img_det_keypoints"], plot_dict["img_val_keypoints"]], labels=['Detected keypoints', 'True keypoints'])

    return plot_dict
