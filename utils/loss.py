import torch
import torch
from torch import nn
from torch.autograd import Function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        """

        Get loss value for detection headmap

        :param pred: predicted heatmap
        :param gt: ground truth heatmap
        :return: value representing difference
        """
        assert pred.size() == gt.size()
        l = ((pred - gt) ** 2).expand_as(pred)
        l = l.sum(dim=3).sum(dim=2).sum(dim=1)
        return l

def make_input(t, device,requires_grad=False, need_cuda = True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    if need_cuda:
        inp = inp.to(device)
    return inp

def singleTagLoss(pred_tag, keypoints):
    """
    associative embedding loss for one image
    """
    eps = 1e-6
    tags = []
    pull = 0
    for i in keypoints:
        tmp = []
        for j in i:
            if j[1] > 0:
                tmp.append(pred_tag[j[0]])
        if len(tmp) == 0:
            continue
        tmp = torch.stack(tmp)
        tags.append(torch.mean(tmp, dim=0))
        pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

    if len(tags) == 0:
        return make_input(torch.zeros([1]).float()), make_input(torch.zeros([1]).float())

    tags = torch.stack(tags)[:, 0]

    num = tags.size()[0]
    size = (num, num, tags.size()[1])
    A = tags.unsqueeze(dim=1).expand(*size)
    B = A.permute(1, 0, 2)

    diff = A - B
    diff = torch.pow(diff, 2).sum(dim=2)[:, :, 0]
    push = torch.exp(-diff)
    push = (torch.sum(push) - num)
    return push/((num - 1) * num + eps) * 0.5, pull/(num + eps)


def tagLoss(tags, keypoints):
    """
    accumulate the tag loss for each image in the batch
    """
    pushes, pulls = [], []
    keypoints = keypoints.cpu().data.numpy()
    for i in range(tags.size()[0]):
        push, pull = singleTagLoss(tags[i], keypoints[i%len(keypoints)])
        pushes.append(push)
        pulls.append(pull)
    return torch.stack(pushes), torch.stack(pulls)


def test_tag_loss():
    t = make_input(torch.Tensor((1, 2)), requires_grad=True)
    t.register_hook(lambda x: print('t', x))
    loss = singleTagLoss(t, [[[0, 1]], [[1, 1]]])[0]
    loss.backward()


class AElossFunction(Function):
    @staticmethod
    def forward(ctx, tags, keypoints, return_tags=False):
        batch_size = tags.size()[0]
        tag_dim = tags.size()[2]
        num_people = keypoints.size()[1]

        output = torch.zeros(torch.Size((batch_size, 2)), device=device)
        mean_tags = torch.zeros(torch.Size((batch_size, num_people, tag_dim + 1)), device=device)

        # mean_tags: (batch_size x num_people x (tag_dim + 1)),
        #            keeps both mean tag and number of joints (for backprop)
        for b in range(batch_size):
            cur_people_count = 0
            # pull loss
            for p in range(num_people):
                valid_keypoints = keypoints[b, p, (keypoints[b, p, :, -1] == 1), 0]
                len_valid_kpts = len(valid_keypoints)
                if len_valid_kpts > 0:
                    valid_tags = tags[b, valid_keypoints, 0]
                    mean_tags[b, p, 1] = len_valid_kpts
                    mean_tags[b, p, 0] = torch.mean(valid_tags)
                    output[b, 1] += torch.sum(torch.square(valid_tags - mean_tags[b, p, 0])) / len_valid_kpts
                    cur_people_count += 1
            if cur_people_count == 0:
                continue
            output[b, 1] /= cur_people_count

            # push loss
            for p1 in range(cur_people_count - 1):
                for p2 in range(p1 + 1, cur_people_count):
                    output[b, 0] += torch.exp(-(torch.square(mean_tags[b, p1, 0] - mean_tags[b, p2, 0])))
            if cur_people_count > 1:
                output[b, 0] /= cur_people_count * (cur_people_count - 1) / 2
            output[b, 0] *= 0.5
        ctx.save_for_backward(tags, keypoints, mean_tags)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tags, keypoints, mean_tags = ctx.saved_tensors
        batch_size = tags.size()[0]
        tag_dim = tags.size()[2]
        num_people = keypoints.size()[1]

        grad_input = torch.zeros(tags.size(), device=device)
        mean_grad = torch.zeros(num_people, device=device)

        for b in range(batch_size):
            cur_people_count = torch.sum(mean_tags[b, :, 1] != 0)
            if cur_people_count == 0:
                continue

                mean_grad.fill_(0)
                if cur_people_count > 1:
                    factor = 0.5 * grad_output[b, 0] / (cur_people_count * (cur_people_count - 1) / 2)
                    for p1 in range(cur_people_count - 1):
                        for p2 in range(p1 + 1, cur_people_count):
                            diff = mean_tags[b, p1, 0] - mean_tags[b, p2, 0]
                            grad = 2 * factor * (-torch.exp(-torch.square(diff))) * diff
                            mean_grad[p1] += grad
                            mean_grad[p2] -= grad
                for p in range(num_people):
                    #Extremely slow line
                    valid_keypoints = keypoints[b, p, (keypoints[b, p, :, -1] == 1), 0]
                    len_valid_kpts = len(valid_keypoints)
                    if len_valid_kpts > 0:
                        valid_tags = tags[b, valid_keypoints, 0]
                        factor = 1/tag_dim/mean_tags[b, p, 1]/cur_people_count * grad_output[b, 1]
                        grad = 2 * (valid_tags - mean_tags[b, p, 0]) * factor
                        mean_grad[p] -= torch.sum(grad)
                        grad_input[b, valid_keypoints, 0] += grad
                        grad_input[b, valid_keypoints, 0] += mean_grad[p] / mean_tags[b, p, 1]

            # use the following two lines if you need to debug the backward loss
            # import pdb
            # pdb.set_trace()
            return grad_input, torch.zeros(keypoints.size(), device=device)


class AEloss(nn.Module):
    @staticmethod
    def forward(inp, input1):
        output = AElossFunction.apply(inp, input1)
        return output


class calc_loss(torch.nn.Module):
    """
    combine HeatmapLoss and push and pull loss
    """
    def __init__(self, config):
        super(calc_loss, self).__init__()
        self.myAEloss = AEloss()
        self.heatmapLoss = HeatmapLoss()
        self.nstack = config['nstack']
        self.max_num_light = config['max_num_light']

    def forward(self, preds, keypoints=None, heatmaps=None):
        dets = preds[:, :, :self.max_num_light]
        tags = preds[:, :, self.max_num_light:self.max_num_light*2]
        keypoints = keypoints.long()
        batchsize = tags.size()[0]
        tag_loss = []
        for i in range(self.nstack):
            tag = tags[:, i].contiguous().view(batchsize, -1, 1)
            tag_loss.append(self.myAEloss(tag, keypoints))
        tag_loss = torch.stack(tag_loss, dim=1)

        detection_loss = []
        for i in range(self.nstack):
            detection_loss.append(self.heatmapLoss(dets[:, i], heatmaps))
        detection_loss = torch.stack(detection_loss, dim=1)
        return tag_loss[:, :, 0], tag_loss[:, :, 1], detection_loss


if __name__ == '__main__':
    test_tag_loss()
