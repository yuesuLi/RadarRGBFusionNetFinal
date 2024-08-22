import torch
import torch.nn as nn
import torch.nn.functional as F


class RegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float().unsqueeze(2)

        loss = F.l1_loss(pred * mask, target * mask, reduction='none')
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)
        return loss

class MyRegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
    '''

    def __init__(self):
        super(MyRegLoss, self).__init__()

    def forward(self, out, target):
        '''
            Arguments:
              out, target: B x C x H x W
            '''

        mask = (target != 0.)
        mask = mask.int()  # B x C x H x W
        idx = torch.nonzero(mask)
        num_pos = mask.sum()
        # smooth_l1_loss = nn.SmoothL1Loss()

        loss = 0
        if num_pos == 0:
            return loss
        for i in range(idx.shape[0]):
            # pos_pred = out[list(idx[i])]
            # tmp_pos_loss = torch.log(pos_pred+1e-6) * torch.pow(1 - pos_pred, 2)
            tmp_loss = F.smooth_l1_loss(out[list(idx[i])], target[list(idx[i])])
            loss += tmp_loss

        return loss


class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target):
    '''
    Arguments:
      C=1
      out, target: B x C x H x W
    '''

    gt = torch.pow(1 - target, 4)
    # neg_loss = torch.log(1 - out+1e-6) * torch.pow(out, 2) * gt
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()
    # print('\ngt:{:.4f}'.format(-gt))
    # print('neg_loss:{:.4f}'.format(neg_loss))
    # print('true_pos_loss:{:.4f}'.format(- (pos_loss + neg_loss) / num_pos))

    mask = (target == 1.)
    mask = mask.int()   # B x C x H x W
    idx = torch.nonzero(mask)
    num_pos = mask.sum()

    pos_loss = 0
    for i in range(idx.shape[0]):
        pos_pred = out[list(idx[i])]
        # tmp_pos_loss = torch.log(pos_pred+1e-6) * torch.pow(1 - pos_pred, 2)
        tmp_pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        pos_loss += tmp_pos_loss
        # print('*************************pos_pred:{:.4f}*****************************'.format(pos_loss))
        # print('result:', pos_pred)

    # print('pos_loss:', pos_loss)
    # print('neg_loss:', neg_loss)

    if num_pos == 0:
      return - neg_loss
    # print('\nonly_neg_loss:{:.4f}'.format(-neg_loss))
    # print('num_pos:{:.4f}, pos_loss:{:.4f}'.format(num_pos, pos_loss))
    # print('true_pos_loss:{:.4f}'.format(- (pos_loss + neg_loss) / num_pos))
    return - (pos_loss + neg_loss) / num_pos
