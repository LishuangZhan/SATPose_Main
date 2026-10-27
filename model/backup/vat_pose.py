import torch
import torch.nn as nn
from einops import rearrange
from model.block.vanilla_transformer_encoder import Transformer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce
from model.vision_transformer import huge_patch14_120_in21k, base_patch14_120_in21k, small_patch14_120_in21k


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tactile_encoder = small_patch14_120_in21k(num_joints=18, has_logits=False, windowSize=1)

        self.encoder = nn.Sequential(
            nn.Conv1d(3 * args.n_joints, args.channel, kernel_size=1),
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        self.windowSize = args.frames

        self.Transformer = Transformer(args.layers, args.channel, args.d_hid, length=args.frames)
        self.Transformer_reduce = Transformer_reduce(len(args.stride_num), args.channel, args.d_hid, \
                                                     length=args.frames, stride_num=args.stride_num)

        self.fcn = nn.Sequential(
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.Conv1d(args.channel, 3 * args.out_joints, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.Conv1d(args.channel, 3 * args.out_joints, kernel_size=1)
        )

    def forward(self, x, pressure):
        tactile_feature = self.tactile_encoder(pressure)  # (batch, 22, 1)
        shape = tactile_feature.shape
        tactile_feature = tactile_feature.reshape(shape[0], 1, shape[1], 1)
        tactile_feature = tactile_feature.repeat(1, int(self.windowSize), 1, 1)  # (batch, frames, 22, 1)
        x = torch.cat((x, tactile_feature), axis=3)

        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        x = self.encoder(x)
        x = x.permute(0, 2, 1).contiguous()

        x = self.Transformer(x)

        x_VTE = x
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.fcn_1(x_VTE)
        x_VTE = rearrange(x_VTE, 'b (j c) f -> b f j c', j=J).contiguous()

        x = self.Transformer_reduce(x)

        x = x.permute(0, 2, 1).contiguous()
        x = self.fcn(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x, x_VTE
