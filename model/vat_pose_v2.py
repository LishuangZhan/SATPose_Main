import torch
import torch.nn as nn
from einops import rearrange
from model.block.vanilla_transformer_encoder import Transformer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce
from model.vision_transformer import huge_patch14_120_in21k, base_patch14_120_in21k, small_patch14_120_in21k, \
    small_patch14_224_in21k


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tactile_encoder = small_patch14_120_in21k(num_joints=256, has_logits=False, windowSize=27)

        self.encoder = nn.Sequential(
            nn.Conv1d(2 * args.n_joints, args.channel, kernel_size=1),
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        self.windowSize = args.frames

        self.Transformer = Transformer(args.layers, args.channel, args.d_hid, length=args.frames)
        self.Transformer_reduce = Transformer_reduce(len(args.stride_num), args.channel, args.d_hid, length=args.frames,
                                                     stride_num=args.stride_num)

        self.fcn = nn.Sequential(
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.Conv1d(args.channel, 3 * args.out_joints, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.Conv1d(args.channel, 3 * args.out_joints, kernel_size=1)
        )

    def forward(self, x, pressure):
        # # print(pressure.shape)  # (b, 1, f, h, w)
        # batch_size_p = pressure.shape[0]
        # frames_num_p = pressure.shape[2]
        # height_p = pressure.shape[3]
        # width_p = pressure.shape[4]
        # pressure = pressure.reshape(-1, 1, height_p, width_p)  # pressure.shape (batch*frames, 1, h, w)
        # tactile_feature = self.tactile_encoder(pressure)
        # # print(tactile_feature.shape)  # (batch*frames, 256)
        # tactile_feature = tactile_feature.reshape(batch_size_p, frames_num_p, -1)

        # print(pressure.shape)  # (b, 1, f, h, w)
        frames_num_p = pressure.shape[2]
        height_p = pressure.shape[3]
        width_p = pressure.shape[4]
        pressure = pressure.reshape(-1, frames_num_p, height_p, width_p)  # pressure.shape (batch, frames, h, w)
        tactile_feature = self.tactile_encoder(pressure)
        # print(tactile_feature.shape)  # (batch, 256)
        tactile_feature = tactile_feature.unsqueeze(1)
        tactile_feature = tactile_feature.repeat(1, int(self.windowSize), 1)  # (batch, frames, 256)

        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        x = self.encoder(x)

        x = x.permute(0, 2, 1).contiguous()

        # x = torch.cat((x, tactile_feature), axis=2)
        x = x + tactile_feature
        # x = tactile_feature

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


# class Model_224(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.tactile_encoder = small_patch14_224_in21k(num_joints=256, has_logits=False, windowSize=1)
#
#         self.encoder = nn.Sequential(
#             nn.Conv1d(2 * args.n_joints, args.channel, kernel_size=1),
#             nn.BatchNorm1d(args.channel, momentum=0.1),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.25)
#         )
#
#         self.windowSize = args.frames
#
#         self.Transformer = Transformer(args.layers, args.channel, args.d_hid, length=args.frames)
#         self.Transformer_reduce = Transformer_reduce(len(args.stride_num), args.channel, args.d_hid, \
#                                                      length=args.frames, stride_num=args.stride_num)
#
#         self.fcn = nn.Sequential(
#             nn.BatchNorm1d(args.channel, momentum=0.1),
#             nn.Conv1d(args.channel, 3 * args.out_joints, kernel_size=1)
#         )
#
#         self.fcn_1 = nn.Sequential(
#             nn.BatchNorm1d(args.channel, momentum=0.1),
#             nn.Conv1d(args.channel, 3 * args.out_joints, kernel_size=1)
#         )
#
#     def forward(self, x, pressure):
#         tactile_feature = self.tactile_encoder(pressure)
#         tactile_feature = tactile_feature.unsqueeze(1)
#         tactile_feature = tactile_feature.repeat(1, int(self.windowSize), 1)
#
#         B, F, J, C = x.shape
#         x = rearrange(x, 'b f j c -> b (j c) f').contiguous()
#
#         x = self.encoder(x)
#
#         x = x.permute(0, 2, 1).contiguous()
#
#         # x = torch.cat((x, tactile_feature), axis=2)
#         x = x + tactile_feature
#         # x = tactile_feature
#
#         x = self.Transformer(x)
#
#         x_VTE = x
#         x_VTE = x_VTE.permute(0, 2, 1).contiguous()
#         x_VTE = self.fcn_1(x_VTE)
#         x_VTE = rearrange(x_VTE, 'b (j c) f -> b f j c', j=J).contiguous()
#
#         x = self.Transformer_reduce(x)
#
#         x = x.permute(0, 2, 1).contiguous()
#         x = self.fcn(x)
#         x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()
#         return x, x_VTE
