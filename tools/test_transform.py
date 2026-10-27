import torch
import torchvision.transforms as transforms
import numpy as np

# 创建一个随机的 PyTorch 张量
x = torch.rand(64, 3, 256, 256)

# # 构建转换流水线
# transform = transforms.Compose([
#     transforms.Resize(224),
#     # transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# 构建转换流水线
transform_2 = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(120),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 构建转换流水线
transform_3 = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(120),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# for i in range(64):
# # 将张量转换为 PIL 图像对象
#     pil_image = transforms.functional.to_pil_image(x[i].permute(1,2,0).numpy().astype(np.uint8))




#     # 对 PIL 图像对象进行转换
#     transformed_image = transform(pil_image)

#     # 将转换后的图像转换为 PyTorch 张量
#     # transformed_tensor = transforms.functional.to_tensor(transformed_image)

#     print(np.array(transformed_image).mean())
#     break

# trans_ori = 

print(transform_2(x).shape, transform_2(x[0]).shape)
print(transform_3(x)[0], transform_2(x[0]))
