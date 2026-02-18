import torch
import torchvision.transforms as transforms


class Crop:
    def __init__(self, global_crop_size, local_crop_size=0):
        self._global_crop_size = global_crop_size
        self._local_crop_size = local_crop_size

        self.global_crop = transforms.RandomCrop(global_crop_size)
        self.local_crop = transforms.RandomCrop(local_crop_size)

    def single_crop(self, x, crop):
        return crop(x)

    def random_crop(self, x, count, crop):
        crops = []
        for _ in range(count):
            crops.append(self.single_crop(x, crop))

        crops = torch.stack(crops, dim=1)
        return crops

    def __call__(self, x, num_global, num_local):
        global_views = self.random_crop(x, num_global, self.global_crop)
        local_views = self.random_crop(x, num_local, self.local_crop)

        return global_views, local_views
