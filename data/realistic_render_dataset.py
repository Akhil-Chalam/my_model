import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np

class realistic_render_dataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode):
        if mode == "validation" or mode == "test":
            opt.load_size = 256
        else:
            opt.load_size = 286
        opt.crop_size = 256
        opt.label_nc = 150
        opt.semantic_nc = 151 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.mode = mode
        self.real, self.rendered, self.masks, self.paths = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        real = Image.open(os.path.join(self.paths[0], self.real[idx])).convert('RGB')
        rendered = Image.open(os.path.join(self.paths[1], self.rendered[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.paths[2], self.masks[idx]))
        real, rendered, mask = self.transforms(real, rendered, mask)
        return {"real": real, "rendered": rendered, "mask": mask, "name": self.real[idx]}

    def list_images(self):

        path_rendered = os.path.join(self.opt.dataroot, "trainA")
        path_mask = os.path.join(self.opt.dataroot, "masks")
        path_real = os.path.join(self.opt.dataroot, "trainB")

        rendered_list = sorted(os.listdir(path_rendered))
        mask_list = sorted(os.listdir(path_mask))
        real_list = sorted(os.listdir(path_real))

        if self.mode == "training":
            rendered_list = rendered_list[:12452]
            mask_list = mask_list[:12452]
            real_list = real_list[:12452]
        elif self.mode == "validation":
            rendered_list = rendered_list[12452:14008]
            mask_list = mask_list[12452:14008]
            real_list = real_list[12452:14008]
        elif self.mode == "test":
            rendered_list = rendered_list[14008:15565]
            mask_list = mask_list[14008:15565]
            real_list = real_list[14008:15565]
                

        for i in range(len(real_list)):
            assert os.path.splitext(rendered_list[i])[0].split("_")[2] == os.path.splitext(mask_list[i])[0].split("_")[2], '%s and %s are not matching' % (rendered_list[i], mask_list[i])
            assert os.path.splitext(rendered_list[i])[0].split("_")[3] == os.path.splitext(mask_list[i])[0].split("_")[3], '%s and %s are not matching' % (rendered_list[i], mask_list[i])
        return real_list, rendered_list, mask_list (path_real, path_rendered, path_mask)

    def transforms(self, real, rendered, mask):
        assert real.size == mask.size
        assert mask.size == rendered.size
        # resize
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        real = TR.functional.resize(real, (new_width, new_height), Image.BICUBIC)
        rendered = TR.functional.resize(rendered, (new_width, new_height), Image.BICUBIC)
        mask = TR.functional.resize(mask, (new_width, new_height), Image.NEAREST)
        # crop
        crop_x = random.randint(0, np.maximum(0, new_width -  self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        real = real.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        rendered = rendered.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        mask = mask.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_training):
            if random.random() < 0.5:
                real = TR.functional.hflip(real)
                rendered = TR.functional.hflip(rendered)
                mask = TR.functional.hflip(mask)
        # to tensor
        real = TR.functional.to_tensor(real)
        rendered = TR.functional.to_tensor(rendered)
        mask = TR.functional.to_tensor(mask)
        # normalize
        real = TR.functional.normalize(real, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        rendered = TR.functional.normalize(rendered, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return real, rendered, mask