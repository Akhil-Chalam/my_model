import torch



def get_dataloaders(opt):
    dataset_name   = opt.datamode+"_dataset"
    file = __import__("dataloaders."+dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, mode = "training")
    dataset_val   = file.__dict__[dataset_name].__dict__[dataset_name](opt, mode = "validation")
    dataset_test   = file.__dict__[dataset_name].__dict__[dataset_name](opt, mode = "test")
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True, num_workers = opt.num_workers)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size, shuffle = False, drop_last=False, num_workers = opt.num_workers)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = opt.batch_size, shuffle = False, drop_last=False, num_workers = opt.num_workers)

    return dataloader_train, dataloader_val, dataloader_test