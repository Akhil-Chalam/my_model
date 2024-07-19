
import models.models as models
import data.dataloader as dataloaders
import utils.utils as utils
import config


#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
_, _, dataloader_test = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
model = models.model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

#--- iterate over validation set ---#
for i, data_i in enumerate(dataloader_test):
    # _, label = models.preprocess_input(opt, data_i)
    rendered = data_i['rendered'].cuda() if opt.gpu_ids != "-1" else data_i['rendered']
    mask = data_i['real'].cuda() if opt.gpu_ids != "-1" else data_i['mask']
    generated = model(None, rendered, "generate", None)
    image_saver(rendered, generated, data_i["name"])