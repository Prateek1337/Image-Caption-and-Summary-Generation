from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input

from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from PIL import Image
import h5py
import progressbar

print()
dataDir = '.'
dataType = 'train2017'

instAnnFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco_inst=COCO(instAnnFile)

capsAnnFile='{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(capsAnnFile)

img_size = (224,224)

def get_image(imgID,coco_inst=coco_inst,dataDir='.',dataType='train2017',size=img_size):
    assert len(list([imgID])) == 1
    imgName = coco_inst.loadImgs(imgID)[0]['file_name']
    I = Image.open('%s/%s/%s'%(dataDir,dataType,imgName))
    if not size is None :
        I = I.resize(size=size,resample=Image.LANCZOS)
    img = np.array(I)/255.0

    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img

def getAllIds(coco_inst=coco_inst) :
    cats = coco_inst.loadCats(coco_inst.getCatIds())
    nms = [cat['name'] for cat in cats]
    
    catIds = coco_inst.getCatIds(catNms=[nms])
    imgIds = coco_inst.getImgIds(catIds=catIds)

    return sorted(imgIds)

def get_captions(imgId):
    annIds = coco_caps.getAnnIds(imgIds=imgId);
    anns = coco_caps.loadAnns(annIds)
    
    return anns

def image_to_activations(id_list,batch_size=32):
    num_images = len(id_list)
    
    widgets=[
    '[Progress] ',
    progressbar.Bar(),' ',progressbar.Counter(format='%(value)02d/%(max_value)d'),
    ' [', progressbar.Timer(), '] '
    ]
    
    bar = progressbar.ProgressBar(max_value=num_images,widgets=widgets)
    tensor_shape = (num_images,vgg_out_shape)
    batch_shape = (batch_size,) + img_shape + (3,)
    
    activation_values = np.zeros(shape = tensor_shape, dtype = np.float16)
    batch_images      = np.zeros(shape = batch_shape, dtype = np.float16)

    start = 0
    
    while start < num_images :
        end = start + batch_size
        
        if num_images < end:
            end = num_images
        
        for i,imgId in enumerate(id_list[start:end]):
            img = get_image(imgId)
            # print("Image with id "+str(imgId)+" got")
            batch_images[i] = img
        
        curr_batch_size = end - start
        activations_batch = vgg_model.predict(batch_images[0:curr_batch_size])
        activation_values[start:end] = activations_batch[0:curr_batch_size]
        del activations_batch
        start = end
        bar.update(end)
        
    return activation_values


image_model = VGG16(include_top=True, weights='imagenet')

VGG_last_layer = image_model.get_layer('fc2')
vgg_model = Model(inputs = image_model.input, outputs = VGG_last_layer.output)

img_shape = K.int_shape(vgg_model.input)[1:3]

vgg_out_shape = K.int_shape(vgg_model.output)[1]

id_list = getAllIds()

vgg_activations = image_to_activations(id_list)

print()

activation_file = h5py.File("vgg_activations.h5","w")
activation_file.create_dataset('last_layer_activations',data=vgg_activations)
activation_file.close()

print("Activations are saved in a file named vgg_activations.h5")