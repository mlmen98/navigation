class ModelConfig(object):
    data_dir = None
    # debug = 'store_true'
    debug = False
    interest_label = [6, 7, 8, 9, 10] # road/flat/sidewalk etc.
    seg_num_classes = 2 # foreground/background
    classification_categories = ['move_forward', 'turn_left', 'turn_right', 'turn_around', 'target_found']
    classification_num_classes = len(classification_categories)
    height = 512
    width = 512
    depth = 3
    min_scale = 0.5
    max_scale = 2.0
    ignore_label = 255
    bn_decay = 0.9997
    num_image = {
        'train': 3985,
        'validation': 1500,
    }


class TrainingConfig(object):
    clean_model_dir = 0
    train_epochs = 100
    epochs_per_eval = 1
    tensorboard_images_max_outputs = 6
    batch_size = 8
    learning_rate_policy = 'poly' # choices=['poly', 'piecewise']
    max_iter = 30000
    base_architecture = 'resnet_v2_50' # choices=['resnet_v2_50', 'resnet_v2_101']
    initial_learning_rate = 5e-2
    end_learning_rate = 1e-6
    initial_global_step = 0
    power = 0.9
    momentum = 0.9
    weight_decay = 5e-4 # regulization
    freeze_batch_norm = False
    pre_trained_model = './resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'
    model_dir = './new'




