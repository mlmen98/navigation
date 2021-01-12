"""
training model
"""
from keras_implementation import generator
from keras_implementation.model import model_fn
from keras_implementation.train_utils import create_callbacks, launch_tensorboard
from keras_implementation.config import *


def main():
    # training
    training_type = TrainingConfig.training_branch
    model_type = ModelConfig.backbone
    unit_model, classification_model, segmentation_model = model_fn()
    unit_model_dir = model_type + '_merge_model'
    segmentation_dir_city = model_type + '_segmentation_model_on_cityscape'
    segmentation_dir_finetune = model_type + '_segmentation_model'
    if os.path.exists(segmentation_dir_finetune):
        segmentation_dir = segmentation_dir_finetune
    else:
        segmentation_dir = segmentation_dir_city
    classification_dir = model_type + '_classification_model'
    if os.path.exists(unit_model_dir):
        print ('loading model from {}'.format(unit_model_dir))
        unit_model.load_weights(unit_model_dir, by_name=True, skip_mismatch=True)
    if os.path.exists(segmentation_dir):
        print ('loading model from {}'.format(segmentation_dir))
        segmentation_model.load_weights(segmentation_dir, by_name=True, skip_mismatch=True)
    if os.path.exists(classification_dir):
        print ('loading model from {}'.format(classification_dir))
        classification_model.load_weights(classification_dir, by_name=True, skip_mismatch=True)
    if training_type == 'unit':
        callbacks = create_callbacks(model_type + '_merge_model')
        train_generator = generator.combine_generator(folder='train', batch_size=16)
        val_generator = generator.combine_generator(folder='val', batch_size=16)
        print ('start training...')
        unit_model.fit_generator(generator=train_generator,
                             steps_per_epoch=300,
                             epochs=30,
                             verbose=1,
                            callbacks=callbacks,
                            validation_data=val_generator,
                            validation_steps=1,
                             )
        unit_model.save(model_type + '_merge_model')
    if training_type == 'segmentation':
        if TrainingConfig.segmentation_on_cityscape:
            train_generator = generator.segmentation_generator(batch_size=16)
            val_generator = generator.segmentation_generator('val', batch_size=16)
            steps_per_epoch = 1500
            save_name = model_type + '_segmentation_model_on_cityscape'
        else:
            train_generator = generator.segmentation_generator_test(batch_size=16)
            val_generator = generator.segmentation_generator_test(batch_size=16)
            steps_per_epoch = 100
            save_name = model_type + '_segmentation_model'
        callbacks = create_callbacks(save_name)
        print('start training...')
        segmentation_model.fit_generator(generator=train_generator,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=10,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=val_generator,
                                 validation_steps=1,
                                 workers=3,
                                 )
        segmentation_model.save(save_name)
    if training_type == 'classification':
        train_generator = generator.classification_generator(batch_size=16)
        val_generator = generator.classification_generator('val', batch_size=16)
        print('start training...')
        callbacks = create_callbacks(model_type + '_classification_model')
        classification_model.fit_generator(generator=train_generator,
                                 steps_per_epoch=100,
                                 epochs=10,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=val_generator,
                                 validation_steps=1,
                                 workers=3,
                                 )
        classification_model.save(model_type + '_classification_model')


if __name__ == '__main__':
    main()