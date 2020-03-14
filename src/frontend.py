"""

"""

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.data_utils.DataSequence import DataSequence
from src.backend import ENET, VGG, UNET
from tensorflow.keras import callbacks

import segmentation_models as sm

class Segment(object):


    def __init__(self, backend,
                 input_size, nb_classes):

        """
        Model Factory that fetches the corresponding model based on the backend that has been defined
        and initiates the training process

        :param backend: define the backbone architecture for the training
        :param input_size: the size of the input image
        :param nb_classes: the number of classes
        """
        self.input_size = input_size
        self.nb_classes = nb_classes

        if backend == "ENET":
            self.feature_extractor = ENET(self.input_size, self.nb_classes).build()
        elif backend == "VGG":
            self.feature_extractor = VGG(self.input_size, self.nb_classes).build()
        elif backend == "UNET":
            self.feature_extractor = UNET(self.input_size, self.nb_classes).build()
        else:
            raise ValueError('No such arch!... Please check the backend in config file')

    def train(self, train_configs, model_configs):

        """
         Train the model based on the training configurations
        :param train_configs: Configuration for the training
        """
        optimizer = Adam(train_configs["learning_rate"])

        # Data sequence for training
        train_gen = DataSequence( train_configs["train_images"] , train_configs["train_annotations"],  train_configs["train_batch_size"],  model_configs['classes'] , model_configs['im_height'] , model_configs['im_width'] , model_configs['out_height'] , model_configs['out_width'], do_augment=True)

        # Data sequence for validation
        val_gen = DataSequence( train_configs["val_images"] , train_configs["val_annotations"],  train_configs["val_batch_size"],  model_configs['classes'] , model_configs['im_height'] , model_configs['im_width'] , model_configs['out_height'] , model_configs['out_width'], do_augment=True)

        # Configure the model for training
        # https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
        # if model_configs['classes'] == 2:
        #     loss_function = 'binary_crossentropy'
        # else:
        #     loss_function = 'categorical_crossentropy'

        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() if model_configs['classes'] == 2 else sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        

        iou = sm.metrics.IOUScore(threshold=0.5)
        fscore = sm.metrics.FScore(threshold=0.5)
        metrics = [iou, fscore]

    
        self.feature_extractor.compile(optimizer=optimizer, loss=total_loss,
            metrics=metrics)

        # Load pretrained model
        if train_configs["load_pretrained_weights"]:
            print("Loading pretrained weights: " + train_configs["pretrained_weights_path"])
            self.feature_extractor.load_weights(train_configs["pretrained_weights_path"])

        # Define the callbacks for training
        tb = TensorBoard(log_dir=train_configs["logs_dir"], write_graph=True)
        mc = ModelCheckpoint(mode="max", filepath=str(train_configs["save_model_name"]).replace(".h5", "") + ".{epoch:03d}.h5", monitor="f1-score",
            save_best_only=True,
            save_weights_only=False, verbose=2)
        callback = [tb, mc]


        # Train the model on data generated batch-by-batch by the DataSequence generator
        self.feature_extractor.fit_generator(train_gen,
            steps_per_epoch=len(train_gen),
            validation_data=val_gen, 
            validation_steps=len(val_gen),
            epochs=train_configs["nb_epochs"],
            verbose=1,
            shuffle=True, callbacks=callback,
            workers=6,
            max_queue_size=24,
            use_multiprocessing=True
            )
