
def train_model(train_steps, val_steps, out_dir):
    """
    Trains a neural network model using the provided training and validation steps and saves the output.

    The function compiles the model with Adam optimizer and Crossentropy loss function,
    defines learning rate schedules, prepares callbacks, and initiates the training process.
    The model's performance is monitored using accuracy as the metric. Outputs, including
    TensorBoard logs and weights, are saved to the specified directory.

    Args:
        train_steps (int): The number of training steps per epoch.
        val_steps (int): The number of validation steps to perform.
        out_dir (str): The directory where outputs like logs and model weights will be saved.
    """

    global IMG_SIZE
    name = "example-net-{}-{}-".format(IMG_SIZE, args.arch) + time.strftime(
        "%m-%d-%Y.%H:%M:%S", time.gmtime(time.time())
    )
    lr = args.lr
    with strategy.scope():

        # Compile model, and train
        def lr_scheduler(epoch, lr):
            return lr * 0.96 ** ((epoch) / 10)

        model.compile(
            tf.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.Crossentropy(),
            metrics=["accuracy"],
        )
        model.summary()
        if args.nni:
            print(">>> in args.nni")
            callbacks = [
                tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                NNICallback(),
            ]
        elif args.out:
            callbacks = [
                tf.keras.callbacks.TensorBoard(
                    log_dir="logs/{}".format(name), histogram_freq=1
                ),
                tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(
                    out_dir + "/weights_" + "{epoch}.h5",
                    verbose=1,
                    save_weights_only=True,
                ),
            ]
        else:
            callbacks = [
                # add custom callback
                tf.keras.callbacks.TensorBoard(
                    log_dir="logs/{}".format(name), histogram_freq=1
                ),
                tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
            ]
        if args.nni:
            model.fit(
                train_data_gen,
                steps_per_epoch=train_steps,
                epochs=15,
                callbacks=callbacks,
                class_weight=None,
            )
        else:
            model.fit(
                train_data_gen,
                steps_per_epoch=train_steps,
                epochs=args.epochs,
                callbacks=callbacks,
                validation_data=val_data_gen,
                validation_steps=val_steps,
                validation_freq=1,
                class_weight=None,
            )


class NNICallback(tf.keras.callbacks.Callback):
    """
    A custom Keras Callback that reports model performance metrics to the NNI (Neural Network Intelligence) toolkit.

    It is used to report the validation accuracy to the NNI framework at the end of each epoch,
    as well as at the end of training to facilitate hyperparameter tuning and model optimization.
    
    Methods:
        on_epoch_end: Reports the intermediate validation accuracy to NNI after each epoch.
        on_train_end: Reports the final validation accuracy to NNI after training is finished.
    """
    def on_epoch_end(self, epoch, logs=None):
        val_steps = args.val_steps // args.batch_size
        evaluation = model.evaluate(x=val_data_gen, steps=val_steps)
        nni.report_intermediate_result(evaluation[1])

    def on_train_end(self, logs=None):
        val_steps = args.val_steps // args.batch_size
        evaluation = model.evaluate(x=val_data_gen, steps=val_steps)
        nni.report_final_result(evaluation[1])

