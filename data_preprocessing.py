def create_tf_example(tfrecord_data):
    """
    Creates a TensorFlow SequenceExample object from the provided tfrecord data.
    
    The function iterates through tfrecord_data to encode images, labels, and filenames
    into a SequenceExample format, which is then written to a TFRecord file. The function
    continues this process until all data is processed or tfrecord_data is empty.
    
    Args:
        tfrecord_data (list): A list of tuples containing file paths to images and their 
                              corresponding labels.
    """

    while 1:
        print(len(tfrecord_data))
        if len(tfrecord_data) == 0:
            break

        tf_example = tf.train.SequenceExample()
        tf_example.context.feature["image/width"].int64_list.value.append(img_width)
        tf_example.context.feature["image/height"].int64_list.value.append(img_height)
        tf_example.context.feature["image/format"].bytes_list.value.append(
            "jpg".encode()
        )
        image_features = tf_example.feature_lists.feature_list["image/encoded"]
        label_features = tf_example.feature_lists.feature_list[
            "image/object/class/label"
        ]
        filename_features = tf_example.feature_lists.feature_list["image/filename"]

        for filename_img, label in tfrecord_data[:seq_len]:
            with tf.io.gfile.GFile(filename_img, "rb") as fid:
                encoded_image = fid.read()

            image_features.feature.add().bytes_list.value.append(encoded_image)
            label_features.feature.add().int64_list.value.append(int(label))
            filename_features.feature.add().bytes_list.value.append(
                filename_img.encode()
            )
        try:
            for _ in range(interleave):
                tfrecord_data.pop(0)
        except IndexError:
            pass
        write_tffile(tf_example)


# -------------------------------------------------------------------------------


def parse_tfrecord(size=256, data_aug=True, file_name=False):
    """
    Returns a function that parses TFRecord entries into training data.

    This function creates another function that can be used to map over a dataset. It
    decodes the JPEG images, applies resizing, and optionally applies data augmentation
    techniques such as random flips, rotations, brightness, saturation, and hue adjustments.
    
    Args:
        size (int): Target size to resize images.
        data_aug (bool): Flag to determine whether data augmentation should be applied.
        file_name (bool): Flag to determine whether file names should be returned 
                          alongside the images and labels.

    Returns:
        function: A function that takes a serialized TFRecord and returns either a tuple
                  of (image, label) or (image, label, filename) depending on the file_name flag.
    """

    def func(tfrecord):
        x = tf.io.parse_example(tfrecord, IMAGE_FEATURE_MAP)
        x_train = tf.image.decode_jpeg(x["image/example"], channels=3)
        img = tf.image.resize(x_train, (size, size))
        if data_aug:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            if tf.random.uniform([], 0, 10) < 2:
                img = tf.image.rot90(img)
            img = tf.image.random_brightness(img, max_delta=0.999)
            img = tf.image.random_saturation(img, 0.1, 2.5)
            img = tf.image.random_hue(img, 0.1)
        class_label = x["image/object/class/label"]

        # Normalize the image
        img = img / 255

        if file_name:
            x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
            filename = x["image/filename"]
            return img, class_label.values, filename.values
        else:
            return img, class_label.values
        return func


def dataset_setup(dataset_type="train", data_aug=True):
    """
    Sets up a TensorFlow dataset from TFRecord files for training or validation.

    This function creates a TFRecordDataset from files within a specific directory,
    applies shuffling, repeats for a number of epochs, maps the data through the 
    parse_tfrecord function, batches the data, and prepares it for training by prefetching.
    
    Args:
        dataset_type (str): A string indicating the type of dataset to setup, e.g., "train".
        data_aug (bool): Flag to determine whether data augmentation should be applied.

    Returns:
        tuple: A tuple containing the dataset and the raw TFRecordDataset. The former is
               batched and preprocessed, ready for training, while the latter is the 
               unprocessed TFRecordDataset.
    """
        
    # Create a list of all the training tfrecord files
    files = []
    tfrecord_fnames = os.listdir(
        os.path.join(args.dataset_dir, dataset_type, "tfrecords")
    )
    for i in range(1, len(tfrecord_fnames) + 1):
        files.append(
            os.path.join(
                args.dataset_dir,
                dataset_type,
                "tfrecords/bp-%s-%s.tfrecord" % (dataset_type, str(i).zfill(3)),
            )
        )
    # Create training dataset and add mapping function to parse through tfrecord files
    # in order to generate training input and labels with online data augmentation
    dataset = tf.data.TFRecordDataset(
        files, num_parallel_reads=6, buffer_size=512 * 1024
    )
    data = (
        dataset.shuffle(args.batch_size * 128)
        .repeat(args.epochs)
        .map(
            parse_tfrecord(size=IMG_HEIGHT, data_aug=data_aug),
            deterministic=True,
            num_parallel_calls=4,
        )
        .batch(args.batch_size, drop_remainder=True)
        .prefetch(8)
    )
    return data, dataset