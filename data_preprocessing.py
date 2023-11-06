def create_tf_example(tfrecord_data):

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



def parse_tfrecord(size=256, data_aug=True, file_name=False):
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