import random
import glob
import os
import argparse
import scipy.misc
import numpy as np
import tensorflow as tf
from itertools import permutations
from typing import List, Tuple

def path_to_image_tensor(path: str, height: int, width: int, channels: int=3) -> tf.Tensor:
  return tf.cast(tf.image.resize_image_with_pad(tf.image.decode_png(tf.read_file(path), channels=channels), height, width), tf.uint8)

def process_files(data_path: ((str, str, str), (str, str, str))) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
  s_img = path_to_image_tensor(data_path[0][0], 224, 224)
  s_depth = path_to_image_tensor(data_path[0][1], 224, 224, 1)
  s_mask = path_to_image_tensor(data_path[0][2], 224, 224, 1)
  q_img = path_to_image_tensor(data_path[1][0], 500, 500)
  q_depth = path_to_image_tensor(data_path[1][1], 500, 500, 1)
  q_mask = path_to_image_tensor(data_path[1][2], 500, 500, 1)
  return s_img, s_depth, s_mask, q_img, q_depth, q_mask

def normalize_n_structure_data(
    s_img: tf.Tensor, 
    s_depth: tf.Tensor, 
    s_mask: tf.Tensor, 
    q_img: tf.Tensor, 
    q_depth: tf.Tensor, 
    q_mask: tf.Tensor
  ) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
  s_img = 2.0*((tf.cast(s_img, tf.float32)/255.0)-0.5) # -1 to 1
  s_depth = tf.cast(s_depth, tf.float32)/255.0
  s_mask = tf.cast(s_mask, tf.float32)/255.0
  q_img = 2.0*((tf.cast(q_img, tf.float32)/255.0)-0.5)  # -1 to 1
  q_depth = tf.cast(q_depth, tf.float32)/255.0
  q_mask = tf.cast(q_mask, tf.float32)/255.0
  s_mask =  tf.tile(s_mask, [1,1,3])
  return s_img, s_depth, s_mask, q_img, q_depth, q_mask

def get_dataset(file_list: List[Tuple[Tuple[str, str, str], Tuple[str, str, str]]]) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensor_slices(file_list).map(process_files)

def get_dataset_from_generated_tfrec(tf_file: str) -> tf.data.Dataset:
    return tf.data.TFRecordDataset(tf_file). \
        batch(6, drop_remainder=True). \
        map(lambda ds: (
            tf.ensure_shape(tf.image.decode_png(ds[0]),(224, 224, 3)), 
            tf.ensure_shape(tf.image.decode_png(ds[1]),(224, 224, 1)),
            tf.ensure_shape(tf.image.decode_png(ds[2]),(224, 224, 1)),
            tf.ensure_shape(tf.image.decode_png(ds[3]),(500, 500, 3)), 
            tf.ensure_shape(tf.image.decode_png(ds[4]),(500, 500, 1)),
            tf.ensure_shape(tf.image.decode_png(ds[5]),(500, 500, 1))
        ))

def encode_data(
    s_img: tf.Tensor, 
    s_depth: tf.Tensor,
    s_mask: tf.Tensor, 
    q_img: tf.Tensor, 
    q_depth: tf.Tensor,
    q_mask: tf.Tensor
    ) -> tf.Tensor:
    return tf.stack([
        tf.image.encode_png(s_img),
        tf.image.encode_png(s_depth),
        tf.image.encode_png(s_mask),
        tf.image.encode_png(q_img),
        tf.image.encode_png(q_depth),
        tf.image.encode_png(q_mask)
    ])

def get_data_file_lists(data_dir, sample_per_scene, test_classes, ktest_scenes, shuffled=True, include_negatives=False):
    random.seed(0)

    test_classes = set(test_classes)
    known_test_set = set(ktest_scenes)
    train_files = []
    test_files = []
    ktest_files = []
    per_class = {}
    if include_negatives:
        black_img_path = os.path.join(data_dir, "__BLAcK__.png")
        if not os.path.exists(black_img_path):
            scipy.misc.imsave(black_img_path, np.zeros((500, 500)))
    for c in os.listdir(data_dir):
        d = os.path.join(data_dir, c)
        if os.path.isdir(d):
        #     choices = []
            per_class[c] = []
            for i in os.listdir(d):
                sub = os.path.join(d, i)
                files = []
                for f in glob.iglob(os.path.join(sub, "*[0-9].png")):
                    fn, ext = f.rsplit('.', 1)
                    mask_f = fn + '_mask.' + ext
                    depth_f = fn + '_depth.' + ext
                    if not os.path.exists(mask_f) or not os.path.exists(depth_f):
                        continue
                    files.append((f, depth_f, mask_f))
                    per_class[c].append((f, depth_f, mask_f))
            #       choices.extend(random.sample(files, sample_per_scene))
                if not files:
                    print(f'no matching file in {d}')
                    continue
                choices = random.sample(files, sample_per_scene)
                data = list(permutations(choices, 2))
                if include_negatives and len(per_class)>=2:
                    l = len(data)
                    for idx in range(l):
                        n = c
                        while n == c or len(per_class.get(n, [])) < 2:
                            n = random.choice(list(per_class.keys()))
                        r = random.choice(per_class[n])
                        data.append((data[idx][0], (r[0], r[1], black_img_path)))
                if c in test_classes:
                    test_files.extend(data)
                elif i in known_test_set:
                    ktest_files.extend(data)
                else:
                    train_files.extend(data)
    if shuffled:
        random.shuffle(train_files)
        random.shuffle(test_files)
        random.shuffle(ktest_files)
    return train_files, test_files, ktest_files

def generate_tfrec_files(data_dir, sample_per_scene, test_classes, ktest_scenes, negatives):
    train_files, test_files, ktest_files = get_data_file_lists(
        data_dir, 
        sample_per_scene, 
        test_classes, 
        ktest_scenes,
        include_negatives=negatives
    )
            
    print(f"Train size: {len(train_files)}\nTest size: {len(test_files)}\nKnown class test size: {len(ktest_files)}")

    print("Generating TFRecord files...")

    sess = tf.Session()

    for (f,d) in {'train.tfrec': train_files, 'test.tfrec': test_files, 'ktest.tfrec': ktest_files}.items():
        if not len(d): continue
        tfrec = tf.data.experimental.TFRecordWriter(f)
        ds = get_dataset(d).flat_map(lambda a,b,c,d,e,f: tf.data.Dataset.from_tensor_slices(encode_data(a,b,c,d,e,f)))
        sess.run(tfrec.write(ds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dataset as 3 tfrecord files (train, test(unknown), test(known))')
    parser.add_argument('-s',dest='sample_per_scene', type=int, nargs='?', default=4, help='samples to take per scene folder')
    parser.add_argument('--test_cls', dest='test_classes', type=str, nargs='*', default=["tomato", "toothpaste"], help='classes to put in test set')
    parser.add_argument('--ktest_scene', dest='ktest_scenes', type=str, nargs='*', default=["cap_3", "food_box_3", "soda_can_2"], help='classes to put in test set')
    parser.add_argument('-d', dest='dir', type=str, nargs='?', default="dataset", help='root directory of RGBD dataset')
    parser.add_argument('--negatives', dest='negatives', action='store_true')
    args = parser.parse_args()

    generate_tfrec_files(args.dir,
        args.sample_per_scene,
        args.test_classes,
        args.ktest_scenes,
        args.negatives)