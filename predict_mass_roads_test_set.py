import os
import numpy as np
from PIL import Image
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches
# import gif
from tqdm import tqdm

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tokenizer import Tokenizer
from test_config import CFG
from models.model import (
    Encoder,
    Decoder,
    EncoderDecoder,
)
from utils import (
    seed_everything,
    test_generate,
    postprocess,
    permutations_to_polygons,
    calculate_slice_bboxes,
)
import time


def get_rectangle_params_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def draw_bboxes(
    plot_ax,
    bboxes,
    class_labels,
    get_rectangle_corners_fn=get_rectangle_params_from_pascal_bbox,
):
    for bbox, label in zip(bboxes, class_labels):
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left, width, height, linewidth=4, edgecolor="black", fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left, width, height, linewidth=2, edgecolor="white", fill=False,
        )
        rx, ry = rect_1.get_xy()

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)
        plot_ax.annotate(label, (rx+width, ry+height), color='white', fontsize=20)

# @gif.frame
def show_image(image, bboxes=None, class_labels=None, draw_bboxes_fn=draw_bboxes):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    if bboxes:
        draw_bboxes_fn(ax, bboxes, class_labels)

    # plt.show()


def bounding_box_from_points(points):
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0]/2) * 2
    odd_locations = even_locations + 1
    X = np.take(points, even_locations.tolist())
    Y = np.take(points, odd_locations.tolist())
    bbox = [X.min(), Y.min(), X.max()-X.min(), Y.max()-Y.min()]
    bbox = [int(b) for b in bbox]
    return bbox


def single_annotation(image_id, poly):
    _result = {}
    _result["image_id"] = int(image_id)
    _result["category_id"] = 100 
    _result["score"] = 1
    _result["segmentation"] = poly
    _result["bbox"] = bounding_box_from_points(_result["segmentation"])
    return _result


def main(args):
    BATCH_SIZE = int(args.batch_size)  # 24
    PATCH_SIZE = int(args.img_size)  # 224
    INPUT_HEIGHT = int(args.input_size)  # 224
    INPUT_WIDTH = int(args.input_size)  # 224

    # EXPERIMENT_NAME = f"CYENS_CLUSTER_train_Pix2PolyFullDataExps_inria_coco_224_negAug_run1_deit3_small_patch16_384_in21ft1k_Rotaugs_LinearWarmupLRS_NoShuffle_1.0xVertexLoss_10.0xPermLoss_0.0xVertexRegLoss__2xScoreNet_initialLR_0.0004_bs_16_Nv_192_Nbins384_LbSm_0.0_500epochs"
    EXPERIMENT_PATH = args.experiment_path
    EXPERIMENT_NAME = os.path.basename(os.path.abspath(EXPERIMENT_PATH))
    CHECKPOINT_NAME = args.checkpoint_name
    CHECKPOINT_PATH = f"runs/{EXPERIMENT_NAME}/logs/checkpoints/{CHECKPOINT_NAME}.pth"

    SPLIT = args.split  # 'val' or 'test'

    test_image_dir = f"data/mass_roads_1500/test/sat"
    val_image_dir = f"data/mass_roads_1500/val/sat"

    if SPLIT == "test":
        test_images = []
        for im in os.listdir(test_image_dir):
            test_images.append(im)
        image_dir = test_image_dir
        images = test_images
    elif SPLIT == "val":
        val_images = []
        for im in os.listdir(val_image_dir):
            val_images.append(im)
        image_dir = val_image_dir
        images = val_images
    else:
        raise ValueError("Specify either test or val split for prediction.")

    test_transforms = A.Compose(
        [
            A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=INPUT_WIDTH,
        height=INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
    decoder = Decoder(
        cfg=CFG,
        vocab_size=tokenizer.vocab_size,
        encoder_len=CFG.NUM_PATCHES,
        dim=256,
        num_heads=8,
        num_layers=6
    )
    model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
    model.to(CFG.DEVICE)
    model.eval()

    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epochs_run']

    print(f"Model loaded from epoch: {epoch}")
    ckpt_desc = f"epoch_{epoch}"
    if "best_valid_loss" in os.path.basename(CHECKPOINT_PATH):
        ckpt_desc = f"epoch_{epoch}_bestValLoss"
    elif "best_valid_metric" in os.path.basename(CHECKPOINT_PATH):
        ckpt_desc = f"epoch_{epoch}_bestValMetric"
    else:
        pass


    results_dir = os.path.join(f"runs/{EXPERIMENT_NAME}", f"{SPLIT}_predictions", ckpt_desc)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "raster_preds"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "polygon_preds"), exist_ok=True)


    with torch.no_grad():
        for idx, image in enumerate(tqdm(images)):
            print(f"<---------Processing {idx+1}/{len(images)}: {image}----------->")
            img_name = image
            if os.path.exists(os.path.join(results_dir, 'raster_preds', img_name)):
                continue
            img = Image.open(os.path.join(image_dir, img_name))
            img = np.array(img)

            slice_bboxes = calculate_slice_bboxes(
                image_height=img.shape[1],
                image_width=img.shape[0],
                slice_height=PATCH_SIZE,
                slice_width=PATCH_SIZE,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )

            speed = []
            predictions = []
            for bi, box in enumerate(tqdm(slice_bboxes)):
                xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = box
                patch = img[ymin_top_left:ymax_bottom_right, xmin_top_left:xmax_bottom_right]
                patch = test_transforms(image=patch.astype(np.float32))['image'][None]

                all_coords = []
                all_confs = []
                t0 = time.time()
                batch_preds, batch_confs, perm_preds = test_generate(model, patch, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
                speed.append(time.time() - t0)
                vertex_coords, confs = postprocess(batch_preds, batch_confs, tokenizer)

                all_coords.extend(vertex_coords)
                all_confs.extend(confs)

                coords = []
                for i in range(len(all_coords)):
                    if all_coords[i] is not None:
                        coord = torch.from_numpy(all_coords[i])
                    else:
                        coord = torch.tensor([])

                    padd = torch.ones((CFG.N_VERTICES - len(coord), 2)).fill_(CFG.PAD_IDX)
                    coord = torch.cat([coord, padd], dim=0)
                    coords.append(coord)
                batch_polygons = permutations_to_polygons(perm_preds, coords, out='torch')  # [0, 224]

                for ip, pp in enumerate(batch_polygons):
                    if pp is not None:
                        for p in pp:
                            if p is not None:
                                p = torch.fliplr(p)
                                p = p[p[:, 0] != CFG.PAD_IDX]
                                p = p * (PATCH_SIZE / INPUT_WIDTH)
                                p[:, 0] = p[:, 0] + xmin_top_left
                                p[:, 1] = p[:, 1] + ymin_top_left
                                if len(p) > 0:
                                    if (p[0] == p[-1]).all():
                                        p = p [:-1]
                                p = p.view(-1).tolist()
                                if len(p) > 0:
                                    predictions.append(single_annotation(idx, [p]))
                # For debugging
                # if bi >= 10:
                #     break

            H, W = img.shape[0], img.shape[1]

            polygons_mask = np.zeros((H, W))
            for pred in predictions:
                poly = np.array(pred['segmentation'])
                poly = poly.reshape((poly.shape[-1]//2, 2))
                cv2.polylines(polygons_mask, [np.int32(poly)], isClosed=False, color=1., thickness=5)
            polygons_mask = (polygons_mask*255).astype(np.uint8)

            cv2.imwrite(os.path.join(results_dir, 'raster_preds', img_name), polygons_mask)
            print("Average model speed: ", np.mean(speed), " [s / patch]")
            print("Time for a single tile: ", np.sum(speed), " [s / tile]")

            with open(f"{results_dir}/polygon_preds/{img_name.split('.')[0]}.json", "w") as fp:
                fp.write(json.dumps(predictions))


    ############# Visualizations #################:
    # frames = []
    # for slice in tqdm(slice_bboxes):
    #     frames.append(show_image(img, [slice], ['']))
    #     # if sid > 40:
    #     #     break

    # gif.save(frames, "overlapping_patches.gif",
    #          duration=15)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--experiment_path", help="path to experiment folder to evaluate.")
    parser.add_argument("-c", "--checkpoint_name", help="Choice of checkpoint to evaluate in experiment.")
    parser.add_argument("-s", "--split", help="Dataset split to use for prediction ('test' or 'val').")
    parser.add_argument("--img_size", help="Original image size.")
    parser.add_argument("--input_size", help="Image size of input to network.")
    parser.add_argument("--batch_size", help="Batch size to network.")
    args = parser.parse_args()

    main(args)

