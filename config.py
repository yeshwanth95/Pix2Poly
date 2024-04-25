import torch

class CFG:
    IMG_PATH = ''
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET = f"inria_coco_224_negAug"
    # PREPROCESS = check_preprocess(DATASET) if "inria" in DATASET and "coco" not in DATASET else False
    TRAIN_IMG_DIR = f"./data/{DATASET}/train_images"
    TRAIN_MASK_DIR = f"./data/{DATASET}/train_gts"
    VAL_IMG_DIR = f"./data/{DATASET}/val_images"
    VAL_MASK_DIR = f"./data/{DATASET}/val_gts"
    TEST_IMG_DIR = f"./data/{DATASET}/val_images"
    TEST_MASK_DIR = f"./data/{DATASET}/val_gts"
    if "crowdai" in DATASET or "coco" in DATASET:
        TRAIN_DATASET_DIR = f"./data/{DATASET}/train"
        VAL_DATASET_DIR = f"./data/{DATASET}/val"
        TEST_IMAGES_DIR = f"./data/{DATASET}/val/images"

    TRAIN_DDP = True
    NUM_WORKERS = 2
    PIN_MEMORY = True
    LOAD_MODEL = False

    if "inria" in DATASET:
        N_VERTICES = 192  # maximum number of vertices per image in dataset.
        # N_VERTICES = 384  # maximum number of vertices per image in dataset.
    elif "crowdai" in DATASET:
        N_VERTICES = 256  # maximum number of vertices per image in dataset.
    elif "spacenet" in DATASET:
        N_VERTICES = 192  # maximum number of vertices per image in dataset.

    SINKHORN_ITERATIONS = 100
    # MAX_LEN = 512  # maximum sequence length during training.
    MAX_LEN = (N_VERTICES*2) + 2
    if "crowdai" in DATASET:
        IMG_SIZE = 300
    elif "inria" in DATASET:
        IMG_SIZE = 224
    elif "spacenet" in DATASET:
        IMG_SIZE = 224
    INPUT_SIZE = 224
    PATCH_SIZE = 8
    INPUT_HEIGHT = INPUT_SIZE
    INPUT_WIDTH = INPUT_SIZE
    NUM_BINS = INPUT_HEIGHT*1
    LABEL_SMOOTHING = 0.0
    vertex_loss_weight = 1.0
    perm_loss_weight = 10.0
    vertex_reg_loss_weight = 0.0
    angle_reg_loss_weight = 0.0
    SHUFFLE_TOKENS = False  # order gt vertex tokens randomly every time

    BATCH_SIZE = 24
    if "crowdai" in DATASET:
        BATCH_SIZE = 18 # batch size per gpu; effective batch size = BATCH_SIZE * NUM_GPUs
    START_EPOCH = 0
    NUM_EPOCHS = 500
    MILESTONE = 0
    SAVE_BEST = True
    SAVE_LATEST = True
    SAVE_EVERY = 10
    VAL_EVERY = 1
    # SAVE_EVERY = NUM_EPOCHS
    # VAL_EVERY = 50

    MODEL_NAME = f'vit_small_patch{PATCH_SIZE}_{INPUT_SIZE}_dino'
    # MODEL_NAME = 'deit_small_patch16_224'
    # MODEL_NAME = 'resnet50'
    # MODEL_NAME = 'hrnet_w48'
    # NUM_PATCHES = 196
    # NUM_PATCHES = 576
    NUM_PATCHES = int((INPUT_SIZE // PATCH_SIZE) ** 2)

    LR = 4e-4
    WEIGHT_DECAY = 1e-4

    generation_steps = (N_VERTICES * 2) + 1  # sequence length during prediction. Should not be more than max_len
    run_eval = False

    # EXPERIMENT_NAME = f"debug_run_Pix2Poly224_Bins{NUM_BINS}_fullRotateAugs_permLossWeight{perm_loss_weight}_LR{LR}_BS{BATCH_SIZE}_{NUM_EPOCHS}epochs"
    EXPERIMENT_NAME = f"CYENS_CLUSTER_train_Pix2Poly_AUGSRUNS_{DATASET}_run1_{MODEL_NAME}_AffineRotaugs0.8_LinearWarmupLRS_{vertex_loss_weight}xVertexLoss_{perm_loss_weight}xPermLoss__2xScoreNet_initialLR_{LR}_bs_{BATCH_SIZE}_Nv_{N_VERTICES}_Nbins{NUM_BINS}_{NUM_EPOCHS}epochs"

    if "debug" in EXPERIMENT_NAME:
        BATCH_SIZE = 10
        NUM_WORKERS = 0
        SAVE_BEST = False
        SAVE_LATEST = False
        SAVE_EVERY = NUM_EPOCHS
        VAL_EVERY = 50

    if LOAD_MODEL:
        CHECKPOINT_PATH = f"runs/{EXPERIMENT_NAME}/logs/checkpoints/latest.pth"  # full path to checkpoint to be loaded if LOAD_MODEL=True
    else:
        CHECKPOINT_PATH = ""