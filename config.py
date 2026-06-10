import torch

class CFG:
    IMG_PATH = ''
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    supported datasets are:
    - inria_coco_224_negAug
    - spacenet_coco
    - whu_buildings_224_coco
    - mass_roads_224
    """
    DATASET = f"inria_coco_224_negAug"
    if "coco" in DATASET:
        TRAIN_DATASET_DIR = f"./data/{DATASET}/train"
        VAL_DATASET_DIR = f"./data/{DATASET}/val"
        TEST_IMAGES_DIR = f"./data/{DATASET}/val/images"
    elif "mass_roads" in DATASET:
        TRAIN_DATASET_DIR = f"./data/{DATASET}/train"
        VAL_DATASET_DIR = f"./data/{DATASET}/valid"
        TEST_IMAGES_DIR = f"./data/{DATASET}/test/images"


    TRAIN_DDP = True
    NUM_WORKERS = 2
    PIN_MEMORY = True
    LOAD_MODEL = False

    if "inria" in DATASET:
        N_VERTICES = 192  # maximum number of vertices per image in dataset.
    elif "spacenet" in DATASET:
        N_VERTICES = 192  # maximum number of vertices per image in dataset.
    elif "whu_buildings" in DATASET:
        N_VERTICES = 144  # maximum number of vertices per image in dataset.
    elif "mass_roads" in DATASET:
        N_VERTICES = 192  # maximum number of vertices per image in dataset.

    SINKHORN_ITERATIONS = 100
    MAX_LEN = (N_VERTICES*2) + 2
    if "inria" in DATASET:
        IMG_SIZE = 224
    elif "spacenet" in DATASET:
        IMG_SIZE = 224
    elif "whu_buildings" in DATASET:
        IMG_SIZE = 224
    elif "mass_roads" in DATASET:
        IMG_SIZE = 224
    INPUT_SIZE = 224
    PATCH_SIZE = 8
    INPUT_HEIGHT = INPUT_SIZE
    INPUT_WIDTH = INPUT_SIZE
    NUM_BINS = INPUT_HEIGHT*1
    LABEL_SMOOTHING = 0.0
    vertex_loss_weight = 1.0
    perm_loss_weight = 10.0
    SHUFFLE_TOKENS = False  # order gt vertex tokens randomly every time

    # Tiling configuration
    TILE_SIZE = IMG_SIZE
    TILE_OVERLAP = 34

    # Polygon filtering configuration
    # MIN_POLYGON_AREA: Minimum area in square pixels for a polygon to be considered valid
    # Conversion: 100 sq ft ≈ 9.29 sq meters
    # Since 1 pixel = 0.3m, 1 sq pixel = 0.09 sq meters
    # Therefore, 9.29 sq meters ≈ 103 sq pixels
    MIN_POLYGON_AREA = 103

    # POLYGON_SIMPLIFICATION_TOLERANCE: Maximum distance in pixels that a point can be moved during polygon simplification
    # A pixel is 0.3m, so 2 pixels is 0.6m, or 2 feet, which should be of no consequence
    POLYGON_SIMPLIFICATION_TOLERANCE = 2

    # Prediction configuration
    PREDICTION_BATCH_SIZE = 8  # Batch size for processing tiles during prediction
    
    # Polygon validation configuration
    MERGE_TOLERANCE = 2  # Tolerance for point-in-polygon tests during validation (in pixels, allows points to be slightly outside)
    TILE_OVERLAP_RATIO = 0.5  # Overlap ratio between tiles (0.0 = no overlap, 1.0 = complete overlap)
    
    BATCH_SIZE = 24  # batch size per gpu; effective batch size = BATCH_SIZE * NUM_GPUs
    START_EPOCH = 0
    NUM_EPOCHS = 500
    MILESTONE = 0
    SAVE_BEST = True
    SAVE_LATEST = True
    SAVE_EVERY = 10
    VAL_EVERY = 1

    MODEL_NAME = f'vit_small_patch{PATCH_SIZE}_{INPUT_SIZE}.dino'
    NUM_PATCHES = int((INPUT_SIZE // PATCH_SIZE) ** 2)

    LR = 4e-4
    WEIGHT_DECAY = 1e-4

    generation_steps = (N_VERTICES * 2) + 1  # sequence length during prediction. Should not be more than max_len
    run_eval = False

    # EXPERIMENT_NAME = f"debug_run_Pix2Poly224_Bins{NUM_BINS}_fullRotateAugs_permLossWeight{perm_loss_weight}_LR{LR}__{NUM_EPOCHS}epochs"
    EXPERIMENT_NAME = f"train_Pix2Poly_{DATASET}_run1_{MODEL_NAME}_AffineRotaugs0.8_LinearWarmupLRS_{vertex_loss_weight}xVertexLoss_{perm_loss_weight}xPermLoss__2xScoreNet_initialLR_{LR}_bs_{BATCH_SIZE}_Nv_{N_VERTICES}_Nbins{NUM_BINS}_{NUM_EPOCHS}epochs"

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

