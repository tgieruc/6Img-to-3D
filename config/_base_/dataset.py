dataset_params = dict(
    data_path="/app/data/",
    version="triplane",
    train_data_loader=dict(
        pickled=True,  # If True, first run 'python utils/pickles_generator.py --dataset-config <path to this file> --py-config <path to config.py>' to preprocess the dataset and save as pickles for faster data loading
        phase="train",
        batch_size=1,
        shuffle=True,
        num_workers=12,
        town=["Town01", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"],
        weather=["ClearNoon"],
        vehicle=["vehicle.tesla.invisible"],
        spawn_point=["all"],
        step=["all"],
        selection=["input_images", "sphere_dataset"],
        factor=0.08,
        whole_image=True,
        num_imgs=3,
        depth=True,
        min_cams_train=1,
        max_cams_train=6,
    ),
    val_data_loader=dict(
        pickled=False,  # True is unsupported
        phase="test",  # "train", "test", "full"
        batch_size=1,
        shuffle=False,
        num_workers=12,
        town=["Town02"],
        weather=["ClearNoon"],
        vehicle=["vehicle.tesla.invisible"],
        spawn_point=[3, 7, 12, 48, 98, 66],
        step=["all"],
        selection=["input_images", "sphere_dataset"],
        factor=0.25,
        depth=True,
    ),
)
