__config__ = {

    # dataset path
    'dataset_path': 'data\PVDN',  # Path to the dataset
    'log_path': 'log.txt',  # Path to the log file

    'max_num_car': 8,  # Maximum number of cars
    'max_num_light': 2,  # Maximum number of headlights

    # for model
    'nstack': 3,  # Number of stacks in the model (If GPU memory is insufficient, decrease this to 2)
    'inp_dim': 256,  # Input dimension
    'oup_dim': 4,  # Output dimension
    'increase': 128,  # Increase factor
    'input_res': 512,  # Input resolution
    'output_res': 128,  # Output resolution
    'threshold': 0.95,  # Threshold for predictions
    'min_threshold': 0.5,  # Minimum threshold for predictions

    # for training
    'bn': True,  # Use batch normalization
    'negative_samples': True,  # Use samples where there is no vehicles
    'autocast': True,  # Use automatic mixed precision (AMP)
    'save_all_models': True,  # Save all models during training
    'weighted_dataset': True,  # Use weighted dataset during training
    'day_samples': True,  # Use extra day samples containing multiple vehicles in training
    'override_saved_config': True,  # Override configuration of saved model
    'batch_size': 15,  # Training batch size (validation batch size is 2 * batch_size)
    'epochs': 200,  # Number of training epochs
    'scheduler': 'CosineAnnealingLR',  # Learning rate scheduler type
    'min_lr': 1e-6,  # Minimum learning rate for scheduler
    'val_epoch': 1,  # Validate the model every "config['val_epoch']" epoch
    'learning_rate': 1e-5,  # Initial learning rate
    'num_workers': 4,  # Number of workers for data loading
    'use_data_loader': True,  # Use data loader for training
    'loss': [['push_loss', 5e-1],  # Loss components and their weights
             ['pull_loss', 5e-1],
             ['detection_loss', 1]],
    'model': 'Hourglass',  # Model architecture
}
