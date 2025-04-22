import os
import gc
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.models as models
from torchmetrics.detection import MeanAveragePrecision
import matplotlib.pyplot as plt
import numpy as np
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
import optuna
from tqdm import tqdm
import functools
import json
import utils
import evaluator
import data_process

# Configuration

possible_optimizers = ['SGD', 'Adam', 'AdamW', 'RMSprop']
possible_schedulers = ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'OneCycleLR']

model_name_global = None
op_train_set = None
op_val_set = None
save_path = None

# Our kornia augmentation list (see readme for details)
aug_list = AugmentationSequential(     
    K.RandomMotionBlur(kernel_size=(3, 51), angle=(-180.0, 180.0), direction=(-1.0, 1.0), p=0.4),  # Random motion blur
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.4),  # Random Gaussian blur
    K.RandomSharpness(sharpness=(0.5, 2.0), p=0.3),  # Random sharpness
    K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, p=0.2),  # Random color jitter
    same_on_batch=False,
)

def train(model, train_loader, val_loader, optimizer, lr_scheduler, num_epochs=10, device="cuda", model_name="", save_path=None, trial=None, kornia_aug=False):
    """
    train function for torchvision models
    """
    global model_name_global, aug_list
    
    model.to(device)
    
    train_losses = []
    val_maps = []
    fps_list = []
    train_time = 0.0
    
    best_model_state_dict = model.state_dict()
    best_val_map = 0.0
    best_epoch = 0

    stagnant_epochs = 3  # Number of epochs with insufficient improvement before pruning
    num_fouls = 0  # Number of times the model has failed to improve sufficiently
    
    # Clear memory
    torch.cuda.empty_cache()

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets_to_device = [
                {'boxes': target['boxes'].to(device), 'labels': target['labels'].to(device)}
                for target in targets
            ]
            
            if kornia_aug is True:
                # apply augmentations  
                images = aug_list(images)
            
            # Convert back to list of tensors as expected by the model
            images = [img for img in images]
            
            loss_dict = model(images, targets_to_device)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        _, val_map, fps = evaluator.evaluate_model(model, val_loader, batch_size=val_loader.batch_size, device=device)
        mean_ap = val_map["map_50"].item()

        # Track the best model
        if mean_ap > best_val_map:
            best_val_map = mean_ap
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        
        # Step the learning rate scheduler
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(mean_ap)
            else:
                lr_scheduler.step()

        # Optuna reporting and pruning logic
        if trial is not None:
            trial.report(mean_ap, epoch)
            
            # Check when mean_ap is zero over the last few epochs
            if epoch > 0 and mean_ap == 0.0:
                num_fouls += 1
            else:
                num_fouls = 0
            
            if num_fouls > stagnant_epochs:
                print(f"Trial pruned by Optuna at epoch {epoch}.")
                raise optuna.TrialPruned()

            # Optuna pruning based on the reported value
            if trial.should_prune():
                print(f"Trial pruned by Optuna at epoch {epoch}.")
                raise optuna.TrialPruned()
        
        # Append metrics for tracking
        train_losses.append(epoch_loss)
        val_maps.append(mean_ap)
        fps_list.append(fps)
        
        end_time = time.time()
        train_time += (end_time - start_time)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val mAP@50: {mean_ap:.4f}, FPS: {fps:.2f}, Epoch Time: {(end_time - start_time):.2f} seconds")
    
    os.makedirs(save_path, exist_ok=True)

    if trial is not None:
        # Save the best trained model:
        torch.save(best_model_state_dict, os.path.join(save_path, f"{model_name_global}_{trial.number}_best.pth"))
    elif kornia_aug is True:
        torch.save(best_model_state_dict, os.path.join(save_path, f"{model_name}_aug_best.pth"))
    else:
        torch.save(best_model_state_dict, os.path.join(save_path, f"{model_name}_best.pth"))
    
    print(f"Best Epoch: {best_epoch}, Best Val mAP@50: {best_val_map:.4f}, Training Time: {train_time:.2f} seconds")
    print("Model training complete.")
    
    model_parameters, model_size = utils.calc_model_size(model)
    
    history = {
        "train_losses": train_losses,
        "val_maps": val_maps,
        "best_val_map": best_val_map,
        "fps": np.average(fps_list),
        "train_time": train_time,
        "model_parameters": model_parameters,
        "model_size": model_size,
    }
    
    return history
    
def train_from_config(model_config_path, train_set, val_set, save_path="data/models", kornia_aug=False, with_severity_levels=False):
    """
    reads the model configuration from a json file and trains the model accordingly
    """
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    model_name = model_config["model_name"]
    params = model_config["params"]
    
    train_loader = DataLoader(
        train_set,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=data_process.collate_fn
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=data_process.collate_fn
    )
    
    model = get_model(model_name=model_name, preweight_mode=params['preweight_mode'], with_severity_levels=with_severity_levels)
    if model is None:
        raise ValueError("Invalid model")
    
    if params['optimizer'] == 'SGD':
        optimizer_params = [params['lr'], params['momentum'], params['weight_decay']]
    elif params['optimizer'] == 'Adam':
        optimizer_params = [params['lr'], params['beta1'], params['beta2']]
    elif params['optimizer'] == 'AdamW':
        optimizer_params = [params['lr'], params['weight_decay'], params['beta1'], params['beta2']]
    elif params['optimizer'] == 'RMSprop':
        optimizer_params = [params['lr'], params['weight_decay'], params['momentum']]
    else:
        raise ValueError("Invalid optimizer")
    
    optimizer = get_optimizer(model.parameters(), optimizer_name=params['optimizer'], optimizer_params=optimizer_params)
    
    if params['scheduler'] == 'StepLR':
        scheduler_params = [params['step_size'], params['gamma']]
    elif params['scheduler'] == 'CosineAnnealingLR':
        scheduler_params = [params['T_max'], params['eta_min']]
    elif params['scheduler'] == 'ReduceLROnPlateau':
        scheduler_params = [params['factor'], params['patience']]
    elif params['scheduler'] == 'OneCycleLR':
        scheduler_params = [params['max_lr']]
    else:
        raise ValueError("Invalid scheduler")
    
    scheduler = get_scheduler(optimizer, params['epochs'], len(train_loader), scheduler_name=params['scheduler'], scheduler_params=scheduler_params)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting training for {model_name} with the following parameters:")
    print(params)
    
    history = train(model, train_loader, val_loader, optimizer, scheduler, params['epochs'], device=device, model_name=model_name, save_path=save_path, kornia_aug=kornia_aug)
    
    return history

def get_model(model_name="", trial=None, preweight_mode='fine_tuning', with_severity_levels=False):
    """
    given a model name the function returns the model with the specified preweight_mode
    if trial is not None, the function will use the trial object from optuna to suggest the preweight_mode
    """
    global model_name_global
    
    if trial is not None:
        model_name = model_name_global
        preweight_mode = trial.suggest_categorical('preweight_mode', ['random', 'freezing', 'fine_tuning'])
    
    # Retrieve the list of input channels, based of the dataset (annotations alone / annotations with severity levels)
    if with_severity_levels:
        num_classes = len(data_process.PotholeSeverityLevels)
    else:
        num_classes = len(data_process.PotholeSeverity)

    if str.startswith(model_name, "fasterrcnn"):
        if model_name == "fasterrcnn_resnet50_fpn":
            if preweight_mode == 'random':
                model = models.detection.fasterrcnn_resnet50_fpn()  # No pre-trained weights
            else:
                model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        elif model_name == "fasterrcnn_resnet50_fpn_v2":
            if preweight_mode == 'random':
                model = models.detection.fasterrcnn_resnet50_fpn_v2()  # No pre-trained weights
            else:
                model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)            
        elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
            if preweight_mode == 'random':
                model = models.detection.fasterrcnn_mobilenet_v3_large_fpn()  # No pre-trained weights
            else:
                model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)            
        elif model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
            if preweight_mode == 'random':
                model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()  # No pre-trained weights
            else:
                model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1)        
        else:
            return None
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes) 
        
        if preweight_mode == 'freezing':
            # Freeze all layers except the head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.roi_heads.box_predictor.parameters():
                param.requires_grad = True
        
        return model
          
    if str.startswith(model_name, "retinanet"):
        if model_name == "retinanet_resnet50_fpn":
            if preweight_mode == 'random':
                model = models.detection.retinanet_resnet50_fpn()
            else:
                model = models.detection.retinanet_resnet50_fpn(weights=models.detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1)
        elif model_name == "retinanet_resnet50_fpn_v2":
            if preweight_mode == 'random':
                model = models.detection.retinanet_resnet50_fpn_v2()
            else:
                model = models.detection.retinanet_resnet50_fpn_v2(weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        else:
            return None
        
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = models.detection.retinanet.RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=functools.partial(torch.nn.GroupNorm, 32)
        )
        
        if preweight_mode == 'freezing':
            # Freeze all layers except the head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.classification_head.parameters():
                param.requires_grad = True
        
        return model
    
    if model_name == "fcos_resnet50_fpn":
        if preweight_mode == 'random':
            model = models.detection.fcos_resnet50_fpn()
        else:
            model = models.detection.fcos_resnet50_fpn(weights=models.detection.FCOS_ResNet50_FPN_Weights.COCO_V1)
            
        # Replace the classifier with a Pothole-class output
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = models.detection.fcos.FCOSClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=functools.partial(torch.nn.GroupNorm, 32)
        )
        
        if preweight_mode == 'freezing':
            # Freeze all layers except the head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.classification_head.parameters():
                param.requires_grad = True
        
        return model
          
    if model_name == "ssd300_vgg16":
        if preweight_mode == 'random':
            model = models.detection.ssd300_vgg16()
        else:
            model = models.detection.ssd300_vgg16(weights=models.detection.SSD300_VGG16_Weights.COCO_V1)
        
        in_channels = models.detection._utils.retrieve_out_channels(model.backbone, (300, 300))
        # List containing number of anchors based on aspect ratios.
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # The classification head.
        model.head.classification_head = models.detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        # Image size for transforms.
        model.transform.min_size = (300,)
        model.transform.max_size = 300
        
        if preweight_mode == 'freezing':
            # Freeze all layers except the classification and regression heads
            for param in model.parameters():
                param.requires_grad = False  # Freeze all parameters initially
            # Unfreeze the classification head
            for param in model.head.classification_head.parameters():
                param.requires_grad = True
            # Unfreeze the box regression head
            for param in model.head.regression_head.parameters():
                param.requires_grad = True
        
        return model
    
    if model_name == "ssdlite320_mobilenet_v3_large":
        if preweight_mode == 'random':
            model = models.detection.ssdlite320_mobilenet_v3_large()
        else:
            model = models.detection.ssdlite320_mobilenet_v3_large(weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)

        in_channels = models.detection._utils.retrieve_out_channels(model.backbone, (320, 320))
        # List containing number of anchors based on aspect ratios.
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # The classification head.
        model.head.classification_head = models.detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        # Image size for transforms.
        model.transform.min_size = (320,)
        model.transform.max_size = 320
        
        if preweight_mode == 'freezing':
            # Freeze all layers except the classification and regression heads
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze the classification head
            for param in model.head.classification_head.parameters():
                param.requires_grad = True
            # Unfreeze the box regression head
            for param in model.head.regression_head.parameters():
                param.requires_grad = True
                
        return model
    
    return None

def get_optimizer(model_parameters, trial=None, optimizer_name='SGD', optimizer_params=[]):
    global possible_optimizers
    
    # Suggest optimizer type
    if trial is not None:
        optimizer_name = trial.suggest_categorical('optimizer', possible_optimizers)

    if optimizer_name not in possible_optimizers:
        raise ValueError("Invalid optimizer")
    
    if optimizer_name == 'SGD':
        if trial is not None:
            lr = trial.suggest_float('lr', 5e-3, 5e-2, log=True)
            momentum = trial.suggest_float('momentum', 0.9, 0.99)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        else:
            lr = optimizer_params[0]
            momentum = optimizer_params[1]
            weight_decay = optimizer_params[2]
        return torch.optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    elif optimizer_name == 'Adam':
        if trial is not None:
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            beta1 = trial.suggest_float('beta1', 0.8, 0.999)
            beta2 = trial.suggest_float('beta2', 0.9, 0.999)
        else:
            lr = optimizer_params[0]
            beta1 = optimizer_params[1]
            beta2 = optimizer_params[2]
        return torch.optim.Adam(model_parameters, lr=lr, betas=(beta1, beta2))
    
    elif optimizer_name == 'AdamW':
        if trial is not None:
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-2, log=True)
            beta1 = trial.suggest_float('beta1', 0.8, 0.999)
            beta2 = trial.suggest_float('beta2', 0.9, 0.999)
        else:
            lr = optimizer_params[0]
            weight_decay = optimizer_params[1]
            beta1 = optimizer_params[2]
            beta2 = optimizer_params[3]
        return torch.optim.AdamW(model_parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    
    else:  # RMSprop
        if trial is not None:
            lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-1, 1, log=True)
            momentum = trial.suggest_float('momentum', 0.9, 0.99)
        else:
            lr = optimizer_params[0]
            weight_decay = optimizer_params[1]
            momentum = optimizer_params[2]
        return torch.optim.RMSprop(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

def get_scheduler(optimizer, num_epochs, steps_per_epoch, trial=None, scheduler_name=None, scheduler_params=[]):
    global possible_schedulers
    
    # Suggest scheduler type
    if trial is not None:
        scheduler_name = trial.suggest_categorical('scheduler', possible_schedulers)
    
    if scheduler_name not in possible_schedulers:
        raise ValueError("Invalid scheduler")
    
    if scheduler_name == 'StepLR':
        if trial is not None:
            step_size = trial.suggest_int('step_size', 2, 5)
            gamma = trial.suggest_float('gamma', 0.05, 0.5)
        else:
            step_size = scheduler_params[0]
            gamma = scheduler_params[1]
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name == 'CosineAnnealingLR':
        if trial is not None:
            T_max = trial.suggest_int('T_max', 5, 15)
            eta_min = trial.suggest_float('eta_min', 1e-7, 1e-5, log=True)
        else:
            T_max = scheduler_params[0]
            eta_min = scheduler_params[1]
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_name == 'ReduceLROnPlateau':
        if trial is not None:
            factor = trial.suggest_float('factor', 0.1, 0.5)
            patience = trial.suggest_int('patience', 2, 5)
        else:
            factor = scheduler_params[0]
            patience = scheduler_params[1]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience)
    
    else:  # OneCycleLR
        if trial is not None:
            max_lr = trial.suggest_float('max_lr', 1e-4, 1e-2, log=True)
        else:
            max_lr = scheduler_params[0]
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=num_epochs, steps_per_epoch=steps_per_epoch)

################################# optuna functions #################################
def objective(trial): 
    global op_train_set, op_val_set, save_path_global  # Declare global variables
    ## clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Define hyperparameter search space
    batch_size = trial.suggest_int('batch_size', 4, 8)
    num_epochs = trial.suggest_int('epochs', 10, 20)
    
    train_loader = DataLoader(
        op_train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_process.collate_fn
    )
    
    val_loader = DataLoader(
        op_val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_process.collate_fn
    )

    model = get_model(trial=trial)
    if model is None:
        raise ValueError("Invalid model")
    
    print(f"Checking Model: {model.__class__.__name__}")
    
    # Get optimizer and scheduler
    optimizer = get_optimizer(model.parameters(), trial=trial)
    scheduler = get_scheduler(optimizer, num_epochs, len(train_loader), trial=trial)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    # Train the model
    print(f"Starting Trial #{trial.number}")
    print(trial.params)
    history = train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device=device, save_path=save_path_global, trial=trial)
        
    return history['best_val_map']
    
def run_optimization(model_name, train_set, val_set, study_name="optuna_check", save_path=None, n_trials=50):
    global model_name_global, op_train_set, op_val_set, save_path_global  # Declare global variables
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage=f"sqlite:///data/models/db.sqlite3",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=data_process.seed)
    )
    
    model_name_global = model_name
    op_train_set = train_set
    op_val_set = val_set
    save_path_global = save_path
    
    study.optimize(objective, n_trials=n_trials)
    
    # Print optimization results
    trial = study.best_trial
    print(f"\nBest trial: #{trial.number}")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    model_param = {"model_name": model_name, "best_trial": trial.number, "best_val_map": trial.value, "params": trial.params}
    # Save best params to json, the name is the model name:
    with open(os.path.join(save_path, f"{model_name}_best_params.json"), 'w') as f:
        json.dump(model_param, f)
    
    # Save study results
    study.trials_dataframe().to_csv(os.path.join(save_path,"optimization_results.csv"))
    
    return study