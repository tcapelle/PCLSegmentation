import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import numpy as np
    
    
def resize(img, factor=1):
    return tf.image.resize(img, [img.shape[0]*factor, img.shape[1]*factor]).numpy()


def _create_row(lidar_input, label, class_color_map):
    "Create tuple of images and points"
    label_image = class_color_map[label.reshape(-1)].reshape([label.shape[0], label.shape[1], 3])
    depth_image = lidar_input[:, :, [4]]
    intensity = lidar_input[:, :, [3]]
    points = lidar_input[:, :, :3]
    points_rgb = np.concatenate([points, (255*label_image).astype(int)], axis=-1).reshape(-1, 6)
    
    depth_image, label_image, intensity_image  = map(resize, [depth_image, label_image, intensity])
    return (wandb.Image(label_image), 
            wandb.Image(depth_image), 
            wandb.Image(intensity_image), 
            wandb.Object3D({"type": "lidar/beta", "points":points_rgb}),
           )

def log_input_data(lidar_input, label, class_color_map, step=None):
    "Log to media panel"
    label_image, depth_image, intensity_image, points_rgb = _create_row(lidar_input, 
                                                                        label,
                                                                        class_color_map)
    # log 2 wandb
    wandb.log({'Images/Label Image': label_image},step=step)
    wandb.log({'Images/Depth Image': depth_image},step=step)
    wandb.log({'Images/Intensity Image': intensity_image},step=step)
    wandb.log({"Images/3D_label": points_rgb},step=step)
    
    
    
def _create_pred_row(lidar_input, prediction, label, class_color_map):
    pred_image = class_color_map[prediction.reshape(-1)].reshape([label.shape[0], label.shape[1], 3])
    points = lidar_input[...,:3]
    points_preds_rgb = np.concatenate([points, (255*pred_image).astype(int)], axis=-1).reshape(-1, 6)
    pred_image = resize(pred_image)
    return (wandb.Image(pred_image),
            wandb.Object3D({"type": "lidar/beta", "points":points_preds_rgb}),
           )
    
def log_model_predictions(lidar_input, prediction, label, class_color_map, step=None):
    "Log pred image and points"
    pred_image, points_rgb = _create_pred_row(lidar_input, prediction, label, class_color_map)
    wandb.log({'Images/Prediction Image':pred_image},step=step)
    wandb.log({"Images/3D_preds": points_rgb},step=step)
    
    
def create_pred_table(lidar_inputs, labels, predictions, class_color_map):
    table_data = []
    for i, (lidar_input, label, prediction) in enumerate(zip(lidar_inputs,labels, predictions)):
        input_row = _create_row(lidar_input, label, class_color_map)
        preds_row = _create_pred_row(lidar_input, prediction, label, class_color_map)
        table_data.append(input_row+preds_row)
    table = wandb.Table(columns=["Label Image", "Depth Image", 
                                 "Intensity Image", "LiDAR", "Pred Image", "Pred LiDAR"],
                        data=table_data)
    return table

    
class LogSamplesCallback(WandbCallback):
    "A simple Keras callback to log model predictions"
    
    def __init__(self, dataset, log_epoch_preds=False, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.num_images = 5
        self.log_epoch_preds = log_epoch_preds
    
    
    def compute_preds(self):
        # a batch of images
        (lidar_inputs, lidar_masks), labels, weights = self.dataset.take(1).get_single_element() 
        
        num_images = min(self.num_images, lidar_inputs.shape[0])
        
        # select a fixed number of inputs
        lidar_inputs = lidar_inputs[:num_images, :, :]
        lidar_masks = lidar_masks[:num_images, :, :]
        labels = labels[:num_images, :, :]
        weights = weights[:num_images, :, :]

        # forward pass
        probabilities, predictions = self.model([lidar_inputs, lidar_masks])
        
        return lidar_inputs.numpy(), predictions.numpy(), labels.numpy()
        
        
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        
        if self.log_epoch_preds:
            lidar_inputs, predictions, labels = self.compute_preds()
    
            for i, (lidar_input, prediction, label) in enumerate(zip(lidar_inputs, predictions, labels)):
                log_input_data(lidar_input, label, self.model.CLS_COLOR_MAP)
                log_model_predictions(lidar_input, prediction, label, self.model.CLS_COLOR_MAP)
    
    def on_train_end(self, logs=None):
        super().on_train_end(logs=logs)
        
        lidar_inputs, predictions, labels = self.compute_preds()
        table = create_pred_table(lidar_inputs, predictions, labels, self.model.CLS_COLOR_MAP)

        wandb.log({"pred_val":table})
        