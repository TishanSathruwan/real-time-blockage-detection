import torch 
import numpy as np
from ultralytics import YOLO

class blockageDetectionModel:
    def __init__(self):
        
        # yolo model initiation
        self.detection_model = YOLO("yolov10n.pt")
        
        # depth estimation model initiation
        model_type = "MiDaS_small"
        self.device = torch.device("cpu")
        self.depth_model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device)
        self.depth_model = self.depth_model.eval()

        # Define transformations for depth model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def preprocess(self,output):
        output_min = output.min()
        output_max = output.max()
        output_normalized = 255 * (output - output_min) / (output_max - output_min)
        output_normalized *= 3
        output_normalized = np.repeat(np.expand_dims(output_normalized, 2), 3, axis=2) / 3
        return output_normalized
    
    def forward(self, frame):

        #yolo inference
        yolo_out = self.detection_model(frame)

        # apply transformation 
        input_image = self.transform(frame).to(self.device)
        
        # midas inference
        with torch.no_grad():
            midas_pred = self.depth_model(input_image)

            midas_pred = torch.nn.functional.interpolate(
                midas_pred.unsqueeze(1),
                size=frame.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        midas_raw_out = midas_pred.cpu().numpy()
        depth_out = self.preprocess(midas_raw_out)

        return [yolo_out, depth_out]    


