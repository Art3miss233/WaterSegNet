from PIL import Image
import segmentation_models_pytorch as smp
import torch
import supervision as sv
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('/home/emilia/WaterSegNet'))
from lib.predict import *
from ultralytics import YOLO
import cv2
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import time



class SafePointFinder():
    def __init__(self, seg_chkpt_path, det_chpth_path):
        self.load_model(seg_chkpt_path)
        self.det_model = YOLO(det_chpth_path)
        self.annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)


    def load_model(self, chkpt_path):
        checkpoint = torch.load(chkpt_path)
        state_dict = checkpoint["state_dict"]
        self.seg_model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seg_model.to(self.device)
        self.seg_model.load_state_dict(state_dict)

    def detect_objects(self, img_path, threshold=0.5):
        """
        Detect objects in an image and return a list of detections.
        """
        
        img = cv2.imread(img_path)
        result = self.det_model.predict(img, conf=threshold)[0]
        detections = sv.Detections.from_ultralytics(result)

        labels = [
            f"{self.det_model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detections
        ]

        self.processed_img = self.annotator.annotate(
            scene=img, detections=detections, labels=labels
        )

        return detections
    
    def bbox_to_mask(self, detections, mask_255, padding_factor=0.2):
        """
        Takes a list of detections and a mask image and returns a mask with the bounding boxes
        """
        floating_obj_mask = np.ones(mask_255.shape[:2], dtype=np.uint8) * 255
        human_mask = np.ones(mask_255.shape[:2], dtype=np.uint8) * 255

        for bbox, _, _, class_id, _ in detections:
            bbox = bbox.astype(int)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # Calculate padding
            pad_x = int(width * padding_factor)
            pad_y = int(height * padding_factor)

            # Apply padding
            padded_bbox = [
                bbox[0] - pad_x,
                bbox[1] - pad_y,
                bbox[2] + pad_x,
                bbox[3] + pad_y,
            ]

            if class_id == 1:
                floating_obj_mask[
                    padded_bbox[1] : padded_bbox[3], padded_bbox[0] : padded_bbox[2]
                ] = 0
            elif class_id == 0:
                human_mask[
                    padded_bbox[1] : padded_bbox[3], padded_bbox[0] : padded_bbox[2]
                ] = 0

        return floating_obj_mask, human_mask
    
    def find_closest_safe_point(self, mask, coords=None):
        """
        Calculate distance map and find closest safe point to centre.
        """
        distance_map, indices = ndimage.distance_transform_edt(
            mask, return_indices=True
        )

        if coords is None:
            center_coords = np.array(mask.shape) // 2
            coords = center_coords

        indices_at_coords = indices[:, coords[0], coords[1]]
        self.distance_map = distance_map
       
        return indices_at_coords[1], indices_at_coords[0]

    def draw_processed_mask(self, mask_255, padded_water, dynamic_points):
        """
        Visualize the processed mask.
        """
        mask_rgb = cv2.cvtColor(mask_255, cv2.COLOR_GRAY2RGB)

        water_padding_color = (255, 178, 0)
        dynamic_color = (0, 150, 100)

        padded_area = padded_water > mask_255

        mask_rgb[dynamic_points] = dynamic_color
        mask_rgb[padded_area] = water_padding_color


        return mask_rgb

    def show_results(self, img_raw, mask_255, safe_mask, processed_mask, shaded_image):
        """
        Show the results of the segmentation.
        """
        
        images = [img_raw, mask_255,  safe_mask, processed_mask, shaded_image, self.processed_img]
        titles = ["Original image", 
                  "Mask", 
                  "Safe region", 
                  "Processed mask", 
                  "Image with mask overlay and safe point", 
                  "Image with detections"
                  ]
        for i, img in enumerate(images):
            plt.figure(figsize=(10, 5))
            plt.imshow(img)
            plt.title(titles[i])
            plt.axis("off")
            plt.show()

        plt.imshow(self.distance_map)
        plt.colorbar()
        plt.title("Distance map")
        plt.show()


        plt.show()


    def apply_padding(self, mask_255, padding_factor):
        """
        Apply padding to the mask around the water.
        """
        kernel = np.ones((padding_factor, padding_factor), np.uint8)
        padded_mask = cv2.dilate(mask_255, kernel, iterations=1)
        return padded_mask


    def apply_mask_shade(self, image, mask_255, padded_mask, closest_point, dynamic_points):
        """
        Apply overlay of water to the mask.
        Apply padded colour area to the mask.
        Add a circle to the center of the mask and the closest safe point.

        """
        image = np.asarray(image)

        water_color = (255, 0, 0)
        water_padding_color = (255, 178, 0)
        dynamic_color = (0, 150, 100)
        point_color = (255, 255, 255)
        color_overlay = np.full_like(image, water_color, dtype=np.uint8)

        alpha = 0.3
        mask_area = mask_255 == 255
        blended_image = np.where(
            mask_area[..., None],  # Expand dimensions for broadcasting
            cv2.addWeighted(image, 1 - alpha, color_overlay, alpha, 0),
            image,
        )

        padded_area = padded_mask > mask_255
        blended_image[dynamic_points] = dynamic_color
        blended_image[padded_area] = water_padding_color
        center = mask_255.shape[1] // 2, mask_255.shape[0] // 2

        cv2.circle(blended_image, center, radius=10, color=(255, 255, 255), thickness=-1)
        cv2.circle(blended_image, center, radius=50, color=(255, 255, 255), thickness=10)
        cv2.circle(blended_image, closest_point, radius=10, color=point_color, thickness=-1)
        cv2.circle(blended_image, closest_point, radius=50, color=point_color, thickness=10)

        cv2.putText(
            blended_image,
            "Centre",
            (center[0] - 160, center[1] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 255),
            10,
        )
        cv2.putText(
            blended_image,
            "Safe point",
            (closest_point[0] - 100, closest_point[1] + 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            point_color,
            10,
        )

        return blended_image


    def define_safe_mask(self, detections, mask_255, use_dynamic_points=False):
        floating_obj_mask, human_mask = self.bbox_to_mask(detections, mask_255)
        safe_mask = mask_255.copy()
        human_points = (mask_255 == 0) & (human_mask == 0)
        safe_mask[human_points] = 255
        dynamic_points = (mask_255 == 0) & (floating_obj_mask == 0)

        if not use_dynamic_points:
            safe_mask[dynamic_points] = 255

        return safe_mask, dynamic_points



    def find_safepoints(self, 
                        img_path, 
                        padding_factor=20, 
                        threshold=0.5, 
                        show_results=False, 
                        use_dynamic_points=True
                        ):
        img_raw = Image.open(img_path)
        mask = predict_image(self.seg_model, img_raw, self.device)
        mask_255 = mask.astype(np.uint8) * 255
        detections = self.detect_objects(img_path, threshold=threshold)


        safe_mask, dynamic_points = self.define_safe_mask(
            detections, mask_255, use_dynamic_points=use_dynamic_points
        )
        start_time = time.time()
        padded_mask = self.apply_padding(safe_mask, padding_factor)

        closest_point = self.find_closest_safe_point(padded_mask)
        end_time = time.time()
        print(f"Time to find safe point: {end_time - start_time}")

        if show_results:
            processed_mask = self.draw_processed_mask(
                safe_mask, padded_mask, dynamic_points
            )
            shaded_image = self.apply_mask_shade(
                img_raw, safe_mask, padded_mask, closest_point, dynamic_points
            )

            self.show_results(img_raw, mask_255, safe_mask, processed_mask, shaded_image)

        return closest_point, mask_255

        
if __name__ == "__main__":
    seg_path = "/home/emilia/WaterSegNet/checkpoints/checkpoints_padded/model-UnetPlusPlus_resnet34_adam_b16.ckpt"
    det_path = "/home/emilia/msc_ros2/ML/checkpoints/search_nano_model.pt"
    test_img = "/home/emilia/msc_ros2/ML/src/Segmentation/test/trondheim.jpg"
    spf = SafePointFinder(seg_path, det_path)
    spf.find_safepoints(test_img, show_results=True)