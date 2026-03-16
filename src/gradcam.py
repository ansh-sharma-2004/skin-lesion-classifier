import torch
import torch.nn.functional as F
import numpy as np
import cv2

LABEL_MAP_INV = {
    0: 'Nevus',
    1: 'Pigmented benign keratosis',
    2: 'Melanoma, NOS',
    3: 'Basal cell carcinoma',
    4: 'Squamous cell carcinoma, NOS',
    5: 'Dermatofibroma',
    6: 'Solar or actinic keratosis'
}

class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Hook into the last conv block of EfficientNet
        target_layer = model.blocks[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx=None):
        self.model.eval()

        # Forward pass
        output = self.model(image_tensor)

        # Use predicted class if none specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for specific class
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=[2, 3])  # (1, C)

        # Weighted sum of activation maps
        cam = torch.zeros(self.activations.shape[2:])  # (H, W)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]

        # ReLU + normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.numpy(), class_idx

def apply_heatmap(original_image, cam, alpha=0.4):
    """
    Overlay GradCAM heatmap on original image
    original_image: numpy array (H, W, 3) RGB uint8
    cam: numpy array (H, W) float 0-1
    """
    # Resize cam to match image
    h, w  = original_image.shape[:2]
    cam   = cv2.resize(cam, (w, h))

    # Apply colormap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    return overlaid

def predict_and_explain(model, image_tensor, original_image):
    """
    Full pipeline - returns prediction, confidence, and heatmap
    image_tensor: preprocessed tensor ready for model
    original_image: raw numpy array for visualization
    """
    grad_cam = GradCAM(model)
    cam, class_idx = grad_cam.generate(image_tensor)

    # Get probabilities
    with torch.no_grad():
        output = model(image_tensor)
        probs  = F.softmax(output, dim=1)[0]

    confidence  = probs[class_idx].item()
    label       = LABEL_MAP_INV[class_idx]
    heatmap_img = apply_heatmap(original_image, cam)

    return {
        'label'      : label,
        'confidence' : confidence,
        'class_idx'  : class_idx,
        'probs'      : probs.numpy(),
        'heatmap'    : heatmap_img
    }