import os, numpy as np, torch, torch.nn.functional as F
import cv2
from PIL import Image
import torchvision.transforms as transforms
from config import cfg


class GradCAMVisualizer:
    def __init__(self, model, target_layer, height=14, width=14, device=None, val_transform=None, class_names=None,
                 dataset_type=None):
        self.model = model
        self.target_layer = target_layer
        self.height = cfg["gradcam"]["height"]
        self.width = cfg["gradcam"]["width"]
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.val_transform = val_transform
        self.class_names = class_names or cfg["dataset"]["classes"]
        self.dataset_type = dataset_type or cfg["dataset"]["name"]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        if self.dataset_type:
            os.makedirs(f'./gradcam/{self.dataset_type}', exist_ok=True)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def reshape_transform(self, tensor):
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                          self.height, self.width,
                                          tensor.size(2))
        result = result.permute(0, 3, 1, 2)
        return result

    def generate_cam(self, input_image, class_idx=None):
        self.model.eval()
        input_image.requires_grad_(True)
        output = self.model(input_image)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        acts = self.reshape_transform(self.activations)
        grads = self.reshape_transform(self.gradients)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_min = cam.view(cam.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = F.interpolate(cam, size=(input_image.shape[2], input_image.shape[3]),
                            mode='bilinear', align_corners=False)
        return cam[0, 0].cpu().numpy()

    def generate_gradcam_heatmap(self, image, class_idx=None, save_path=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_image = transform(Image.fromarray(image)).unsqueeze(0).to(self.device)
        cam = self.generate_cam(input_image, class_idx)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)
        superimposed_img = heatmap * cfg["gradcam"]["alpha"] + image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        if save_path:
            cv2.imwrite(save_path, superimposed_img)
        return superimposed_img

    def generate_class_gradcams(self, test_images, test_labels, num_classes):
        self.model.to(self.device)
        self.model.eval()
        for cls in range(num_classes):
            indices = np.where(test_labels == cls)[0]
            if len(indices) == 0:
                continue
            idx = indices[0]
            original_image = test_images[idx]
            image_pil = Image.fromarray(original_image)
            input_tensor = self.val_transform(image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                pred_cls = torch.argmax(output, dim=1).item()
            save_path = f'./gradcam/{self.dataset_type}/class_{cls}_pred_{pred_cls}_{self.class_names[cls]}_heatmap_tissue_overlay.png'
            superimposed_img = self.generate_gradcam_heatmap(
                original_image,
                class_idx=cls,
                save_path=save_path
            )

# if generate_gradcams:
#    target_layer = model.encoder.layers[-1].ln_1
#    class_names = ['benign', 'insitu', 'invasive', 'normal']
#    visualizer = GradCAMVisualizer(model, target_layer, device=device, val_transform=val_transform, class_names=class_names, dataset_type=dataset_type)
#    visualizer.generate_class_gradcams(test_images, test_labels, num_classes)
