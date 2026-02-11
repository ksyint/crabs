import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BackboneKeypointExtractor(nn.Module):
    def __init__(self, backbone, descriptor_dim=256, max_num_keypoints=1024, detection_threshold=0.01):
        super().__init__()
        self.backbone = backbone
        self.descriptor_dim = descriptor_dim
        self.max_num_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold

        self._build_feature_layers()

    def _build_feature_layers(self):
        if hasattr(self.backbone, 'backbone'):
            resnet = self.backbone.backbone
        elif hasattr(self.backbone, 'model'):
            resnet = self.backbone.model
        else:
            resnet = self.backbone

        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        feat_channels = 1024
        self.score_head = nn.Sequential(
            nn.Conv2d(feat_channels, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid(),
        )

        self.desc_head = nn.Sequential(
            nn.Conv2d(feat_channels, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.descriptor_dim, 1),
        )

    def _extract_features(self, image):
        x1 = self.layer1(image)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x3

    def _detect_keypoints(self, score_map, image_shape):
        b, _, h, w = score_map.shape
        score_map_flat = score_map.view(b, -1)

        num_kpts = min(self.max_num_keypoints, h * w)
        topk_scores, topk_indices = torch.topk(score_map_flat, num_kpts, dim=1)

        mask = topk_scores > self.detection_threshold
        keypoints_list = []
        scores_list = []

        for i in range(b):
            valid = mask[i]
            indices = topk_indices[i][valid]
            scores = topk_scores[i][valid]

            kpts_y = (indices // w).float()
            kpts_x = (indices % w).float()

            scale_x = image_shape[-1] / w
            scale_y = image_shape[-2] / h
            kpts_x = kpts_x * scale_x
            kpts_y = kpts_y * scale_y

            keypoints_list.append(torch.stack([kpts_x, kpts_y], dim=-1))
            scores_list.append(scores)

        return keypoints_list, scores_list

    def _sample_descriptors(self, descriptor_map, keypoints, image_shape):
        b, c, h, w = descriptor_map.shape
        descriptors_list = []

        for i in range(b):
            kpts = keypoints[i]
            if len(kpts) == 0:
                descriptors_list.append(torch.zeros(0, c, device=descriptor_map.device))
                continue

            grid_x = 2.0 * kpts[:, 0] / max(image_shape[-1] - 1, 1) - 1.0
            grid_y = 2.0 * kpts[:, 1] / max(image_shape[-2] - 1, 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)

            descs = F.grid_sample(
                descriptor_map[i:i+1], grid, mode="bilinear", align_corners=True
            )
            descs = descs.reshape(c, -1).t()
            descs = F.normalize(descs, p=2, dim=-1)
            descriptors_list.append(descs)

        return descriptors_list

    def forward(self, data):
        image = data["image"]
        features = self._extract_features(image)

        score_map = self.score_head(features)
        descriptor_map = self.desc_head(features)

        keypoints, scores = self._detect_keypoints(score_map, image.shape)
        descriptors = self._sample_descriptors(descriptor_map, keypoints, image.shape)

        max_n = max(len(k) for k in keypoints) if keypoints else 0
        if max_n == 0:
            max_n = 1

        b = image.shape[0]
        padded_kpts = torch.zeros(b, max_n, 2, device=image.device)
        padded_scores = torch.zeros(b, max_n, device=image.device)
        padded_descs = torch.zeros(b, max_n, self.descriptor_dim, device=image.device)

        for i in range(b):
            n = len(keypoints[i])
            if n > 0:
                padded_kpts[i, :n] = keypoints[i]
                padded_scores[i, :n] = scores[i]
                padded_descs[i, :n] = descriptors[i]

        return {
            "keypoints": padded_kpts,
            "keypoint_scores": padded_scores,
            "descriptors": padded_descs,
        }
