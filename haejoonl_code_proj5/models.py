import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # Shared MLP over points
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # Global feature → classifier
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = points.permute(0, 2, 1)  # (B, 3, N)

        x = F.relu(self.bn1(self.conv1(x)))    # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))    # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))    # (B, 1024, N)

        # Symmetric function: max pooling
        x = torch.max(x, 2)[0]                 # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))      # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))      # (B, 256)
        x = self.dropout(x)
        x = self.fc3(x)                        # (B, num_classes)

        # CrossEntropyLoss expects raw logits
        return x



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super(seg_model, self).__init__()

        # Encoder
        self.conv0 = nn.Conv1d(3, 64, 1)
        self.bn0 = nn.BatchNorm1d(64)

        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # Decoder: 64(local) + 1024(global) = 1088
        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(512, 256, 1)
        self.bn5 = nn.BatchNorm1d(256)

        self.conv6 = nn.Conv1d(256, 128, 1)
        self.bn6 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)
        self.conv7 = nn.Conv1d(128, num_seg_classes, 1)

    def forward(self, points):
        """
        points: (B, N, 3)
        returns: (B, N, num_seg_classes) logits
        """
        x = points.permute(0, 2, 1)  # (B, 3, N)

        x0 = F.relu(self.bn0(self.conv0(x)))     # (B, 64, N)
        x1 = F.relu(self.bn1(self.conv1(x0)))    # (B, 64, N)
        x2 = F.relu(self.bn2(self.conv2(x1)))    # (B, 128, N)
        x3 = F.relu(self.bn3(self.conv3(x2)))    # (B, 1024, N)

        # Global feature
        x_global = torch.max(x3, 2, keepdim=True)[0]   # (B, 1024, 1)
        x_global = x_global.expand(-1, -1, x.size(2))  # (B, 1024, N)

        # Match GitHub: concat 64-d local (x1) with 1024-d global → 1088
        x_concat = torch.cat([x1, x_global], dim=1)    # (B, 1088, N)

        x = F.relu(self.bn4(self.conv4(x_concat)))     # (B, 512, N)
        x = F.relu(self.bn5(self.conv5(x)))            # (B, 256, N)
        x = self.dropout(x)
        x = F.relu(self.bn6(self.conv6(x)))            # (B, 128, N)
        x = self.conv7(x)                              # (B, num_seg_classes, N)

        return x.permute(0, 2, 1)                      # (B, N, num_seg_classes)



# =========================================================
# Helper functions for PointNet++-style locality
# =========================================================

def index_points(points, idx):
    """
    Gather points by index.

    points: (B, N, C)
    idx: (B, S, K) or (B, S) indices into N

    Returns:
        grouped_points: (B, S, K, C) if idx is (B,S,K)
                        (B, S, C)     if idx is (B,S)
    """
    B, N, C = points.shape
    if idx.dim() == 2:
        # (B, S)
        S = idx.shape[1]
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)  # (B,S,C)
        # gather along N dimension
        points_expanded = points  # (B,N,C)
        grouped = torch.gather(points_expanded, 1, idx_expanded)
        return grouped  # (B,S,C)
    elif idx.dim() == 3:
        # (B, S, K)
        B, S, K = idx.shape
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # (B,S,K,C)
        points_expanded = points.unsqueeze(1).expand(-1, S, -1, -1)  # (B,S,N,C)
        grouped = torch.gather(points_expanded, 2, idx_expanded)  # (B,S,K,C)
        return grouped
    else:
        raise ValueError("idx must be 2D or 3D")


def farthest_point_sample(xyz, npoint):
    """
    Farthest point sampling (FPS) in pure PyTorch.

    xyz: (B, N, 3)
    npoint: number of points to sample

    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B,1,3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B,N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]

    return centroids  # (B, npoint)


def knn_point(k, xyz, new_xyz):
    """
    k-NN grouping.

    xyz: (B, N, 3)  - all points
    new_xyz: (B, S, 3) - center points

    Returns:
        idx: (B, S, k) indices of k nearest neighbors in xyz for each new_xyz
    """
    # pairwise distances: (B, S, N)
    dists = torch.cdist(new_xyz, xyz)  # uses efficient CUDA kernel if available
    idx = dists.topk(k, dim=-1, largest=False)[1]  # (B,S,k)
    return idx


class PointNetSetAbstraction(nn.Module):
    """
    Simplified PointNet++ Set Abstraction (single-scale grouping).

    - Samples npoint centers (via FPS)
    - For each center, groups k neighbors via k-NN
    - Applies an MLP (as 1x1 convs) on grouped points
    - Max-pools over neighbors to get a feature per center
    """

    def __init__(self, npoint, k, in_channels, mlp_channels):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.k = k

        layers = []
        last_c = in_channels
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for out_c in mlp_channels:
            conv = nn.Conv2d(last_c, out_c, kernel_size=1)
            bn = nn.BatchNorm2d(out_c)
            self.convs.append(conv)
            self.bns.append(bn)
            last_c = out_c

    def forward(self, xyz, points=None):
        """
        xyz: (B, N, 3)
        points: (B, N, C_in) or None

        Returns:
            new_xyz: (B, S, 3) sampled centers
            new_points: (B, S, C_out) features
        """
        B, N, _ = xyz.shape
        if self.npoint is not None:
            # Sample centers with FPS
            fps_idx = farthest_point_sample(xyz, self.npoint)  # (B,S)
            new_xyz = index_points(xyz, fps_idx)               # (B,S,3)
        else:
            # No sampling: treat all points as centers
            new_xyz = xyz
            fps_idx = None
            self.npoint = N

        # k-NN grouping from centers to original points
        idx = knn_point(self.k, xyz, new_xyz)  # (B,S,k)
        grouped_xyz = index_points(xyz, idx)   # (B,S,k,3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # relative coords

        if points is not None:
            grouped_points = index_points(points, idx)  # (B,S,k,C_in)
            # concat XYZ-relative + features
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B,S,k,3+C_in)
        else:
            grouped_points = grouped_xyz  # (B,S,k,3)

        # (B,S,k,C) -> (B,C,S,k)
        grouped_points = grouped_points.permute(0, 3, 1, 2)

        # MLP over grouped points
        for conv, bn in zip(self.convs, self.bns):
            grouped_points = F.relu(bn(conv(grouped_points)))

        # Max over neighbors (k)
        new_points = torch.max(grouped_points, dim=-1)[0]  # (B,C_out,S)
        new_points = new_points.permute(0, 2, 1)          # (B,S,C_out)

        return new_xyz, new_points


class PointNetLocalAggregation(nn.Module):
    """
    Local neighborhood aggregation without downsampling.
    This is used for segmentation: keeps N points, but for each point
    aggregates its k neighbors with a PointNet-style MLP.
    """

    def __init__(self, k, in_channels, mlp_channels):
        super(PointNetLocalAggregation, self).__init__()
        self.k = k
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        last_c = in_channels
        for out_c in mlp_channels:
            self.convs.append(nn.Conv2d(last_c, out_c, kernel_size=1))
            self.bns.append(nn.BatchNorm2d(out_c))
            last_c = out_c

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3)
        points: (B, N, C_in)

        Returns:
            new_points: (B, N, C_out)
        """
        B, N, _ = xyz.shape
        # Use each point as center
        idx = knn_point(self.k, xyz, xyz)  # (B,N,k)
        grouped_xyz = index_points(xyz, idx)   # (B,N,k,3)
        grouped_xyz = grouped_xyz - xyz.unsqueeze(2)  # relative coords

        grouped_points = index_points(points, idx)  # (B,N,k,C_in)
        # concat local coords + features
        grouped = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B,N,k,3+C_in)

        # (B,N,k,C) -> (B,C,N,k)
        grouped = grouped.permute(0, 3, 1, 2)

        for conv, bn in zip(self.convs, self.bns):
            grouped = F.relu(bn(conv(grouped)))

        # Max over neighbors (k)
        new_points = torch.max(grouped, dim=-1)[0]  # (B,C_out,N)
        new_points = new_points.permute(0, 2, 1)    # (B,N,C_out)
        return new_points


# =========================================================
#  PointNet++-style Classification Model
# =========================================================

class cls_model_pointnet2(nn.Module):
    def __init__(self, num_classes=3, k=32):
        super().__init__()

        # SA1: input = xyz + xyz = 6 channels
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            k=k,
            in_channels=6,                # FIXED
            mlp_channels=[64, 64, 128]
        )

        # SA2: after concat: xyz(3) + feat(128) = 131 → grouped: 3 + 131 = 134
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            k=k,
            in_channels=134,              # FIXED
            mlp_channels=[128, 256, 512]
        )

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, points):
        xyz = points  # (B,N,3)

        # SA1: pass xyz as both input coords AND initial features
        l1_xyz, l1_points = self.sa1(xyz, points=xyz)   # (B,512,3), (B,512,128)

        # concat xyz and features
        sa2_input = torch.cat([l1_xyz, l1_points], dim=-1)  # (B,512,131)

        # SA2
        l2_xyz, l2_points = self.sa2(l1_xyz, sa2_input)     # (B,128,3), (B,128,512)

        # Global pooling
        x = torch.max(l2_points, dim=1)[0]  # (B,512)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# =========================================================
#  PointNet++-style Segmentation Model
# =========================================================

class seg_model_pointnet2(nn.Module):
    """
    Simplified PointNet++ segmentation:
      - Per-point MLP -> local k-NN aggregation (xyz + features)
      - Global feature + per-point decoder

    Input:  (B, N, 3)
    Output: (B, N, num_seg_classes) logits
    """
    def __init__(self, num_seg_classes=6, k=16):
        super(seg_model_pointnet2, self).__init__()

        # Per-point MLP
        self.conv0 = nn.Conv1d(3, 64, 1)
        self.bn0 = nn.BatchNorm1d(64)

        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        # Local aggregation over k-NN neighborhoods.
        # Per-point features have C_in=64, we will concat xyz inside -> 3+64=67.
        self.local_agg = PointNetLocalAggregation(
            k=k,
            in_channels=67,          # 3 (xyz) + 64 (features)
            mlp_channels=[64, 128, 128]
        )

        # Decoder: local(128) + global(128) = 256
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.4)
        self.conv4 = nn.Conv1d(128, num_seg_classes, 1)

    def forward(self, points):
        """
        points: (B, N, 3)
        returns: (B, N, num_seg_classes)
        """
        B, N, _ = points.shape
        xyz = points  # (B,N,3)

        x = points.permute(0, 2, 1)           # (B,3,N)
        x = F.relu(self.bn0(self.conv0(x)))   # (B,64,N)
        x = F.relu(self.bn1(self.conv1(x)))   # (B,64,N)

        point_feats = x.permute(0, 2, 1)      # (B,N,64)

        # ✅ Pass only features. LocalAggregation adds xyz internally.
        local_feats = self.local_agg(xyz, point_feats)   # (B,N,128)

        # Global feature
        global_feat = torch.max(local_feats, dim=1, keepdim=True)[0]  # (B,1,128)
        global_feat = global_feat.repeat(1, N, 1)                     # (B,N,128)

        # Fuse local + global
        feat = torch.cat([local_feats, global_feat], dim=-1)  # (B,N,256)
        feat = feat.permute(0, 2, 1)                          # (B,256,N)

        feat = F.relu(self.bn2(self.conv2(feat)))             # (B,256,N)
        feat = F.relu(self.bn3(self.conv3(feat)))             # (B,128,N)
        feat = self.dropout(feat)
        feat = self.conv4(feat)                               # (B,num_seg_classes,N)

        return feat.permute(0, 2, 1)                          # (B,N,num_seg_classes)

