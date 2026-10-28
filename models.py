
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainCNN(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Conv block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # Conv block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.rr_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(128 + 16, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, rr):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        feat = self.gap(x).squeeze(-1) # (B, 128)
        rr_feat = self.rr_fc(rr)       # (B, 16)
        combined = torch.cat([feat, rr_feat], dim=1) # (B, 144)

        h = F.relu(self.fc1(combined))
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits


class ECGAttentionCNN(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Conv block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # Conv block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        # Multi-head Self Attention
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(128)

        # Pooling & classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.rr_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(128 + 16, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, rr):
        # x shape: (B, 1, 252)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x))) # (B, 128, seq_len)

        x_perm = x.permute(0, 2, 1) # (B, seq_len, 128)
        attn_out, attn_weights = self.attn(x_perm, x_perm, x_perm)

        # Residual connection + LayerNorm
        x_norm = self.ln(x_perm + attn_out)

        # Back to (B, 128, seq_len)
        x_back = x_norm.permute(0, 2, 1)
        feat = self.gap(x_back).squeeze(-1) # (B, 128)

        rr_feat = self.rr_fc(rr)            # (B, 16)
        combined = torch.cat([feat, rr_feat], dim=1) # (B, 144)
        h = F.relu(self.fc1(combined))
        h = self.dropout(h)
        logits = self.fc2(h)

        return logits, attn_weights


class WideCNN(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        # Wider conv blocks
        self.conv1 = nn.Conv1d(1, 48, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(48)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(48, 96, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(96)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(96, 192, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(192)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.rr_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(192 + 16, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, rr):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        feat = self.gap(x).squeeze(-1)
        rr_feat = self.rr_fc(rr)
        combined = torch.cat([feat, rr_feat], dim=1)

        h = F.relu(self.fc1(combined))
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    dummy_x = torch.randn(4, 1, 252)
    dummy_rr = torch.randn(4, 2)

    plain = PlainCNN()
    attn = ECGAttentionCNN()
    wide = WideCNN()

    print(f"PlainCNN parameters:         {count_parameters(plain):,}")
    print(f"ECGAttentionCNN parameters:  {count_parameters(attn):,}")
    print(f"WideCNN (ablation) params:   {count_parameters(wide):,}")

    out_plain = plain(dummy_x, dummy_rr)
    out_attn, weights = attn(dummy_x, dummy_rr)
    out_wide = wide(dummy_x, dummy_rr)

    print(f"Plain output shape: {out_plain.shape}")
    print(f"Attn output shape:  {out_attn.shape}, weights shape: {weights.shape}")
    print(f"Wide output shape:  {out_wide.shape}")
