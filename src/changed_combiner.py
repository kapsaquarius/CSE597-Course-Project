import torch
from torch import nn
import torch.nn.functional as F


class Combiner(nn.Module):
    """
    Combiner module which fuses textual and visual information.
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int, dropout_rate=0.5):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        :param dropout_rate: dropout probability
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Add multi-head attention for feature interaction
        self.multihead_attention = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=4, batch_first=True)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        # Layer normalization for stability
        self.layer_norm1 = nn.LayerNorm(projection_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.dropout3 = nn.Dropout(dropout_rate)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Make logit scale learnable
        self.logit_scale = nn.Parameter(torch.ones([]) * 100)

    def forward(self, image_features: torch.tensor, text_features: torch.tensor,
                target_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """
        predicted_features = self.combine_features(image_features, text_features)
        target_features = F.normalize(target_features, dim=-1)

        logits = self.logit_scale.exp() * predicted_features @ target_features.T
        return logits

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        text_projected_features = self.layer_norm1(self.dropout1(F.relu(self.text_projection_layer(text_features))))
        image_projected_features = self.layer_norm1(self.dropout2(F.relu(self.image_projection_layer(image_features))))

        # Attention mechanism
        combined_proj = torch.stack([text_projected_features, image_projected_features], dim=1)
        attn_output, _ = self.multihead_attention(combined_proj, combined_proj, combined_proj)
        attn_combined_features = attn_output.mean(dim=1)  # Average attention output

        # Concatenate the original and attention-enhanced features
        raw_combined_features = torch.cat((attn_combined_features, image_projected_features), dim=-1)

        combined_features = self.layer_norm2(self.dropout3(F.relu(self.combiner_layer(raw_combined_features))))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)

        # Add residual connections
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * image_features
        return F.normalize(output, dim=-1)
