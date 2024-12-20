# model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(
            self, 
            num_features,
            num_classes,
            num_hidden_layers=1,
            hid=12,
            in_head=3,
            out_head=3,
            dropout=0.41
        ):
        super(GAT, self).__init__()
        
        # List to hold all GATConv layers
        self.layers = torch.nn.ModuleList()
        
        # First layer from input features to hidden dimension
        self.layers.append(GATConv(num_features, hid, heads=in_head, dropout=dropout))
        
        # Add hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(GATConv(hid * in_head, hid, heads=in_head, dropout=dropout))
        
        # Last layer from hidden dimension to output classes
        self.layers.append(GATConv(hid * in_head, num_classes, concat=False, heads=out_head, dropout=dropout))
        
        # Learnable class-specific attention weights
        self.class_attention_weights = torch.nn.Parameter(torch.ones(num_classes, 1))
    
    def forward(self, data, return_attention=False):
        x, edge_index = data.x, data.edge_index
        
        # Apply dropout to the input features
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Attention list to collect attention weights
        attention_weights = []
        
        # Pass through each layer except the last
        for layer in self.layers[:-1]:
            if return_attention:
                x, (edges, attn) = layer(x, edge_index, return_attention_weights=True)
                attention_weights.append((edges, attn))
            else:
                x = layer(x, edge_index)
            
            x = F.elu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Pass through the final layer without activation and dropout afterwards
        if return_attention:
            x, (edges, attn) = self.layers[-1](x, edge_index, return_attention_weights=True)
            attention_weights.append((edges, attn))
        else:
            x = self.layers[-1](x, edge_index)
        
        # Apply class-specific attention weights
        class_attention = self.class_attention_weights[data.y].view(-1, 1)
        x = x * class_attention
        
        # Return logits and attention weights if requested
        if return_attention:
            return F.log_softmax(x, dim=1), attention_weights
        else:
            return F.log_softmax(x, dim=1)
        
def initialize_gat_model(num_features, num_classes, hid, in_head, out_head, dropout_rate, num_hidden_layers=2):
    return GAT(
        num_features=num_features,
        num_classes=num_classes,
        num_hidden_layers=num_hidden_layers,
        hid=hid,
        in_head=in_head,
        out_head=out_head,
        dropout=dropout_rate)