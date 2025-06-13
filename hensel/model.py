import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        attention_weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(attention_weights * x, dim=1)

class HenselModel(nn.Module):
    def __init__(self, vector_size_title, vector_size_num, scale, 
                 domain_vocab_size, tld_vocab_size, user_vocab_size,
                 hidden_size=256, dropout_rate=0.2):
        super().__init__()
        
        # Embedding layers
        self.domain_embedding = nn.Embedding(domain_vocab_size, hidden_size)
        self.tld_embedding = nn.Embedding(tld_vocab_size, hidden_size)
        self.user_embedding = nn.Embedding(user_vocab_size, hidden_size)
        
        # Feature processing layers
        self.title_processor = nn.Sequential(
            nn.Linear(vector_size_title, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.numeric_processor = nn.Sequential(
            nn.Linear(vector_size_num, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention layers
        self.title_attention = AttentionLayer(hidden_size, hidden_size // 2)
        self.numeric_attention = AttentionLayer(hidden_size, hidden_size // 2)
        
        # Feature fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size * 2),  # 5 inputs: title, numeric, domain, tld, user
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, features_num, title_emb, domain_idx, tld_idx, user_idx):
        # Process embeddings
        domain_emb = self.domain_embedding(domain_idx)
        tld_emb = self.tld_embedding(tld_idx)
        user_emb = self.user_embedding(user_idx)
        
        # Process title and numeric features
        title_processed = self.title_processor(title_emb)
        numeric_processed = self.numeric_processor(features_num)
        
        # Apply attention
        title_attended = self.title_attention(title_processed.unsqueeze(1))
        numeric_attended = self.numeric_attention(numeric_processed.unsqueeze(1))
        
        # Concatenate all features
        combined = torch.cat([
            title_attended,
            numeric_attended,
            domain_emb,
            tld_emb,
            user_emb
        ], dim=1)
        
        # Fusion and output
        fused = self.fusion_layer(combined)
        output = self.output_layers(fused)
        
        return output 