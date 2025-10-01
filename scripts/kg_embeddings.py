#!/usr/bin/env python3
"""
Knowledge Graph Embedding Models
===============================

This module provides implementations of Knowledge Graph embedding models
for recommendation systems, including TransE and DistMult.

SUPPORTED MODELS:
- TransE: Translational embedding for knowledge graphs
- DistMult: Multiplicative embedding for knowledge graphs

USAGE:
    from kg_embeddings import TransEModel, train_transe_model
    
    # Train TransE embeddings
    user_embeddings, item_embeddings = train_transe_model(
        data_dir, user_mapping, item_mapping, device, epochs=20
    )

This module is designed to be imported and used by the main training script
for generating Knowledge Graph embeddings that enhance recommendation models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

class TransEModel(nn.Module):
    """
    TransE (Translating Embeddings) model for Knowledge Graph embeddings.
    
    TransE learns entity and relation embeddings by representing relationships
    as translations in the embedding space: h + r ≈ t
    
    Args:
        num_entities (int): Number of entities in the knowledge graph
        num_relations (int): Number of relations in the knowledge graph
        embedding_dim (int): Dimension of the embedding vectors (default: 768)
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim=768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        # TransE embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings using Xavier uniform"""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, head, relation, tail):
        """
        TransE forward pass: h + r ≈ t
        Args:
            head: [batch_size] - head entity indices
            relation: [batch_size] - relation indices  
            tail: [batch_size] - tail entity indices
        Returns:
            score: [batch_size] - TransE scores (lower is better)
        """
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        # TransE score: ||h + r - t||_2
        score = torch.norm(h + r - t, p=2, dim=1)
        return score
    
    def get_entity_embeddings(self):
        """Get all entity embeddings"""
        return self.entity_embeddings.weight.data
    
    def get_relation_embeddings(self):
        """Get all relation embeddings"""
        return self.relation_embeddings.weight.data

def train_transe_model(data_dir, user_mapping, item_mapping, device, epochs=10):
    """
    Train TransE model on Knowledge Graph triples for recommendation systems.
    
    This function loads Knowledge Graph triples from the data directory,
    trains a TransE model to learn entity and relation embeddings,
    and returns user and item embeddings for use in recommendation models.
    
    Args:
        data_dir (str): Path to data directory containing KG triples
        user_mapping (dict): User ID to index mapping
        item_mapping (dict): Item ID to index mapping  
        device (torch.device): Device to use for training (cpu/cuda)
        epochs (int): Number of training epochs (default: 10)
    
    Returns:
        tuple: (user_embeddings, item_embeddings) as PyTorch tensors
    """
    print("🧠 Training TransE model...")
    
    # Load interaction data for training (only train split)
    interactions = pd.read_csv(os.path.join(data_dir, 'train_split.tsv'), sep='\t', header=None,
                              names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    # Create triples from interactions
    triples = []
    for _, row in interactions.iterrows():
        user_id = user_mapping[row['user_id']]
        item_id = item_mapping[row['item_id']]
        rating = row['rating']
        
        if rating >= 4:  # Positive interaction
            triples.append((user_id, 0, item_id))  # relation 0 = likes
        else:  # Negative interaction
            triples.append((user_id, 1, item_id))  # relation 1 = dislikes
    
    print(f"📊 Using {len(triples)} triples for TransE training (full dataset)")
    
    # Initialize TransE model
    num_entities = max(len(user_mapping), len(item_mapping))
    transe_model = TransEModel(num_entities, 2, embedding_dim=512).to(device)
    
    # Training setup
    optimizer = optim.Adam(transe_model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop with batch processing for speed
    print(f"🔄 Training TransE for {epochs} epochs...")
    batch_size = 1000  # Process 1000 triples at once
    
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(triples)
        
        # Process in batches
        num_batches = len(triples) // batch_size
        pbar = tqdm(range(num_batches), desc=f"TransE Epoch {epoch+1}/{epochs}")
        
        for batch_idx in pbar:
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(triples))
            batch_triples = triples[start_idx:end_idx]
            
            # Convert to tensors
            head = torch.tensor([t[0] for t in batch_triples], dtype=torch.long, device=device)
            relation = torch.tensor([t[1] for t in batch_triples], dtype=torch.long, device=device)
            tail = torch.tensor([t[2] for t in batch_triples], dtype=torch.long, device=device)
            
            # Forward pass
            positive_scores = transe_model(head, relation, tail)
            
            # Create negative samples (corrupt tail entities)
            negative_tail = torch.randint(0, num_entities, (len(batch_triples),), device=device)
            negative_scores = transe_model(head, relation, negative_tail)
            
            # Loss: positive scores should be low, negative scores should be high
            positive_labels = torch.zeros_like(positive_scores)
            negative_labels = torch.ones_like(negative_scores)
            
            loss = criterion(-positive_scores, positive_labels) + criterion(-negative_scores, negative_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"✅ TransE Epoch {epoch+1}/{epochs} completed - Avg Loss: {avg_loss:.4f}")
    
    print("✅ TransE training completed!")
    
    # Extract user and item embeddings
    all_embeddings = transe_model.get_entity_embeddings()
    
    # Extract user embeddings
    user_indices = list(user_mapping.values())
    user_embeddings = all_embeddings[user_indices]
    
    # Extract item embeddings
    item_indices = list(item_mapping.values())
    item_embeddings = all_embeddings[item_indices]
    
    return user_embeddings, item_embeddings
