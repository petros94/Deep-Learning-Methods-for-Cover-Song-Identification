import json
import torch
import random

from datasets.SimpleDataset import SimpleDataset
from utils.generic import get_device
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from pytorch_metric_learning import miners, losses, distances
from datasets.TripletDataset import TripletDataset

def train_embeddings_loss(model: torch.nn.Module,
                            train_set: SimpleDataset,
                            valid_set: SimpleDataset,
                            n_epochs: int, 
                            patience: int, 
                            lr: float, 
                            checkpoints_path: str, 
                            results_path: str):

    device = get_device()
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    distance = distances.LpDistance(normalize_embeddings=True)    
    loss_func = losses.AngularLoss(distance=distance)
    miner = miners.AngularMiner(angle=20, distance=distance)
    valid_miner = miners.AngularMiner(angle=0, distance=distance)

    train_emb, train_labels = calculate_embeddings(train_set)
    valid_emb, valid_labels = calculate_embeddings(valid_set)
    
    best_loss = 1000000
    current_patience = 0
    
    losses_history = {
        'epoch': [],
        'train': [],
        'valid': []
    }
    
    for epoch in range(n_epochs):
        
        print(32*"=")
        print(f"Epoch {epoch}")
        model.train()
        epoch_loss = 0
        mean_triplets = 0


        # N X 1 X num_feats X num_samples, N
        data = train_emb.to(device)
        labels = train_labels.to(device)

        optimizer.zero_grad()
        embeddings = model(data)
        triplets = miner(embeddings, labels)

        loss = loss_func(embeddings, labels, triplets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        mean_triplets += miner.num_triplets

        print('Evaluating model')
        model.eval()
        valid_loss=0
        with torch.no_grad():
            data = valid_emb.to(device)
            labels = valid_labels.to(device)

            embeddings = model(data)
            triplets = valid_miner(embeddings, labels)

            loss = loss_func(embeddings, labels, triplets)
            valid_loss += loss.item()
        
        if valid_loss < best_loss:
            print("New best random loss, saving model")
            best_loss = valid_loss
            current_patience = 0
        
            # Export best model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                }, checkpoints_path + "/checkpoint.tar")
        
            # Save results
            with open(results_path + '/train_results.json', "w") as f:
                json.dump({'n_epochs': epoch, 'valid_loss': valid_loss, 'train_loss': epoch_loss}, f)
        else:
            current_patience +=1
        
        print(f"Epoch {epoch} random triplet valid loss: {valid_loss}")
            # test(train_set=train_set, valid_set=valid_set, model=model, accuracy_calculator=acc_calc)            
        print(f"Epoch {epoch} train loss: {epoch_loss}, mean triplets: {int(float(mean_triplets))}, perf score: {epoch_loss*int(float(mean_triplets))}")
        losses_history['train'].append(epoch_loss)
        losses_history['valid'].append(valid_loss)
        losses_history['epoch'].append(epoch)
        
        if current_patience >= patience:
            print(f"No further improvement after {patience} epochs, breaking.")
            break
        
    # Load best model
    checkpoint = torch.load(checkpoints_path + "/checkpoint.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return losses_history

def calculate_embeddings(dataset):
    device = get_device()
    model_name_or_path = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
    wav2vec_model_config = AutoConfig.from_pretrained(model_name_or_path)
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    wav2vec_pretrained_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name_or_path,
                                                                              output_hidden_states=True).to(device)

    for param in wav2vec_pretrained_model.parameters():
        param.requires_grad = False

    embeddings = []
    labels = []
    for i in range(len(dataset)):
        frames, label, song_name = dataset[i]
        frames = torch.tensor(frames)
        print(frames.size())

        inputs = wav2vec_feature_extractor(frames, sampling_rate=16000, return_tensors="pt", padding=False)
        with torch.no_grad():
            wav2vec_output = wav2vec_pretrained_model(**inputs)
            last_hidden_layer_emb = torch.mean(wav2vec_output.hidden_states[-1], dim=1)
            # last_conv_layer_emb = self.pretrained_model(**inputs).extract_features
            embeddings.append(last_hidden_layer_emb)
            labels.append(label)

    return torch.stack(embeddings, dim=0), torch.tensor(labels)
