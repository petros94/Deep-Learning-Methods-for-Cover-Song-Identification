import json
import torch
import random

from utils.generic import get_device 

from pytorch_metric_learning import miners, losses, reducers, distances
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

def RandomMiner(embeddings, labels):
    anchors = []
    positives = []
    negatives = []
    for idx, label in enumerate(labels):
        
        anchor = idx
        pos_ids = torch.argwhere(labels == label).squeeze()
        pos = idx
        while pos == idx:
            pos = random.choice(pos_ids).item()
            
        neg_ids = torch.argwhere(labels != label).squeeze()
        neg = random.choice(neg_ids).item()
        
        anchors.append(anchor)
        positives.append(pos)
        negatives.append(neg)
        print(anchor, pos, neg)
        
    return (anchors, positives, negatives)
        

def train_hard_triplet_loss(model: torch.nn.Module, train_set, valid_set, n_epochs, patience, batch_size, lr, checkpoints_path, results_path):

    device = get_device()
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    margin = 1.00
    distance = distances.LpDistance(normalize_embeddings=False)
    batch_semihard_miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard", distance=distance)
    batch_all_miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="all", distance=distance)
    batch_hard_miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="hard", distance=distance)
    loss_func = losses.TripletMarginLoss(margin=margin, distance=distance)
    miner = batch_all_miner
    valid_miner = RandomMiner
    acc_calc = AccuracyCalculator(k=512)
    
    criterion = torch.nn.TripletMarginLoss()
    collate_fn_test = getattr(valid_set, "collate_fn", None)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_test)

    train_batches = len(train_set)
    valid_batches = len(valid_set)
    
    best_loss = 1000000
    current_patience = 0
    for epoch in range(n_epochs):
        mean_triplets = 0
        print(32*"=")
        print(f"Epoch {epoch}")
        epoch_loss = 0
        model.train()

        for i in range(len(train_set)):
            # N X 1 X num_feats X num_samples, N
            (data, labels) = train_set[i]
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(data)
            triplets = miner(embeddings, labels)
         
            loss = loss_func(embeddings, labels, triplets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i%16==0:
                mean_triplets += miner.num_triplets
                print(f'batch {i}/{train_batches}, loss: {loss.item()}, triplets: {miner.num_triplets}')

        if epoch%1==0:
            print('Evaluating model')
            model.eval()
            valid_loss=0
            with torch.no_grad():
                for i in range(len(valid_set)):
                    # N X 1 X num_feats X num_samples, N
                    (data, labels) = valid_set[i]
                    data = data.to(device)
                    
                    embeddings = model(data)
                    triplets = valid_miner(embeddings, labels)
                
                    loss = loss_func(embeddings, labels, triplets)
                    valid_loss += loss.item()
                    
                # for batch, (x, metadata) in enumerate(valid_dataloader):     
                
                #     (anchor, pos, neg) = x 

                #     anchor.to(device)
                #     pos.to(device)
                #     neg.to(device)

                #     anchor_out = model(anchor)
                #     pos_out = model(pos)
                #     neg_out = model(neg)

                #     loss = criterion(anchor_out, pos_out, neg_out)
                #     valid_loss += loss.item()
                    
                if valid_loss < best_loss:
                    print("New best random loss, saving model")
                    best_loss = valid_loss
                    current_patience = 0
                
                    # Export best model checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_loss/valid_batches,
                        }, checkpoints_path + "/checkpoint.tar")
                
                    # Save results
                    with open(results_path + '/train_results.json', "w") as f:
                        json.dump({'n_epochs': epoch, 'valid_loss': valid_loss/valid_batches, 'train_loss': epoch_loss/train_batches}, f)
                else:
                    current_patience +=1
                    
                print(f"Epoch {epoch} random triplet valid loss: {valid_loss/valid_batches}")
                
                
            if current_patience == patience:
                print(f"No further improvement after {patience} epochs, breaking.")
                break
        
            test(train_set=train_set, valid_set=valid_set, model=model, accuracy_calculator=acc_calc)            
        print(f"Epoch {epoch} train loss: {epoch_loss/train_batches}, mean triplets: {int(float(mean_triplets)/train_batches)}, perf score: {epoch_loss*int(float(mean_triplets)/train_batches)}")
        
    # Load best model
    checkpoint = torch.load(checkpoints_path + "/checkpoint.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    
def get_all_embeddings(data_set, model):
    device = get_device()
    embeddings = []
    with torch.no_grad():
        songs, labels = data_set.get_full_songs()
        for s in songs:
            # 1 X num_feats X num_samples, N
            s = s.to(device)
            s = s.unsqueeze(0)
            embeddings.append(model(s))
            
    print(f"get_all_embeddings: songs len: {len(embeddings)}")
    return torch.cat(embeddings), torch.tensor(labels)
    
### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, valid_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(valid_set, model)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print(accuracies)
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))