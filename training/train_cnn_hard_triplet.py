import json
import torch

from utils.generic import get_device 

from pytorch_metric_learning import miners, losses, reducers, distances

def train_hard_triplet_loss(model: torch.nn.Module, train_set, valid_set, n_epochs, patience, batch_size, lr, checkpoints_path, results_path):

    device = get_device()
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    margin = 1.00
    distance = distances.LpDistance(normalize_embeddings=False)
    batch_semihard_miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard", distance=distance)
    batch_all_miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="all", distance=distance)
    loss_func = losses.TripletMarginLoss(margin=margin, distance=distance)
    miner = batch_all_miner
    
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
                    hard_pairs = miner(embeddings, labels)
                
                    loss = loss_func(embeddings, labels, hard_pairs)
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
                    
        print(f"Epoch {epoch} train loss: {epoch_loss/train_batches}, mean triplets: {int(float(mean_triplets)/train_batches)}, perf score: {epoch_loss*int(float(mean_triplets)/train_batches)}")
        
    # Load best model
    checkpoint = torch.load(checkpoints_path + "/checkpoint.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    
def get_all_embeddings(data_set, model):
    device = get_device()
    embeddings = []
    all_labels = []
    with torch.no_grad():
        for i in range(len(data_set)):
            # N X 1 X num_feats X num_samples, N
            (data, labels) = data_set[i]
            data = data.to(device)
            
            embeddings.append(model(data))
            all_labels.extend(labels)
            
    return torch.cat(embeddings), torch.tensor(all_labels)
    
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