import json
import torch

from utils.generic import get_device 

from pytorch_metric_learning import miners, losses

def train_hard_triplet_loss(model: torch.nn.Module, train_set, valid_set, n_epochs, patience, lr, checkpoints_path, results_path):

    device = get_device()
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    miner = miners.BatchHardMiner()
    loss_func = losses.TripletMarginLoss()

    train_batches = len(train_set)
    valid_batches = len(valid_set)
    
    best_loss = 1000000
    current_patience = 0
    for epoch in range(n_epochs):
        print(32*"=")
        print(f"Epoch {epoch}")
        epoch_loss = 0
        total_frames = 0
        model.train()

        for i in range(len(train_set)):
            # N X 1 X num_feats X num_samples, N
            (data, labels) = train_set[i]
            
            optimizer.zero_grad()
            embeddings = model(data)
            hard_pairs = miner(embeddings, labels)
         
            loss = loss_func(embeddings, labels, hard_pairs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i%16==0:
                print(f'batch {i}/{train_batches}, loss: {loss.item()}')

        if epoch%1==0:
            print('Evaluating model')
            model.eval()
            valid_loss=0
            with torch.no_grad():
                for i in range(len(valid_set)):
                    # N X 1 X num_feats X num_samples, N
                    (data, labels) = train_set[i]     
                

                    embeddings = model(data)
                    hard_pairs = miner(embeddings, labels)
                
                    loss = loss_func(embeddings, labels, hard_pairs)
                    valid_loss += loss.item()
            
                print(f"Epoch {epoch} valid loss: {valid_loss/valid_batches}")
                if valid_loss < best_loss:
                    print("New best loss, saving model")
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
                
                
        if current_patience == patience:
            print(f"No further improvement after {patience} epochs, breaking.")
            break
                    
        print(f"Epoch {epoch} train loss: {epoch_loss/train_batches}")
        
    # Load best model
    checkpoint = torch.load(checkpoints_path + "/checkpoint.tar")
    model.load_state_dict(checkpoint['model_state_dict'])