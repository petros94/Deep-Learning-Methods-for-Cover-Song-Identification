import json
import torch

from utils.generic import get_device 

from pytorch_metric_learning import miners, losses

def train_hard_triplet_loss(model: torch.nn.Module, train_set, valid_set, n_epochs, patience, batch_size, lr, checkpoints_path, results_path, second_valid_set):

    device = get_device()
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    miner = miners.BatchEasyHardMiner(
        pos_strategy=miners.BatchEasyHardMiner.SEMIHARD,
        neg_strategy=miners.BatchEasyHardMiner.HARD)
    loss_func = losses.TripletMarginLoss(margin=1.00)
    
    if second_valid_set is not None:
        criterion = torch.nn.TripletMarginLoss()
        collate_fn_test = getattr(second_valid_set, "collate_fn", None)
        random_triplet_valid_dataloader = torch.utils.data.DataLoader(second_valid_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_test)

    train_batches = len(train_set)
    valid_batches = len(valid_set)
    
    best_loss = 1000000
    current_patience = 0
    for epoch in range(n_epochs):
        print(32*"=")
        print(f"Epoch {epoch}")
        epoch_loss = 0
        model.train()

        for i in range(len(train_set)):
            # N X 1 X num_feats X num_samples, N
            (data, labels) = train_set[i]
            data = data.to(device)
            
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
                    data = data.to(device)
                

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
                    
            # Evaluate on random triplets
            if second_valid_set is not None:
                random_valid_batches = int(len(second_valid_set)/batch_size)
                valid_loss=0
                with torch.no_grad():
                    for batch, (x, metadata) in enumerate(random_triplet_valid_dataloader):     
                    
                        (anchor, pos, neg) = x 

                        anchor.to(device)
                        pos.to(device)
                        neg.to(device)

                        anchor_out = model(anchor)
                        pos_out = model(pos)
                        neg_out = model(neg)

                        loss = criterion(anchor_out, pos_out, neg_out)
                        valid_loss += loss.item()
                print(f"Epoch {epoch} random triplet valid loss: {valid_loss/random_valid_batches}")
                
                
        if current_patience == patience:
            print(f"No further improvement after {patience} epochs, breaking.")
            break
                    
        print(f"Epoch {epoch} train loss: {epoch_loss/train_batches}")
        
    # Load best model
    checkpoint = torch.load(checkpoints_path + "/checkpoint.tar")
    model.load_state_dict(checkpoint['model_state_dict'])