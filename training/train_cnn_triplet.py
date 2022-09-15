import json
import torch

from utils.generic import get_device 

def train_triplet_loss(model: torch.nn.Module, train_set, valid_set, n_epochs, patience, batch_size, lr, save_path):

    device = get_device()
    collate_fn_train = getattr(train_set, "collate_fn", None)
    collate_fn_test = getattr(valid_set, "collate_fn", None)

    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.TripletMarginLoss()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_test)


    train_batches = int(len(train_set)/batch_size)
    valid_batches = int(len(valid_set)/batch_size)
    
    best_loss = 1000000
    current_patience = 0
    for epoch in range(n_epochs):
        print(32*"=")
        print(f"Epoch {epoch}")
        epoch_loss = 0
        model.train()

        for batch, (x, metadata) in enumerate(train_dataloader):

            optimizer.zero_grad()
            (anchor, pos, neg) = x 

            anchor.to(device)
            pos.to(device)
            neg.to(device)

            anchor_out = model(anchor)
            pos_out = model(pos)
            neg_out = model(neg)

            loss = criterion(anchor_out, pos_out, neg_out)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch%16==0:
                print(f'batch {batch}/{train_batches}, loss: {loss.item()}')

        if epoch%1==0:
            print('Evaluating model')
            model.eval()
            valid_loss=0
            with torch.no_grad():
                for batch, (x, metadata) in enumerate(valid_dataloader):     
                
                    (anchor, pos, neg) = x 

                    anchor.to(device)
                    pos.to(device)
                    neg.to(device)

                    anchor_out = model(anchor)
                    pos_out = model(pos)
                    neg_out = model(neg)

                    loss = criterion(anchor_out, pos_out, neg_out)
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
                        }, save_path + "/checkpoint.tar")
                    
                    # Save results
                    with open(save_path + '/train_results.json') as f:
                        json.dump({'n_epochs': epoch, 'valid_loss': valid_loss/valid_batches, 'train_loss': epoch_loss/train_batches}, f)
                else:
                    current_patience +=1
                
                
        if current_patience == patience:
            print(f"No further improvement after {patience} epochs, breaking.")
            break
                    
        print(f"Epoch {epoch} train loss: {epoch_loss/train_batches}")