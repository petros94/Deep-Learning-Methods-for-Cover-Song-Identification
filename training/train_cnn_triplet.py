import torch 

def train_triplet_loss(model: torch.nn.Module, train_set, valid_set, n_epochs, batch_size, lr):

    collate_fn = getattr(train_set, "collate_fn", None)

    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.TripletMarginLoss()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


    train_batches = int(len(train_set)/batch_size)
    valid_batches = int(len(valid_set)/batch_size)
    for epoch in range(n_epochs):
        print(32*"=")
        print(f"Epoch {epoch}")
        epoch_loss = 0
        model.train()

        for batch, x in enumerate(train_dataloader):

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
                for batch, x in enumerate(valid_dataloader):     
                
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

        print(f"Epoch {epoch} train loss: {epoch_loss/train_batches}")