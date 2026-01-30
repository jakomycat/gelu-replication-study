import torch

# Train model by one epochs
def train_step(dataloader, device, model, loss_function, optimizer):
    model.train()

    train_loss, train_acc = 0, 0

    # Training loop
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward
        y_pred = model(X)
        loss = loss_function(y_pred, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate batch loss
        train_loss += loss.item()

        # Accumulate batch accuracy
        train_acc += (y_pred.argmax(1) == y).type(torch.float).sum().item()

    # Obtain mean of loss and accuracy
    final_loss = train_loss / len(dataloader)
    final_acc = train_acc / len(dataloader)

    return final_loss, final_acc

# Validate test data
def test_step(dataloader, device, model, loss_function):
    model.eval() # Evaluation mode

    test_loss, test_acc = 0, 0

    # Validation loop
    with torch.no_grad(): # Not calculate gradients
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Prediction
            y_pred = model(X)

            # Calculate batch loss and accuracy for data test 
            test_loss += loss_function(y_pred, y).item()
            test_acc += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Obtain mean of loss and accuracy
    final_loss = test_loss / len(dataloader)
    final_acc = test_acc / len(dataloader)

    return final_loss, final_acc

# Function main to train model
def train_model(train_loader, test_loader, device, model, loss_function, optimizer, epochs=15):
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Train loop for epochs
    for epoch in range(epochs):
        # Train step
        train_loss, train_acc = train_step(train_loader, device, model, loss_function, optimizer)
        
        # Validation
        test_loss, test_acc = test_step(test_loader, device, model, loss_function)

        # Save results
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # Show progress
        print(
            f"Epoca: {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    return history
    