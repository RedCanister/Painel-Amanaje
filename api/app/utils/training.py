
def train_pytorch(model, x, y, criterion, optimizer, epochs, accum_steps, kwargs**):

    model.train()
    
    for epoch in range(epochs):
        
        # training metric
        total_loss = 0

        outputs, _ = model(x, ) #kwargs

        loss = criterion(outputs.squueze(), y)

        # retropropagation
        loss.backward()

        # weight update

        if (epoch + 1) % accum_steps == 0 and accum_steps:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

        avg_loss = total_loss / epochs

    
    return model, avg_loss


def train_sklearn(model_, X, y, params, random_state, kwargs**):

    model = model_(params, random_state)

    model.fit(X, y)

    return model