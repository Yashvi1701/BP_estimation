import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=50,
    scheduler=None
):

    train_losses = []
    val_losses = []

    train_sbp_losses = []
    train_dbp_losses = []

    val_sbp_losses = []
    val_dbp_losses = []


    for epoch in range(epochs):

        # =====================
        # Training
        # =====================

        model.train()

        running_loss = 0.0
        running_sbp_loss = 0.0
        running_dbp_loss = 0.0


        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)


            optimizer.zero_grad()


            predictions = model(X_batch)


            sbp_loss = criterion(
                predictions[:,0],
                y_batch[:,0]
            )

            dbp_loss = criterion(
                predictions[:,1],
                y_batch[:,1]
            )


            loss = sbp_loss + dbp_loss


            loss.backward()

            optimizer.step()


            running_loss += loss.item()
            running_sbp_loss += sbp_loss.item()
            running_dbp_loss += dbp_loss.item()



        train_loss = running_loss / len(train_loader)

        train_sbp_loss = running_sbp_loss / len(train_loader)

        train_dbp_loss = running_dbp_loss / len(train_loader)



        # =====================
        # Validation
        # =====================

        model.eval()

        val_loss = 0.0
        val_sbp_loss = 0.0
        val_dbp_loss = 0.0


        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)


                predictions = model(X_batch)


                sbp_loss = criterion(
                    predictions[:,0],
                    y_batch[:,0]
                )

                dbp_loss = criterion(
                    predictions[:,1],
                    y_batch[:,1]
                )


                loss = sbp_loss + dbp_loss


                val_loss += loss.item()
                val_sbp_loss += sbp_loss.item()
                val_dbp_loss += dbp_loss.item()



        val_loss /= len(val_loader)

        val_sbp_loss /= len(val_loader)

        val_dbp_loss /= len(val_loader)



        # Scheduler update

        if scheduler is not None:
            scheduler.step(val_loss)



        # Store losses

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_sbp_losses.append(train_sbp_loss)
        train_dbp_losses.append(train_dbp_loss)

        val_sbp_losses.append(val_sbp_loss)
        val_dbp_losses.append(val_dbp_loss)



        # RMSE

        train_sbp_rmse = np.sqrt(train_sbp_loss)
        train_dbp_rmse = np.sqrt(train_dbp_loss)

        val_sbp_rmse = np.sqrt(val_sbp_loss)
        val_dbp_rmse = np.sqrt(val_dbp_loss)



        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train MSE: {train_loss:.2f} "
            f"(SBP RMSE: {train_sbp_rmse:.2f}, DBP RMSE: {train_dbp_rmse:.2f}) | "
            f"Val MSE: {val_loss:.2f} "
            f"(SBP RMSE: {val_sbp_rmse:.2f}, DBP RMSE: {val_dbp_rmse:.2f})"
        )


    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_sbp_loss": train_sbp_losses,
        "train_dbp_loss": train_dbp_losses,
        "val_sbp_loss": val_sbp_losses,
        "val_dbp_loss": val_dbp_losses
    }


    return model, history, train_losses, val_losses



def evaluate_model(model, test_loader, device):

    model.eval()

    predictions = []
    targets = []


    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)

            outputs = model(X_batch)


            predictions.append(outputs.cpu().numpy())
            targets.append(y_batch.numpy())



    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)



    # =====================
    # Metrics
    # =====================

    sbp_mae = mean_absolute_error(
        targets[:,0],
        predictions[:,0]
    )


    dbp_mae = mean_absolute_error(
        targets[:,1],
        predictions[:,1]
    )


    sbp_rmse = np.sqrt(
        mean_squared_error(
            targets[:,0],
            predictions[:,0]
        )
    )


    dbp_rmse = np.sqrt(
        mean_squared_error(
            targets[:,1],
            predictions[:,1]
        )
    )


    # print("Predictions shape:", predictions.shape)
    # print("Targets shape:", targets.shape)

    print(f"SBP MAE: {sbp_mae:.2f} mmHg")
    print(f"DBP MAE: {dbp_mae:.2f} mmHg")

    print(f"SBP RMSE: {sbp_rmse:.2f} mmHg")
    print(f"DBP RMSE: {dbp_rmse:.2f} mmHg")


    metrics = {
        "SBP_MAE": sbp_mae,
        "DBP_MAE": dbp_mae,
        "SBP_RMSE": sbp_rmse,
        "DBP_RMSE": dbp_rmse
    }


    return predictions, targets, metrics