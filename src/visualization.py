from matplotlib import pyplot as plt
import os

def plot_mnist_classification(all_history, save_dir):
    fig, axs = plt.subplots(1, 2, figsize=(20, 7))

    # Loop
    for history in all_history:
        # Extract data
        loss_train = all_history[history]["train_loss"]
        loss_test = all_history[history]["test_loss"]

        epochs = range(1, len(loss_train) + 1)

        #
        if "without" in history:
            if "ReLU" in history:
                axs[0].plot(epochs, loss_train, color="green", label="ReLU")
                axs[0].plot(epochs, loss_test, color="green", alpha=0.5)
            elif "GELU" in history:
                axs[0].plot(epochs, loss_train, color="blue", label="GELU")
                axs[0].plot(epochs, loss_test, color="blue", alpha=0.5)
            elif "ELU" in history:
                axs[0].plot(epochs, loss_train, color="orange", label="ELU")
                axs[0].plot(epochs, loss_test, color="orange", alpha=0.5)
        elif "with" in history:
            if "ReLU" in history:
                axs[1].plot(epochs, loss_train, color="green", label="ReLU")
                axs[1].plot(epochs, loss_test, color="green", alpha=0.5)
            elif "GELU" in history:
                axs[1].plot(epochs, loss_train, color="blue", label="GELU")
                axs[1].plot(epochs, loss_test, color="blue", alpha=0.5)
            elif "ELU" in history:
                axs[1].plot(epochs, loss_train, color="orange", label="ELU")
                axs[1].plot(epochs, loss_test, color="orange", alpha=0.5)

        
    # Configuration plot
    axs[0].set_xlabel("Epoch")
    axs[1].set_xlabel("Epoch")    

    axs[0].set_ylabel("Log Loss")
    axs[1].set_ylabel("Log Loss")

    axs[0].set_ylim(0.0, 0.14)
    axs[1].set_ylim(0.0, 0.5)

    plt.legend()
            
    plt.savefig(os.path.join(save_dir, f"MNIST_Classification.png"), bbox_inches='tight')

    # Close plots
    plt.close()