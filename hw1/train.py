from concurrent.futures import process
import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)

import json  # to parse the file
import numpy as np  # for numpy
from torch.utils.data import TensorDataset, DataLoader  # pytorch
import model
import matplotlib.pyplot as plt  # plotting


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #

    # https://www.geeksforgeeks.org/read-json-file-using-python/
    # Open file and save as a json object
    f = open('lang_to_sem_data.json')
    train_data = json.load(f)
    # print(train_data['train'])

    # Process the input lines first (create a list of objects that are:  ["input text"],[ ["action classifier"], ["object classifer"]])
    processed_train_data = []
    all_episodes = []
    count = 0
    for episode in train_data['train']:
        count += 1
        if count == 50:
            break
        for item in train_data['train'][0]:
            # print(item[0]) # Instruction
            instruction = preprocess_string(item[0])
            # print(item[1][0]) # Action classifier
            action_classifier = preprocess_string(item[1][0])
            # print(item[1][0]) # Object classifier
            target_classifier = preprocess_string(item[1][1])

            all_episodes.append(
                (instruction, (action_classifier, target_classifier)))
    # Compile all episodes into one list
    processed_train_data.append(all_episodes)

    # print(processed_train_data[0][0]) --> succesful import
    # print("Succesfully imported : " +
    #      str(len(processed_train_data[0])) + " lines of data")

    # Create Train/Val Split
    train_split, val_split = create_train_val_splits(processed_train_data)

    # import ipdb
    # ipdb.set_trace()

    # After data has been preprocessed, we need to tokenize the training set
    # Using default vocab size of 1000
    # code snippet below modified form book_identification lecture code ---------/
    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(
        train_split)

    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(
        train_split)
    # end of modificaion from book_identification lecture code --------------/

    # After creating the tokenizer table, its time to encode the training and validation sets
    # Encoding the training set and validation set
    train_np_x, train_np_y, train_np_z = encode_data(
        train_split, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)

    val_np_x, val_np_y, val_np_z = encode_data(
        val_split, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)

    # Tensoring the training set and validation set
    train_dataset = TensorDataset(torch.from_numpy(
        train_np_x), torch.from_numpy(train_np_y), torch.from_numpy(train_np_z))

    val_dataset = TensorDataset(torch.from_numpy(
        val_np_x), torch.from_numpy(val_np_y), torch.from_numpy(val_np_z))

    # Creating data loaders
    minibatch_size = args.batch_size
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=minibatch_size)

    val_loader = DataLoader(
        val_dataset, shuffle=True, batch_size=minibatch_size)

    # Third argument is because of slack message: "and yes maps is a mapping between action label and integer index"
    return train_loader, val_loader, vocab_to_index, actions_to_index, targets_to_index, len_cutoff


def create_train_val_splits(all_data, prop_train=0.8):
    # Function is modified from lecture code
    train_split = []
    val_split = []
    train_episode = []
    val_episode = []
    for episode in all_data:
        val_idxs = np.random.choice(list(range(len(episode))), size=int(
            len(episode)*prop_train + 0.5), replace=False)
        train_episode.extend([episode[idx]
                              for idx in range(len(episode)) if idx not in val_idxs])
        val_episode.extend([episode[idx]
                            for idx in range(len(episode)) if idx in val_idxs])
    train_split.append(train_episode)
    val_split.append(val_episode)

    # print("Train and Val Splits completed, listed below (train then val)")
    # print(str(len(train_episode)))
    # print(str(len(val_episode)))
    return train_split, val_split


def encode_data(data, v2i, seq_len, a2i, t2i):
    # Modified form lecture code's encode data, mine is different because our data is 3 dimensional (2 classses for 1 input)

    n_instructions = len(data[0])
    n_actions = len(a2i)
    n_targets = len(t2i)

    x = np.zeros((n_instructions, seq_len), dtype=np.int32)
    y = np.zeros((n_instructions))
    z = np.zeros((n_instructions), dtype=np.int32)

    # Don't know if i'll need these variables but might be helpful to track this data
    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0

    # WILL DEFINITELY NEED TO REVIEW THIS ITERATION
    for instruction, classes in data[0]:
        # they did this in lecture code but is it really necessary, think we did already earlier
        instruction = preprocess_string(instruction)
        x[idx][0] = v2i["<start>"]  # add start token
        jdx = 1
        for word in instruction.split():
            if len(word) > 0:
                x[idx][jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                # can double check if we want to track unks
                n_unks += 1 if x[idx][jdx] == v2i["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == seq_len - 1:
                    n_early_cutoff += 1
                    break
        x[idx][jdx] = v2i["<end>"]
        # DOUBLE CHECK BELOW CODE BASED ON FINAL ITERATION IMPLEMENTAION : Saving the action and target class for the instruction
        y[idx] = a2i[classes[0]]
        z[idx] = t2i[classes[1]]
        idx += 1

    # COPIED print statements from lecture code to help with debugging
    """ print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(v2i))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d instances without regard to order" % idx) """
    return x, y, z


def setup_model(args, v2i, a2i, t2i, len_cutoff, device, embedding_dimension=20):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #

    # Assuming this is going to be like the BookIdet class setup in lecture code, how do i incorporate the fact that is is 2D

    # how do we go about making this LSTM
    # instantiate model.py model here
    # Arbitrarily set embedding dimension to 3
    model1 = model.InstructionClassifier(
        device, len(v2i), len_cutoff, len(a2i), len(t2i), embedding_dimension)
    return model1


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, action_class, target_class) in loader:
        # put model inputs to device
        inputs, action_class, target_class = inputs.to(
            device), action_class.to(device), target_class.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs)

        # print("Actions_out: shape", actions_out.shape)
        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(
            actions_out.squeeze(), action_class[:].long())
        target_loss = target_criterion(
            targets_out.squeeze(), target_class[:].long())

        loss = action_loss + target_loss
        # print("loss calculated as:", loss)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(action_class[:].cpu().numpy())
        target_labels.extend(target_class[:].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()
    # each element will be (train, val)
    train_loss_summary_action = []
    val_loss_summary_action = []
    train_acc_summary_action = []
    val_acc_summary_action = []
    train_loss_summary_target = []
    val_loss_summary_target = []
    train_acc_summary_target = []
    val_acc_summary_target = []

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target losaccs: {val_target_acc}"
            )

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #
        # Within for loop, add the tracked data to a list thats keeping track
        train_loss_summary_action.extend([train_action_loss])
        train_loss_summary_target.extend([train_target_loss])
        val_loss_summary_action.extend([val_action_loss])
        val_loss_summary_target.extend([val_target_loss])
        train_acc_summary_action.extend([train_action_acc])
        train_acc_summary_target.extend([train_target_acc])
        val_acc_summary_action.extend([val_action_acc])
        val_acc_summary_target.extend([val_target_acc])

    # outside the for loop, print out the summary tables
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    #
    ax1.set_title("train loss")
    ax1.plot(train_loss_summary_action, label="action")
    ax1.plot(train_loss_summary_target, label="target")
    #
    ax2.set_title("val loss")
    ax2.plot(val_loss_summary_action, label="action")
    ax2.plot(val_loss_summary_target, label="target")
    #
    ax3.set_title("train accuracy")
    ax3.plot(train_acc_summary_action, label="action")
    ax3.plot(train_acc_summary_target, label="target")
    #
    ax4.set_title("val accuracy")
    ax4.plot(val_acc_summary_action, label="action")
    ax4.plot(val_acc_summary_target, label="target")
    #
    fig.legend(["action", "target"])
    plt.show()


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, v2i, a2i, t2i, len_cutoff = setup_dataloader(
        args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, v2i, a2i, t2i, len_cutoff, device)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(
        args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
