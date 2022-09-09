# IMPLEMENT YOUR MODEL CLASS HERE
# Modified based on class coding activity
import torch


class InstructionClassifier(torch.nn.Module):
    def __init__(
        self,
        device,
        vocab_size,
        input_len,
        n_actions,
        n_targets,
        embedding_dim
    ):
        super(InstructionClassifier, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.n_actions = n_actions
        self.n_targets = n_targets

        # embedding layer
        self.embedding = torch.nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)

        # maxpool layer
        # self.maxpool = torch.nn.MaxPool2d((input_len, 1), ceil_mode=True)

        # lstm layer
        # keeping one hidden dimension
        hidden_dim = embedding_dim
        # arbitrarily set hidden dim to embedding dim
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # linear layer for action
        self.fc_action = torch.nn.Linear(hidden_dim, n_actions)

        # linear layer for target
        self.fc_target = torch.nn.Linear(hidden_dim, n_targets)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # Modified code from : https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # Generate embeds
        embeds = self.embedding(x)
        # print("Embeds shape is:", embeds.shape)
        # Run LSTM on embeds
        # not sure the 1, -1 parameters here
        lstm_out = self.lstm(embeds)
        # print("lstm out has been generated, the important tensor shape is:",
        #      lstm_out[1][0].shape)

        # need to extract the correct output tensor from lstm model
        lstm_final = lstm_out[1][0]
        lstm_final = lstm_final.squeeze()  # lets get rid of the extra dimension
        # print("lstm final shape: ", lstm_final.shape)

        # example in website had one space, but we're mapping to 2 different class heads so need two different FC layers (as per lecture discussion)
        action_space = self.fc_action(lstm_final)
        target_space = self.fc_target(lstm_final)
        # print("Action and Target Space have been completed, their output:\n",
        #      action_space.shape, "\n", target_space.shape)
        # action and target space [0] are the tensors with the classification data

        return action_space, target_space
