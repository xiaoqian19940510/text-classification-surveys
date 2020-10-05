class CONFIG(object):
    """docstring for CONFIG"""

    def __init__(self):
        super(CONFIG, self).__init__()

        self.dataset = 'R8'
        self.model = 'gcn'  # 'gcn', 'gcn_cheby', 'dense'
        self.learning_rate = 0.01  # Initial learning rate.
        self.epochs = 200  # Number of epochs to train.
        self.hidden1 = 200  # Number of units in hidden layer 1.
        self.dropout = 0.5  # Dropout rate (1 - keep probability).
        self.weight_decay = 0.  # Weight for L2 loss on embedding matrix.
        self.early_stopping = 10  # Tolerance for early stopping (# of epochs).
        self.max_degree = 3  # Maximum Chebyshev polynomial degree.
