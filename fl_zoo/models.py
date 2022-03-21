import torch.nn as nn


class CNN(nn.Module):
    """CNN described in "Communication-Efficient Learning of Deep Networks
    from Decentralized Data" (https://arxiv.org/pdf/1602.05629.pdf).

    Parameters
    ----------
    in_features : int (default=1)
        Number of channels in the input image.
    num_classes : int (default=10)
        Number of class labels.
    """
    def __init__(self, in_features=1, num_classes=10):
        super(CNN, self).__init__()
        self._in_features = in_features
        self._num_classes = num_classes

        self.conv1 = nn.Conv2d(self._in_features, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, self._num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    """CNN described in "Communication-Efficient Learning of Deep Networks
    from Decentralized Data" (https://arxiv.org/pdf/1602.05629.pdf).

    Parameters
    ----------
    in_features : int (default=784)
        Number input features.
    hidden_dim : int (cdefault=200)
        Number of hidden units.
    num_classes : int (default=10)
        Number of class labels.
    """
    def __init__(self, in_features=784, hidden_dim=200, num_classes=10):
        super(MLP, self).__init__()
        self._in_features = in_features
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes

        self.fc1 = nn.Linear(self._in_features, self._hidden_dim)
        self.fc2 = nn.Linear(self._hidden_dim, self._num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LogisticRegression(nn.Module):
    """Multinomial Logistic Regression"""
    def __init__(self, in_features, num_classes):
        super(LogisticRegression, self).__init__()
        self._in_features = in_features
        self._num_classes = num_classes
        self.fc = nn.Linear(self._in_features, self._num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


MODEL_MAP = {
    'cnn': CNN,
    'mlp': MLP,
    'lr': LogisticRegression
}
