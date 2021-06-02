import torch
import torch.nn as nn
import torchvision.models


class QueryEncoder(nn.Module):
    def __init__(self, c):
        super(QueryEncoder, self).__init__()
        self._encoder = torchvision.models.resnet50(num_classes=c).cuda()

        # Add a hidden layer to form a 2-layer MLP head (as described in MoCo V2)
        orig_fc_in_features = self._encoder.fc.weight.shape[1]
        self._encoder.fc = nn.Sequential(nn.Linear(orig_fc_in_features, orig_fc_in_features), nn.ReLU(), self._encoder.fc)

    def forward(self, queries):
        q = self._encoder(queries)
        return nn.functional.normalize(q, dim=1)

    def get_encoder(self):
        return self._encoder


class KeyEncoder(QueryEncoder):
    def __init__(self, query_encoder, c, m, k):
        super(KeyEncoder, self).__init__(c=c)
        self._m = m
        self._query_encoder = query_encoder
        self._reset_parameters()
        self.register_buffer("_queue", torch.randn(c, k).cuda())
        self._queue = nn.functional.normalize(self._queue, dim=0).cuda()

    def _zip_parameters(self):
        return zip(self._query_encoder.parameters(), self.parameters())

    def _reset_parameters(self):
        for query_param, key_param in self._zip_parameters():
            key_param.data.copy_(query_param.data)
            key_param.requires_grad = False

    def update_parameters(self):
        for query_param, key_param in self._zip_parameters():
            key_param.data = key_param.data * self._m + query_param.data * (1. - self._m)

    def get_query_encoder(self):
        return self._query_encoder

    def get_queue(self):
        return self._queue

    def enqueue(self, keys):
        self._queue = torch.cat([self._queue, keys.T], dim=1)

    def dequeue(self, batch_size):
        self._queue = self._queue[:, batch_size:]


class LinearClassifier(nn.Module):
    def __init__(self, c):
        super(LinearClassifier, self).__init__()
        self._classifier = torchvision.models.resnet50(num_classes=c).cuda()

    def forward(self, x):
        return self._classifier(x)

    def freeze_features(self, query_encoder):
        state_dict = query_encoder.get_encoder().state_dict()
        for name, param in self._classifier.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.data.copy_(state_dict[name])
                param.requires_grad = False

        self._classifier.fc.weight.data.normal_(mean=0.0, std=0.01)
        self._classifier.fc.bias.data.zero_()

    def parameters(self):
        return list(filter(lambda parameter: parameter.requires_grad, self._classifier.parameters()))


class MoCo(nn.Module):
    def __init__(self, c=128, m=0.999, k=65536):
        super(MoCo, self).__init__()
        self._c = c
        self._m = m
        self._query_encoder = QueryEncoder(c=c)
        self._key_encoder = KeyEncoder(query_encoder=self._query_encoder, c=c, m=m, k=k)

    def get_dim(self):
        return self._dim

    def get_m(self):
        return self._m

    def get_query_encoder(self):
        return self._query_encoder;

    def forward(self, queries, keys):
        q = self._query_encoder.forward(queries)

        with torch.no_grad():
            k = self._key_encoder.forward(keys)

        l_pos = torch.matmul(q.reshape([q.shape[0], 1, q.shape[1]]), k.reshape([k.shape[0], k.shape[1], 1])).squeeze(dim=1)
        l_neg = torch.matmul(q, self._key_encoder.get_queue())

        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(q.shape[0], dtype=torch.long).cuda()

        return logits, labels, q, k

    def parameters(self):
        return self._query_encoder.parameters()

    def update_key_encoder(self, keys):
        self._key_encoder.update_parameters()
        self._key_encoder.enqueue(keys=keys)
        self._key_encoder.dequeue(batch_size=keys.shape[0])
