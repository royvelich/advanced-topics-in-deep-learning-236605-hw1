import torch
import torch.nn as nn
import torchvision.models


class QueryEncoder(nn.Module):
    def __init__(self, c):
        super(QueryEncoder, self).__init__()
        self._encoder = torchvision.models.resnet50(num_classes=c).cuda()

    def forward(self, queries):
        q = self._encoder(queries)
        return nn.functional.normalize(q, dim=1)

    def parameters(self):
        return self._encoder.parameters()


class KeyEncoder(QueryEncoder):
    def __init__(self, query_encoder, c, m, k):
        super(KeyEncoder, self).__init__(c=c)
        self._m = m
        self._query_encoder = query_encoder
        self._reset_parameters()
        # self.register_buffer("queue", torch.randn(c, k))
        self._queue = nn.functional.normalize(torch.randn(c, k).cuda(), dim=0)

    def _zip_parameters(self):
        return zip(self._query_encoder.parameters(), self.parameters())

    def _reset_parameters(self):
        for query_param, key_param in self._zip_parameters():
            key_param.data.copy_(query_param.data)

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


# class Trainer:
#     def __init__(self, encoders_bundle, loss_fn, k=65536):
#         self._encoders_bundle = encoders_bundle
#         self._loss_fn = loss_fn
#         self._epochs = epochs
#
#
#
#     def forward(self, queries, keys):
#         q = self.extract_features(queries=queries)
#
#         with torch.no_grad():
#             k = self._key_encoder(keys)
#             k = nn.functional.normalize(k, dim=1)
#
#         l_pos = torch.matmul(q.reshape([q.shape[0], 1, q.shape[1]]), k.reshape([k.shape[0], k.shape[1], 1])).squeeze(dim=1)
#         l_neg = torch.matmul(q, self._queue)
#
#         logits = torch.cat([l_pos, l_neg], dim=1)
#         labels = torch.zeros(q.shape[0])
#
#         return logits, labels
