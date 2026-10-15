"""Sparse group routing agrees with an independent dense incidence reference."""
import pytest
import torch
from torch.nn import functional as F

from lgmrec.models.sparse_hypergraph import SparseGroupHypergraph


def graph():
    # Unequal edge sizes, directed support, and item 4 absent from every other edge.
    return torch.tensor([[1., 2., 1., 0., 0.], [0., 1., 3., 0., 0.],
                         [0., 0., 1., 1., 0.], [0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 1.]], dtype=torch.float64).to_sparse()


def dense_reference(adjacency, embeddings, features, weighting, layers):
    support = adjacency.to_dense() > 0
    normalized = F.normalize(features, dim=1)
    collaborative = F.normalize(embeddings, dim=1)
    rows = []
    for anchor in range(embeddings.shape[0]):
        members = torch.where(support[anchor])[0]
        if weighting == 'uniform':
            probabilities = embeddings.new_ones(members.numel()) / members.numel()
        else:
            query = F.normalize(normalized[members].mean(0), dim=0)
            logits = (normalized[members] * query).sum(1)
            logits = logits + (collaborative[members] * collaborative[anchor]).sum(1)
            probabilities = logits.softmax(0)
        rows.append(embeddings.new_zeros(embeddings.shape[0]).scatter(0, members, probabilities))
    incidence = torch.stack(rows)  # edge x item
    current = embeddings
    for _ in range(layers):
        current = (incidence.T @ (incidence @ current)) / incidence.sum(0)[:, None]
    return current, incidence


@pytest.mark.parametrize('weighting', ['attention', 'uniform'])
@pytest.mark.parametrize('layers', [1, 2])
def test_dense_forward_and_gradients(weighting, layers):
    torch.manual_seed(23)
    embeddings = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    features = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    model = SparseGroupHypergraph(graph(), weighting=weighting, layers=layers)
    actual = model(embeddings, features)
    expected, incidence = dense_reference(graph(), embeddings, features, weighting, layers)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    padded = model.weights(embeddings, features)
    torch.testing.assert_close(padded[model.mask], incidence[graph().to_dense() > 0])
    target = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 10
    actual_grads = torch.autograd.grad((actual * target).sum(), (embeddings, features),
                                      retain_graph=True, allow_unused=True)
    expected_grads = torch.autograd.grad((expected * target).sum(), (embeddings, features),
                                        allow_unused=True)
    for left, right in zip(actual_grads, expected_grads):
        if left is None or right is None:
            assert left is None and right is None
        else:
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)
    if weighting == 'attention':
        assert actual_grads[1].abs().sum() > 0


def test_padding_isolation_mask_and_constant_signal():
    model = SparseGroupHypergraph(graph(), layers=2)
    for anchor in range(5):
        assert torch.all(model.members[anchor][~model.mask[anchor]] == anchor)
    embeddings = torch.arange(20, dtype=torch.float64).reshape(5, 4).requires_grad_()
    features = torch.eye(5, dtype=torch.float64, requires_grad=True)
    weights = model.weights(embeddings, features)
    assert torch.count_nonzero(weights[~model.mask]) == 0
    torch.testing.assert_close(weights.sum(1), torch.ones(5, dtype=torch.float64))
    output = model(embeddings, features)
    torch.testing.assert_close(output[4], embeddings[4], rtol=0, atol=0)
    gradient, = torch.autograd.grad(output[4].sum(), embeddings)
    assert torch.count_nonzero(gradient[:4]) == 0
    changed = embeddings.detach().clone()
    changed[0] = -10000
    torch.testing.assert_close(model(changed, features)[4], output[4], rtol=0, atol=0)
    constant = torch.full_like(embeddings, 3.)
    torch.testing.assert_close(model(constant, features), constant)


def test_support_canonicalization_metadata_and_nonpersistent_buffers():
    source = graph().coalesce()
    duplicate = torch.sparse_coo_tensor(torch.cat((source.indices(), source.indices()), 1),
                                       torch.cat((source.values(), source.values())), source.shape)
    original = SparseGroupHypergraph(source)
    same = SparseGroupHypergraph(duplicate, weighting='uniform')
    assert original.metadata['fingerprint'] == same.metadata['fingerprint']
    assert original.metadata['nnz'] == 9
    assert original.metadata['min_size'] == 1 and original.metadata['max_size'] == 3
    assert original.metadata['singleton_count'] == 2
    assert original.metadata['n_hyperedges'] == 5
    assert torch.equal(original.members, same.members)
    assert not list(original.parameters()) and not original.state_dict()
    modified = source.to_dense()
    modified[3, 0] = 1
    assert SparseGroupHypergraph(modified.to_sparse()).metadata['fingerprint'] != original.metadata['fingerprint']


def test_training_and_eval_match_and_do_not_consume_rng():
    embeddings = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    features = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    before = torch.get_rng_state().clone()
    model = SparseGroupHypergraph(graph())
    model.train()
    first = model(embeddings, features)
    model.eval()
    second = model(embeddings, features)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('overrides', [{'weighting': 'invalid'}, {'layers': 0},
                                      {'layers': True}, {'layers': 1.5}])
def test_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        SparseGroupHypergraph(graph(), **overrides)


def test_invalid_graph_and_input():
    for adjacency in (torch.eye(3), torch.zeros(2, 3).to_sparse(), torch.zeros(0, 0).to_sparse(),
                      torch.zeros(3, 3).to_sparse(), (-torch.eye(3)).to_sparse()):
        with pytest.raises(ValueError):
            SparseGroupHypergraph(adjacency)
    model = SparseGroupHypergraph(graph())
    with pytest.raises(ValueError, match='matrix'):
        model(torch.zeros(4, 2), torch.zeros(5, 2))
    with pytest.raises(ValueError, match='dtype'):
        model(torch.zeros(5, 2), torch.zeros(5, 2, dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_device_move_and_cuda_gradient():
    cpu = SparseGroupHypergraph(graph(), layers=2)
    model = SparseGroupHypergraph(graph(), layers=2).cuda()
    embeddings = torch.arange(20, dtype=torch.float32).reshape(5, 4).requires_grad_()
    features = torch.eye(5, dtype=torch.float32, requires_grad=True)
    before = torch.cuda.get_rng_state().clone()
    actual = model(embeddings.cuda(), features.cuda())
    expected = cpu(embeddings, features)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
    actual.square().sum().backward()
    assert torch.isfinite(embeddings.grad).all() and torch.isfinite(features.grad).all()
    assert torch.equal(before, torch.cuda.get_rng_state())
