"""Exercise 1 (part b): DCP round trip of #9's model + optimizer, bitwise equal (T0, CPU, single process)."""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch


def test_round_trip_bitwise(tmp_path):
    import torch.distributed.checkpoint as dcp
    from training.train import TinyLM

    def make(seed):
        torch.manual_seed(seed)
        m = TinyLM(64, 32, 2, 4, 16)
        o = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x = torch.randint(0, 64, (2, 16))
        torch.nn.functional.cross_entropy(m(x).flatten(0, 1), x.flatten()).backward()
        o.step()
        return m, o

    m1, o1 = make(0)
    dcp.save({"model": m1.state_dict(), "optim": o1.state_dict()}, checkpoint_id=str(tmp_path / "c"))
    m2, o2 = make(1)
    st = {"model": m2.state_dict(), "optim": o2.state_dict()}
    dcp.load(st, checkpoint_id=str(tmp_path / "c"))
    m2.load_state_dict(st["model"])
    o2.load_state_dict(st["optim"])
    for (k, a), b in zip(m1.state_dict().items(), m2.state_dict().values()):
        assert torch.equal(a, b), k
    s1, s2 = o1.state_dict()["state"], o2.state_dict()["state"]
    for k in s1:
        for name in s1[k]:
            assert torch.equal(torch.as_tensor(s1[k][name]), torch.as_tensor(s2[k][name])), (k, name)
