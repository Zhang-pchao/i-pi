import numpy as np
import gc
import weakref

from ipi.engine.forcefields import FFPlumed, ForceRequest


class _Plumed:
    def cmd(self, _command, *_args):
        pass


def test_mtd_update_fallback_uses_force_request():
    ff = FFPlumed.__new__(FFPlumed)
    ff.lastq = np.zeros(3)
    ff.plumed_step = 0
    ff.compute_work = False
    ff.plumed = _Plumed()
    captured = []

    def evaluate(request):
        captured.append(request)
        ff.lastq[:] = request["pos"]

    ff.evaluate = evaluate

    assert ff.mtd_update(np.ones(3), np.eye(3)) == 0.0
    assert len(captured) == 1
    assert isinstance(captured[0], ForceRequest)


def test_evaluate_retains_plumed_force_buffer_until_update():
    class PointerProbe(_Plumed):
        def cmd(self, command, *args):
            if command == "setForces":
                self.buffer = weakref.ref(args[0])

    ff = FFPlumed.__new__(FFPlumed)
    ff.natoms = 1
    ff.lastq = np.zeros(3)
    ff.plumed_step = 0
    ff.charges = np.zeros(1)
    ff.masses = np.ones(1)
    ff.system_force = None
    ff.plumed_data = {}
    ff.plumed = PointerProbe()
    request = ForceRequest(
        {"pos": np.zeros(3), "cell": (np.eye(3), None), "result": None}
    )
    ff.evaluate(request)
    gc.collect()
    retained = ff.plumed.buffer()
    assert retained is not None
    retained[:] = 1.0
    # Subsequent PLUMED work must not mutate the already returned MD force.
    assert np.array_equal(request["result"][1], np.zeros(3))
