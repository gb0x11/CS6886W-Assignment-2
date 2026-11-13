"""Activation-memory accounting.

While a model runs one forward pass, every activation tensor that has to be
held in memory is reported to an `ActivationTracer` when it is produced
(`alloc`) and when its last consumer has run (`free`). From this the tracer
derives two numbers, each in the compressed bit-width and in FP32:

    total  sum of the sizes of all activation tensors of one forward pass
    peak   largest amount of activation memory alive at the same time
           (input + output of the running layer, plus a pending residual)

Sizes are per image: the batch dimension is divided out.
"""


class NullTracer:
    """Used during normal training / evaluation: records nothing."""

    def alloc(self, name, tensor, bits, nbytes=None):
        pass

    def free(self, name):
        pass


class ActivationTracer(NullTracer):
    def __init__(self):
        self.records = []   # one dict per activation tensor, in execution order
        self._live = {}     # name -> (bits at compressed precision, bits at FP32)
        self.peak_bits = 0
        self.peak_bits_fp32 = 0
        self.peak_at = None

    def alloc(self, name, tensor, bits, nbytes=None):
        """Register an activation tensor of `bits` bits per element.

        `nbytes` is the measured size of its bit-packed buffer, if the model
        actually packed it (see PackedMobileNetV2.forward(pack_acts=True)).
        """
        elements = tensor[0].numel()
        rec = {"name": name, "shape": tuple(tensor.shape[1:]), "elements": elements,
               "bits": int(bits), "size_bits": elements * int(bits), "fp32_bits": elements * 32,
               "packed_bytes": nbytes}
        self.records.append(rec)
        self._live[name] = (rec["size_bits"], rec["fp32_bits"])
        live = sum(v[0] for v in self._live.values())
        if live > self.peak_bits:
            self.peak_bits, self.peak_at = live, name
        self.peak_bits_fp32 = max(self.peak_bits_fp32, sum(v[1] for v in self._live.values()))

    def free(self, name):
        self._live.pop(name, None)

    def summary(self) -> dict:
        total = sum(r["size_bits"] for r in self.records)
        total_fp32 = sum(r["fp32_bits"] for r in self.records)
        elements = sum(r["elements"] for r in self.records)
        return {
            "act_tensors": len(self.records),
            "act_elements": elements,
            "act_total_bytes": total / 8,
            "act_total_bytes_fp32": total_fp32 / 8,
            "act_ratio": total_fp32 / total,
            "act_avg_bits": total / elements,
            "act_peak_bytes": self.peak_bits / 8,
            "act_peak_bytes_fp32": self.peak_bits_fp32 / 8,
            "act_peak_ratio": self.peak_bits_fp32 / self.peak_bits,
            "act_peak_at": self.peak_at,
        }
