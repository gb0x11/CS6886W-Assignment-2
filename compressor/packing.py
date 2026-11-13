"""Bit-packing of integer codes into byte buffers (the in-memory format).

A tensor of b-bit codes is stored as a contiguous bit stream inside a uint8
tensor, so n codes occupy exactly ceil(n * b / 8) bytes of RAM, for any
b from 1 to 16 (for example two 4-bit weights per byte, or eight 3-bit
weights in three bytes). This is plain fixed-width packing, written with
shifts and masks: there is no entropy coding, so every element can be read
back at a fixed offset and the buffer is the form the model is held in
memory, not only a smaller file.
"""
from dataclasses import dataclass

import torch

# Bookkeeping stored next to every packed tensor: bit-width, signedness and
# up to four dimensions. Counted as 16 bytes in the size estimates.
PACKED_META_BYTES = 16


@torch.no_grad()
def pack_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack non-negative integers < 2^bits into a uint8 bit stream (MSB first)."""
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    v = values.reshape(-1).to(torch.int64).cpu()
    if v.numel() and (int(v.min()) < 0 or int(v.max()) >= (1 << bits)):
        raise ValueError("value out of range for the requested bit-width")
    shifts = torch.arange(bits - 1, -1, -1, dtype=torch.int64)
    stream = ((v[:, None] >> shifts) & 1).reshape(-1)             # n * bits single bits
    pad = (-stream.numel()) % 8
    if pad:
        stream = torch.cat([stream, stream.new_zeros(pad)])
    place = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.int64)
    return (stream.view(-1, 8) * place).sum(dim=1).to(torch.uint8)


@torch.no_grad()
def unpack_bits(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """Inverse of `pack_bits`: the first `count` values as an int64 tensor."""
    shifts8 = torch.arange(7, -1, -1, dtype=torch.int64)
    stream = ((packed.to(torch.int64).cpu()[:, None] >> shifts8) & 1).reshape(-1)
    stream = stream[:count * bits].view(count, bits)
    place = 1 << torch.arange(bits - 1, -1, -1, dtype=torch.int64)
    return (stream * place).sum(dim=1)


@dataclass
class PackedTensor:
    """A tensor of signed or unsigned b-bit codes held as a packed byte buffer."""
    data: torch.Tensor   # uint8, ceil(numel * bits / 8) bytes
    bits: int
    shape: tuple
    qn: int              # smallest code; codes are stored as (code - qn) >= 0

    @classmethod
    def from_codes(cls, codes: torch.Tensor, bits: int, qn: int) -> "PackedTensor":
        return cls(pack_bits(codes.to(torch.int64) - qn, bits), bits, tuple(codes.shape), qn)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        """Bytes of RAM actually held by the packed buffer."""
        return self.data.numel() * self.data.element_size()

    def codes(self) -> torch.Tensor:
        """Unpack to integer codes (int64) with the original shape."""
        return (unpack_bits(self.data, self.bits, self.numel) + self.qn).view(self.shape)
