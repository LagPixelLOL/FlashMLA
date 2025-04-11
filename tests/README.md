# FlashMLA Tests

This directory contains tests for the FlashMLA CUDA kernel implementation across different CUDA architectures.

## Available Tests

- `test_flash_mla_sm80.py`: Tests for sm80 architecture (Ampere)
- `test_flash_mla_sm86.py`: Tests for sm86 architecture (Ampere)
- `test_flash_mla_sm89.py`: Tests for sm89 architecture (Ada Lovelace)
- `test_flash_mla_sm90.py`: Tests for sm90 architecture (Hopper)

## Running Tests

To run a test for a specific architecture, make sure you have a GPU with the corresponding compute capability and run:

```bash
# For sm80 (A100)
python tests/test_flash_mla_sm80.py

# For sm86 (A10, RTX 3090, etc.)
python tests/test_flash_mla_sm86.py

# For sm89 (RTX 4090, etc.)
python tests/test_flash_mla_sm89.py

# For sm90 (H100)
python tests/test_flash_mla_sm90.py --dtype bf16
# or
python tests/test_flash_mla_sm90.py --dtype fp16
```

Note that sm80, sm86, and sm89 only support bfloat16 data type, while sm90 supports both bfloat16 and float16.

### Memory and Cache Considerations

- sm80 (A100) implementation uses larger block sizes (64x32), more warps (4), and full dimensions (576/512), suitable for datacenter GPUs with large VRAM and cache.
- sm86 (RTX 30xx) implementation uses extremely small block sizes (16x16), minimal warps (1), and reduced dimensions (288/256) to accommodate consumer GPUs with very limited cache.
- sm89 (RTX 40xx) implementation also uses extremely small block sizes (16x16), minimal warps (1), and reduced dimensions (288/256), as even these newer consumer GPUs have very limited cache compared to datacenter GPUs.

The test parameters for sm86 and sm89 have also been drastically reduced (smaller batch sizes, shorter sequences, fewer heads, reduced dimensions) to ensure the tests can run on consumer hardware.

### Dimension Differences

- sm80 and sm90: Head dimension = 576, Output dimension = 512
- sm86 and sm89: Head dimension = 288, Output dimension = 256 (reduced by half)

## Test Parameters

The tests run with various combinations of parameters:
- Batch size
- Sequence length
- Number of heads
- Variable sequence length

Each test compares the output of the FlashMLA implementation with a reference implementation using PyTorch's native operations and reports performance metrics.
