# Exercise 3 — Harden the safetensors loader (medium)

`examples/safetensors_view.hpp` trusts the file. A truncated download, or a malicious file, could make it read past the mapping. Implement `st::validate(const File&)` in `validate.hpp`. It must throw `std::runtime_error` (any message) when:

1. the header length is larger than the file minus 8 bytes
2. a dtype is not one of `F64 F32 F16 BF16 I64 I32 I16 I8 U8 BOOL`
3. `end < begin`, or `end` is past the end of the data section
4. `end - begin != numel(shape) * dtype_size` (the shape product must use overflow-checked multiplication)
5. two tensors' byte ranges overlap
6. a tensor's absolute file offset is not a multiple of its dtype size (we require natural alignment so that typed pointers are legal)

It must also return the total number of parameters (the sum of numel) for a valid file.

**Test files:** `make_test_files.py` writes one valid file (using the official `safetensors` package) and six broken ones (by editing bytes). Run it once, then ctest:

```bash
uv run python exercises/03-safetensors-loader/make_test_files.py build/st-test
cmake -S exercises -B build/ex && cmake --build build/ex -j && ctest --test-dir build/ex -R 03 --output-on-failure
```

(The CMake test passes `build/st-test` to the binary. Run the generator first.)
