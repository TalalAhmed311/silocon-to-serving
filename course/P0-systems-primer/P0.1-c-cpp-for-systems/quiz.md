# P0.1 quiz

<details><summary><b>1.</b> What is <code>sizeof(struct { char a; int b; char c; })</code> on x86-64 Linux, and why?</summary>

**12.** `a` is at offset 0. Then come 3 bytes of padding, so `b` (4-aligned) starts at offset 4. `c` is at offset 8, and 3 bytes of tail padding round the total up to a multiple of 4. Reordering to `int b; char a; char c;` gives 8.
</details>

<details><summary><b>2.</b> Why does a struct get <i>tail</i> padding when there is no field after the last one?</summary>

Because of arrays. In `T arr[2]`, `arr[1]` starts at `sizeof(T)`. If `sizeof` weren't a multiple of `alignof(T)`, the second element's fields would be misaligned.
</details>

<details><summary><b>3.</b> A row-major (R, C) float matrix has strides (C, 1). What are the shape and strides of its transpose, and how many bytes does the transpose copy?</summary>

The shape is (C, R) and the strides are (1, C). **Zero bytes are copied**: a transpose is a new view of the same buffer. Bytes only move when something makes it contiguous.
</details>

<details><summary><b>4.</b> Why is the naive transpose slow even though it reads and writes the same number of bytes as a copy?</summary>

The writes have a stride of R floats, so each write touches a different 64-byte cache line and uses only 4 of its bytes. Lines are evicted before their other 15 floats are written, so the hardware moves far more bytes than the useful payload. Blocking keeps a tile's lines in L1 until they are fully used.
</details>

<details><summary><b>5.</b> Your class has a destructor that frees a pointer, and nothing else. What goes wrong with <code>T b = a;</code>?</summary>

The implicit copy constructor copies the pointer, so both objects free it: a **double free**, which is undefined behavior. Fix it by deleting copy (and adding move) or by writing a deep copy (the rule of three/five).
</details>

<details><summary><b>6.</b> Why must a move constructor be <code>noexcept</code> for <code>std::vector</code> to use it during reallocation?</summary>

`vector` gives the strong exception guarantee. If a move could throw halfway through relocating elements, the original buffer would already be gutted. So unless the move is `noexcept`, `vector` copies instead, and for a move-only type it won't compile in some paths.
</details>

<details><summary><b>7.</b> What does <code>std::span&lt;float&gt;</code> own?</summary>

Nothing. It is a (pointer, length) view. The owner, such as a `vector`, `unique_ptr` or `aligned_buffer`, must outlive it.
</details>

<details><summary><b>8.</b> Why put a tile size in a <code>constexpr</code> function or template parameter instead of a runtime variable?</summary>

The compiler can then unroll and vectorize the inner loops for that exact size, and a `static_assert` can reject a tile that doesn't fit in L1 at compile time.
</details>

<details><summary><b>9.</b> AddressSanitizer reports <code>heap-buffer-overflow ... 0 bytes after 16-byte region</code>. What is the most likely bug?</summary>

An off-by-one: the code wrote element `n` of an `n`-element buffer, usually a `<=` loop bound that should be `<`.
</details>
