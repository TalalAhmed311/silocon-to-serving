# R.2 quiz

<details><summary><b>1.</b> Name the four regions of a process's address space. Which two are freed without any code from you, and which one must you free yourself?</summary>

Code, globals, stack and heap. Code and globals live until the program ends and the OS reclaims them. Stack frames are freed automatically when a function returns. **Heap blocks** (from `new` and `malloc`) are freed by you, or by an owner object.
</details>

<details><summary><b>2.</b> <code>int* f() { int x = 42; return &amp;x; }</code> What is wrong, and how do you fix it?</summary>

`x` lives in `f`'s stack frame, which is popped on return, so the returned pointer dangles. Reading it is undefined behaviour: it may still show 42 until another call reuses the bytes. Return the **value** (`int f()`), or a `std::vector` that owns its own storage.
</details>

<details><summary><b>3.</b> Which <code>delete</code> goes with <code>new float[10]</code>? What about <code>malloc(40)</code> and <code>new int(5)</code>?</summary>

`delete[]` goes with `new[]`, `free` with `malloc`, and plain `delete` with `new int(5)`. Mixing them is undefined behaviour.
</details>

<details><summary><b>4.</b> Give a one-line example of a leak, a double free and a use-after-free.</summary>

Leak: `int* p = new int[4]; p = nullptr;`. Double free: `delete[] p; delete[] p;`. Use-after-free: `delete[] p; p[0] = 1;`.
</details>

<details><summary><b>5.</b> ASan reports <code>located 12 bytes after 16-byte region</code> for an <code>int</code> buffer. Which index was accessed?</summary>

16 bytes is 4 ints. The access is 12 bytes past the end, so at byte offset 28, which is index 7.
</details>

<details><summary><b>6.</b> Why does RAII make a function safe on every exit path, including an early <code>return</code> or an exception?</summary>

The destructor of a stack object runs automatically when the object's scope ends, however the scope is left. If the destructor frees the block, there is no separate `delete` line to forget or to skip.
</details>

<details><summary><b>7.</b> When would you pick <code>shared_ptr</code> over <code>unique_ptr</code>, and what does it cost?</summary>

Only when several owners really share the lifetime of one block, for example a cached prefix used by several requests. It adds a reference count (typically 16 bytes for the pointer plus a control block) and updates it atomically on every copy, which is costly in hot loops or across threads. <code>unique_ptr</code> is one pointer with no counting, and it is the default.
</details>

<details><summary><b>8.</b> A <code>std::vector&lt;float&gt;</code> holds 1,000,000 floats. How many bytes of float data does <code>auto b = a;</code> copy, and how many does <code>auto c = std::move(a);</code> copy?</summary>

The copy duplicates the buffer: 1,000,000 × 4 = 4,000,000 bytes and one new allocation. The move copies none: `c` takes the pointer, size and capacity, and `a` is left empty.
</details>
