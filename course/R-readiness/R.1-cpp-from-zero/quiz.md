# R.1 quiz

<details><summary><b>1.</b> You compile each file with <code>g++ -c</code> and then run <code>g++ main.o -o app</code>, but <code>main.cpp</code> calls <code>square()</code>, defined in <code>mathlib.cpp</code>. Which error do you get, and from which tool?</summary>

A **link error**: `undefined reference to 'square'`. It comes from the linker, not the compiler. `main.o` holds a placeholder for `square`, and no object file given to the linker defines it. Add `mathlib.o` to the link command.
</details>

<details><summary><b>2.</b> What are <code>sizeof(char)</code>, <code>sizeof(int)</code>, <code>sizeof(double)</code> and <code>sizeof(int*)</code> on a 64-bit Linux machine? How many bytes does <code>float w[1000]</code> take?</summary>

1, 4, 8 and 8. `float w[1000]` takes 1000 × 4 = 4000 bytes. A pointer is 8 bytes whatever it points at.
</details>

<details><summary><b>3.</b> <code>int a[5]</code> starts at address <code>0x1000</code>. What is the address of <code>&amp;a[3]</code>, and what is <code>*(a + 3)</code> the same as?</summary>

`0x1000 + 3 × 4 = 0x100c`. `*(a + 3)` is exactly `a[3]`: indexing is pointer arithmetic, and `a + 3` moves by 3 elements, not 3 bytes.
</details>

<details><summary><b>4.</b> After <code>int n = 5; f(n); g(n); h(&amp;n);</code> where <code>f(int x)</code>, <code>g(int&amp; x)</code> and <code>h(int* x)</code> each add 1 to what they were given, what is <code>n</code>?</summary>

**7.** `f` changes a private copy. `g` changes `n` through a reference. `h` changes `n` through the pointer.
</details>

<details><summary><b>5.</b> Why is <code>void print(const std::vector&lt;float&gt;&amp; v)</code> usually better than <code>void print(std::vector&lt;float&gt; v)</code>? When is pass by value the right choice?</summary>

Passing by value copies the vector and all its elements (4 MB for a million floats). `const&` copies only an address and the compiler stops the function from modifying the data. Small cheap types such as `int`, `float` and pointers are fine, and often faster, by value.
</details>

<details><summary><b>6.</b> A header contains a full function definition without <code>inline</code> and two <code>.cpp</code> files include it. What does the linker say, and which rule is broken?</summary>

`multiple definition of ...`. The one-definition rule: a function may be defined only once in the whole program. Each `.cpp` got its own copy through `#include`. Mark the function `inline`, or move the body into one `.cpp`.
</details>

<details><summary><b>7.</b> In <code>{ Logger a("a"); Logger b("b"); }</code> in what order do the destructors run?</summary>

`b` first, then `a`: the reverse of construction. The closing brace runs them automatically.
</details>

<details><summary><b>8.</b> <code>int a[4]; int v = a[4];</code> compiles without error. Why, and what does it do?</summary>

C++ does not check array bounds. Reading `a[4]` is undefined behaviour, so anything may happen: a crash, a garbage value, or a value that looks right. Building with `-fsanitize=address,undefined -g` makes it a certain, reported failure.
</details>
