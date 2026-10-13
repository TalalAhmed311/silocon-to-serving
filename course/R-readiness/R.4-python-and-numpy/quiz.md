# R.4 quiz

<details><summary><b>1.</b> How many bytes is a <code>(4096, 4096)</code> bfloat16 matrix, and how does that compare with float32?</summary>

4096 x 4096 = 16,777,216 elements x 2 bytes = 33,554,432 bytes = **32 MiB**. float32 uses 4 bytes per element, so it is 64 MiB, twice as much.
</details>

<details><summary><b>2.</b> <code>a</code> is a float32 array of shape (3, 4). What are <code>a.strides</code>, and the strides of <code>a.T</code>?</summary>

`a.strides` is `(16, 4)`: a row is 4 floats x 4 bytes = 16 bytes; a column step is 4 bytes. `a.T` has shape (4, 3) and strides `(4, 16)`. No data moved.
</details>

<details><summary><b>3.</b> <code>b = a[:, 0]; b[:] = 0</code>. Does <code>a</code> change? What if you wrote <code>b = a[:, [0]]</code>?</summary>

Yes, the first is a slice, so `b` is a view and `a`'s first column becomes 0. The second uses a list as an index (fancy indexing), which always copies, so `a` is unchanged.
</details>

<details><summary><b>4.</b> Which of these shape pairs can broadcast, and what is the result? (a) (8, 1, 6) and (7, 1); (b) (5, 3) and (5,); (c) (2, 1) and (1, 4).</summary>

Align from the right. (a) (8, 1, 6) with (7, 1): 6 vs 1 ok, 1 vs 7 ok, 8 vs missing ok, result **(8, 7, 6)**. (b) 3 vs 5 fails: **error**. (c) 1 vs 4 and 2 vs 1 both stretch: **(2, 4)**.
</details>

<details><summary><b>5.</b> You multiply (32, 4096) by (4096, 11008). What is the output shape and the FLOP count?</summary>

Shape (32, 11008). FLOPs = 2 x 32 x 4096 x 11008 = 2,885,681,152, about 2.9 GFLOPs.
</details>

<details><summary><b>6.</b> A vectorised function takes 3 ms and your loop took 90 ms. What can you conclude about the speed-up on another machine?</summary>

Only that on your machine, for those array sizes, it was about 30x. The ratio depends on array size, hardware, dtype and library versions, so measure again wherever it matters.
</details>

<details><summary><b>7.</b> Why is <code>assert fast_result == reference_result</code> a bad test for float arrays, and what do you write instead?</summary>

Two correct computations that add numbers in a different order can differ in the last bits, and `==` on arrays does not even give one answer. Use `np.testing.assert_allclose(fast, reference, rtol=1e-5, atol=1e-8)` with tolerances suited to the dtype.
</details>
