# Exercise 2: Dangling return (easy)

`first_squares` fills a local array and returns a pointer to it. The array lives in the function's stack frame. When the function returns, the frame is released and the pointer points at memory that will soon hold something else. This is a **dangling pointer**.

Build the starter with `-Wall` and read the compiler's warning: "address of local variable returned".

**Task:** change the API so it returns the data **by value**: make `first_squares` return a `std::vector<int>`. The vector owns a heap buffer, and returning it moves that buffer to the caller without copying elements.

**Hints**

1. Do not "fix" it by making the array `static`. The second test calls the function twice and keeps both results.
2. Do not return a pointer from `new` either. Then the caller has to remember to `delete[]`, which is what we are trying to avoid.
3. `std::vector<int> out; out.push_back(x);` is all you need.

**Test:** `ctest -R 02-dangling-return --output-on-failure`. The test overwrites the stack between the call and the reads, so a dangling pointer returns garbage (or crashes, since the compiler may return `nullptr`).
