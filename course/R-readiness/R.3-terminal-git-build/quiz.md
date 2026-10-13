# R.3 quiz

<details><summary><b>1.</b> You run <code>make</code>, it prints errors, and then you type <code>echo $?</code>. It prints <code>2</code>. What does that mean?</summary>

The last command failed. Exit code `0` means success; any other number means failure, and the number is a hint chosen by the program. Tools such as CTest and `&&` decide pass or fail from this code alone.
</details>

<details><summary><b>2.</b> What is the difference between <code>cmd &gt; out.txt</code> and <code>cmd &gt; out.txt 2&gt;&amp;1</code>?</summary>

The first writes only stdout to the file; error messages still go to the terminal. The second sends stderr (stream 2) to wherever stdout (stream 1) goes, so the file holds both.
</details>

<details><summary><b>3.</b> What does <code>grep -rn "TODO(run" course | sort | uniq -c | sort -nr | head</code> do, step by step?</summary>

`grep -rn` prints every matching line with file name and line number. `sort` orders those lines. `uniq -c` collapses identical neighbours and prefixes a count. `sort -nr` orders by that count, largest first. `head` keeps the first ten lines. Each program reads the previous one's stdout through a pipe.
</details>

<details><summary><b>4.</b> Why do the course's tests read <code>S2S_SOLUTIONS</code> from the environment, and how do you set it for one command only?</summary>

It lets the same test file check either your starter or the reference solution without editing the test. Set it for one command by putting it in front: <code>S2S_SOLUTIONS=1 uv run pytest path</code>. <code>export S2S_SOLUTIONS=1</code> keeps it for the rest of the shell session.
</details>

<details><summary><b>5.</b> You edited the staging area with <code>git add</code> but have not committed. Where is your change, and is it saved in history?</summary>

It is in the staging area, a list of what the next commit will contain. It is not in history yet. Only <code>git commit</code> writes a commit.
</details>

<details><summary><b>6.</b> After a merge, <code>todo.txt</code> contains <code>&lt;&lt;&lt;&lt;&lt;&lt;&lt; HEAD</code>, <code>=======</code> and <code>&gt;&gt;&gt;&gt;&gt;&gt;&gt; feature</code> lines. What do you do?</summary>

Edit the file to the text you want and delete the three marker lines. Then `git add todo.txt` to mark it resolved, and `git commit` to finish the merge.
</details>

<details><summary><b>7.</b> A project has <code>a.cpp</code>, <code>b.cpp</code> and <code>util.hpp</code> (included by <code>b.cpp</code> only). You edit <code>util.hpp</code>. What does the build system recompile?</summary>

`b.cpp` (to a new `b.o`), then it re-links the executable. `a.o` is still up to date because nothing it depends on changed.
</details>

<details><summary><b>8.</b> The build fails with <code>undefined reference to 'add(int, int)'</code>, and the header declares <code>add</code>. What is wrong?</summary>

This is a linker error, not a compiler error. The declaration was seen but no object file that defines <code>add</code> was given to the linker. In CMake, add the source file to a target and link that target (<code>target_link_libraries</code>).
</details>
