# Exercise 3 — A git checklist (easy, no test)

You will make a scratch repository, commit, branch, cause a merge conflict on purpose and resolve it. There is no automated test. The check is that your `git log --oneline --graph` has the shape shown at the end.

Work **outside the course repo**, so nothing you do here can touch your real work:

```bash
mkdir -p /tmp/git-practice && cd /tmp/git-practice
```

If git says it does not know who you are, set a name and email for this repo only: `git config user.name "Your Name"` and `git config user.email "you@example.com"`.

## Tasks

- [ ] **1. Init.** `git init -b main` then `git status`. It says "No commits yet".
- [ ] **2. First commit.** Make two files, stage them, commit.
  ```bash
  echo "# notes" > notes.md
  printf 'line one\nline two\nline three\n' > todo.txt
  git add notes.md todo.txt
  git status                 # both files are "Changes to be committed" (staged)
  git commit -m "Add notes and todo"
  ```
- [ ] **3. Look.** `git log --oneline` shows one commit. `git show --stat` shows what it changed.
- [ ] **4. Branch.** Create a branch and switch to it in one step, then edit line two and commit.
  ```bash
  git switch -c feature      # the older spelling is: git checkout -b feature
  sed -i 's/line two/line two (feature)/' todo.txt     # on macOS use: sed -i '' ...
  git diff                   # unstaged change: working tree vs staging
  git commit -am "Edit line two on feature"
  ```
- [ ] **5. Diverge.** Go back to `main` and change the **same line** differently.
  ```bash
  git switch main
  sed -i 's/line two/line two (main)/' todo.txt
  git commit -am "Edit line two on main"
  ```
- [ ] **6. Merge and get a conflict.** `git merge feature`. Git prints `CONFLICT (content): Merge conflict in todo.txt`. This is normal, not an error in your repo. `git status` lists `todo.txt` under "Unmerged paths".
- [ ] **7. Read the markers.** Open `todo.txt`:
  ```
  line one
  <<<<<<< HEAD
  line two (main)
  =======
  line two (feature)
  >>>>>>> feature
  line three
  ```
  The part above `=======` is your current branch. The part below is the branch you are merging.
- [ ] **8. Resolve.** Edit the file to the text you want and delete all three marker lines. For example:
  ```
  line one
  line two (both)
  line three
  ```
  Then `git add todo.txt` (this tells git the conflict is resolved) and `git commit -m "Merge feature, resolve line two"`.
- [ ] **9. Check the shape.** `git log --oneline --graph` must look like this (your hashes differ):
  ```
  *   962409d Merge feature, resolve line two
  |\
  | * 6092999 Edit line two on feature
  * | ec03092 Edit line two on main
  |/
  * 5c7bdd2 Add notes and todo
  ```
  `git status -sb` should print only `## main`.
- [ ] **10. Undo safely.** Change a file without staging it, then `git restore todo.txt` puts it back. Stage a change, then `git restore --staged todo.txt` unstages it. These commands only touch your working copy and staging area, never your history.

## Answer these in your own words

1. What is the difference between the working tree, the staging area and a commit?
2. Why did step 6 conflict, when merging a branch that changed a **different** line would not?
3. What does `git push` do that `git commit` does not? (In this scratch repo it has nowhere to push to. In the course repo it sends your commits to a remote.)

## Rules to keep when you work in the real repo

- Commit small, with a message that says why.
- Never commit secrets, keys or `terraform.tfvars` (R.7 covers this).
- Do not run `git push --force` or `git reset --hard` unless you know exactly which commits you are about to lose.
