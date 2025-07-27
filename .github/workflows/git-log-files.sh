git ls-tree -r --name-only HEAD | while read file; do
  echo "===== $file ====="
  git show HEAD:"$file"
  echo
done > full_repo_last_commit.txt