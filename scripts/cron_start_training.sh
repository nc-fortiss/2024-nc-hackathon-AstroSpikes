#!/bin/bash

# Configuration variables
REPOURL="git@github.com:nc-fortiss/2024-nc-hackathon-AstroSpikes.git"
REPOPATH="/home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes"
JOBFILE="/home/lecomte/AstroSpikes/training.log"
TAGFILE="/home/lecomte/AstroSpikes/processed_tags.txt"
COMMAND="sbatch /home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes/astrospikes_train.sh" # REPLACE THIS COMMAND TO START TRAINING
OUTFILE="/home/lecomte/AstroSpikes/output.log"
CLONED=0
STARTDIR=$(pwd)

export WANDB_API_KEY=d8eb14aa69c0f4a4cc666324156979070f9ccb7b

ensure_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
}

ensure_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        ensure_dir "$(dirname "$file")"
        touch "$file"
        echo "Created file: $file"
    fi
}

cleanup() {
    git switch "$STARTBRANCH"
    cd "$STARTDIR" || exit 1
}

is_valid_tag() {
    [[ "$1" =~ ^RUN[0-9A-Za-z_]*$ ]] || return 1
}

get_branches () {
  git branch --remotes --format='%(refname:short)'
}

get_run_tags() {
   git tag -l "RUN[0-9A-Za-z_]*"
}

# takes function name as argument
# returns array of lines from command output

cmd_to_array() {
   local cmd=$1
   tmp_arr=()
   local tempfile="/tmp/cmd_output_$$.txt"
   
   $cmd | grep -v HEAD | sed 's/origin\///' | tr -d ' ' | grep -v '^$' > "$tempfile"
   while IFS= read -r line; do
    tmp_arr+=($line)
   done < $tempfile
   rm "$tempfile"
   echo ${tmp_arr[*]}
}


trap cleanup EXIT INT TERM

# Repository setup
if [ ! -d "$REPOPATH" ]; then
    ensure_dir "$(dirname "$REPOPATH")"
    git clone "$REPOURL" "$REPOPATH"
    if [ $? -ne 0 ]; then
        echo "Failed to clone repository"
        exit 1
    fi
    CLONED=1
fi

cd "$REPOPATH" || exit 1

STARTBRANCH=$(git branch --show-current)

echo "Fetching latest changes"
# Fetch all branches and tags
git fetch --all
git fetch --tags --all

TAGS=($(cmd_to_array get_run_tags))
echo ${TAGS[*]}
echo "Processing tags in branch: $BRANCH"
for TAG in ${TAGS[*]}; do
    if ! is_valid_tag "$TAG"; then
        echo "Invalid tag: $TAG, skipping"
        continue
    fi
    if ! grep -Fq "$TAG" "$TAGFILE"; then
        echo "=== Processing TAG: $TAG (Branch: $BRANCH) at $(date '+%Y-%m-%d %H:%M:%S') ==="
        
        git checkout "$TAG"
        
        {
            echo "=== Processing TAG: $TAG (Branch: $BRANCH) at $(date '+%Y-%m-%d %H:%M:%S') ==="
            eval "$COMMAND"
            echo "=== Finished processing TAG: $TAG (Branch: $BRANCH) at $(date '+%Y-%m-%d %H:%M:%S') ==="
            echo ""
        } >> "$OUTFILE" 2>&1
        
        echo "=== Finished processing TAG: $TAG (Branch: $BRANCH) at $(date '+%Y-%m-%d %H:%M:%S') ==="
        echo "$TAG - $(date '+%Y-%m-%d %H:%M:%S')" >> "$TAGFILE"
    else
        echo "Tag $TAG already processed"
    fi
done

#cleanup