#!/bin/bash

# Configuration variables
REPOURL="git@github.com:nc-fortiss/2024-nc-hackathon-AstroSpikes.git"
REPOPATH="/tmp/2024-nc-hackathon-AstroSpikes"
JOBFILE="/tmp/training.log"
TAGFILE="/tmp/processed_tags.txt"
COMMAND="echo 'Running training for TAG: $TAG'"
OUTFILE="/tmp/output.log"
CLONED=0
STARTDIR=$(pwd)

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

# Fetch all branches and tags
git fetch --all
git fetch --tags

# Get list of all branches (local and remote)
BRANCHES=$(git branch -r | grep -v HEAD | sed 's/origin\///' && git branch --list | sed 's/^* //')
BRANCHES=$(echo "$BRANCHES" | sort -u)

ensure_file "$TAGFILE"

# Process each branch
for BRANCH in $BRANCHES; do
    echo "Processing branch: $BRANCH"
    
    # Try to switch to branch
    if ! git switch "$BRANCH" 2>/dev/null; then
        if ! git switch "origin/$BRANCH" 2>/dev/null; then
            echo "Cannot switch to branch $BRANCH, skipping"
            continue
        fi
    fi
    
    git pull origin "$BRANCH" 2>/dev/null

    # Process tags in this branch
    TAGS=$(git tag -l "RUN[0-9A-Za-z_]*")
    
    echo "Processing tags in branch: $BRANCH"
    for TAG in $TAGS; do
        if ! grep -Fxq "$TAG" "$TAGFILE"; then
            echo "Processing tag: $TAG in branch: $BRANCH"
            
            git checkout "$TAG"
            
            # {
                echo "=== Processing TAG: $TAG (Branch: $BRANCH) at $(date '+%Y-%m-%d %H:%M:%S') ==="
                eval "$COMMAND"
                echo "=== Finished processing TAG: $TAG (Branch: $BRANCH) at $(date '+%Y-%m-%d %H:%M:%S') ==="
                echo ""
            # } >> "$OUTFILE" 2>&1
            
            echo "$TAG" >> "$TAGFILE"
            echo "Tag $TAG processed successfully"
        else
            echo "Tag $TAG already processed"
        fi
    done
done

cleanup