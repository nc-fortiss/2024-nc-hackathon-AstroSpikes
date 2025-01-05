#!/bin/bash

# Configuration variables
REPOURL="git@github.com:nc-fortiss/2024-nc-hackathon-AstroSpikes.git"    # SSH URL of the remote git repository
REPOPATH="/tmp/2024-nc-hackathon-AstroSpikes"             # IMPORTANT: Must be an absolute path to the repository
BRANCHNAME="autostart"                        # Branch name to monitor
JOBFILE="/tmp/training.log"               # Path to store commit information
TAGFILE="/tmp/processed_tags.txt"         # File to store processed tags
COMMAND="echo 'Running training for TAG: $TAG'"  # Command to execute for each tag
OUTFILE="/tmp/output.log"                # Path to store COMMAND output
CLONED=0
STARTDIR=$(pwd)

# Function to ensure directory exists
ensure_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
}

# Function to ensure file exists
ensure_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        ensure_dir "$(dirname "$file")"
        touch "$file"
        echo "Created file: $file"
    fi
}

cleanup() {
    # Change to the original branch
    git switch "$STARTBRANCH"
    # Change back to the original directory
    cd "$STARTDIR" || exit 1
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Step 1: Check and prepare repository
if [ ! -d "$REPOPATH" ]; then
    # Create parent directory if it doesn't exist
    ensure_dir "$(dirname "$REPOPATH")"
    
    # Clone the repository
    git clone "$REPOURL" "$REPOPATH"
    if [ $? -ne 0 ]; then
        echo "Failed to clone repository"
        exit 1
    fi
    CLONED=1
fi

# Change to repository directory
cd "$REPOPATH" || exit 1

STARTBRANCH=$(git branch --show-current)

# Step 2: Fetch latest changes from remote
git fetch origin "$BRANCHNAME"

# Pull the latest changes
git switch "$BRANCHNAME"
git pull origin "$BRANCHNAME"

# Ensure TAGFILE exists
ensure_file "$TAGFILE"

# Step 3: Check for unprocessed tags and run training
# Extract all tags matching the pattern RUNXXX
TAGS=$(git tag -l "RUN[0-9]*")

for TAG in $TAGS; do
    if ! grep -Fxq "$TAG" "$TAGFILE"; then
        # New tag detected, process it
        echo "Processing tag: $TAG"

        # Checkout the tag
        git checkout "$TAG"

        # Run the command for the tag
        {
            echo "=== Processing TAG: $TAG at $(date '+%Y-%m-%d %H:%M:%S') ==="
            eval "$COMMAND"
            echo "=== Finished processing TAG: $TAG at $(date '+%Y-%m-%d %H:%M:%S') ==="
            echo ""
        } >> "$OUTFILE" 2>&1

        # Append the tag to TAGFILE
        echo "$TAG" >> "$TAGFILE"

        echo "Tag $TAG processed successfully"
    else
        echo "Tag $TAG already processed"
    fi
done

# Step 4: Cleanup
cleanup
