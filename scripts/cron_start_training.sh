#!/bin/bash

# Configuration variables
REPOURL="git@github.com:username/repo.git"    # SSH URL of the remote git repository
REPOPATH="/absolute/path/to/repo"             # IMPORTANT: Must be an absolute path to the repository
BRANCHNAME="autostart"                        # Branch name to monitor
JOBFILE="/path/to/job/data.log"              # Path to store commit information
COMMAND="source new_job"                      # Command to execute when new changes are detected
OUTFILE="/path/to/output.log"                # Path to store COMMAND output

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
fi

# Change to repository directory
cd "$REPOPATH" || exit 1

# Step 2: Check for new commits
# Fetch latest changes from remote
git fetch origin "$BRANCHNAME"

# Get the latest commit hash from remote and local
REMOTE_HASH=$(git rev-parse "origin/$BRANCHNAME")
LOCAL_HASH=$(git rev-parse HEAD)

# Step 3: If there are new changes, process them
if [ "$REMOTE_HASH" != "$LOCAL_HASH" ]; then
    # Pull the latest changes
    git pull origin "$BRANCHNAME"
    
    # Ensure JOBFILE exists
    ensure_file "$JOBFILE"
    
    # Get commit information and append to JOBFILE
    {
        echo "---"
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Commit Hash: $REMOTE_HASH"
        echo "Commit Message: $(git log -1 --pretty=%B)"
        echo "Commit Date: $(git log -1 --pretty=%cd --date=format:'%Y-%m-%d %H:%M:%S')"
        echo ""
    } >> "$JOBFILE"
    
    # Ensure OUTFILE exists
    ensure_file "$OUTFILE"
    
    # Execute the command and redirect output
    {
        echo "=== Command execution started at $(date '+%Y-%m-%d %H:%M:%S') ==="
        eval "$COMMAND"
        echo "=== Command execution finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
        echo ""
    } >> "$OUTFILE" 2>&1
    
    echo "Changes processed successfully"
else
    echo "No new changes detected"
fi