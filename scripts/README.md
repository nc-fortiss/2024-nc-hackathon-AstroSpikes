# `cron_start_training.sh` Wiki

## Table of Contents
1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Parameters](#parameters)
4. [Installation](#installation)


## Overview
The `cron_start_training.sh` script automates the detection and processing of all Git tags in a repository that match the pattern `RUNXXX` (or `RUN[0-9A-Za-z_]*` for extended tag names). It ensures no duplicate processing by maintaining a log of processed tags. The script is designed to consider **all tags across the repository** instead of limiting itself to a single branch.

This script is intended to be run periodically using `crontab`.

---

## How It Works
1. **Repository Setup**:
   - The script ensures the repository is cloned locally. If not, it clones the repository from the configured `REPOURL`.
   - It fetches and updates all branches and tags in the repository.

2. **Tag Detection**:
   - The script lists all tags in the repository matching the pattern `RUN[0-9A-Za-z_]*`.
   - For each tag, it checks if it has already been processed using `processed_tags.txt`.

3. **Tag Validation**:
   - Tags must follow the pattern `RUN[0-9A-Za-z_]*`.
   - Invalid tags are ignored.

4. **Tag Processing**:
   - If a tag is valid and unprocessed, the script:
     1. Checks out the commit associated with the tag.
     2. Executes the training command (`COMMAND`), logging the output to `output.log`.
     3. Marks the tag as processed in `processed_tags.txt`.

5. **Repetition**:
   - The script is designed to run periodically using `crontab` to detect and process new tags.

---

## Parameters
Below are the key parameters and variables in the script:

### Configurable Variables
| **Variable**       | **Description**                                                                 |
|---------------------|---------------------------------------------------------------------------------|
| `REPOURL`          | The SSH URL of the Git repository to monitor.                                   |
| `REPOPATH`         | Absolute path to where the repository will be cloned locally.                   |
| `JOBFILE`          | Path to the file where processed commit information is logged.                  |
| `TAGFILE`          | Path to the file where processed tags are stored to avoid duplicate processing. |
| `COMMAND`          | The command to execute for each unprocessed tag (e.g., training).               |
| `OUTFILE`          | Path to the file where the output of the `COMMAND` is logged.                   |

### Key Files
- **`processed_tags.txt`**: Tracks which tags have already been processed.
- **`training.log`**: Logs information about processed tags.
- **`output.log`**: Captures the output of the training command.

---

## Installation

### 1. Update the Script Paths
By default, the script uses `/tmp/` for storage. Update these paths to a non-volatile directory to prevent data and state loss across reboots.

### 2. Make the Script Executable
Ensure the script is executable:
```bash
chmod +x /path/to/cron_start_training.sh

### 3. Register script with crontab
I suggest using crontab instead of cron for scheduling script execution. It should be already installed in the Ubuntu container.
To register the script with crontab, open crontab file by running:
```bash
crontab -e
```

Paste this line at the end to execute the script every 10 minutes:

```bash
*/10 * * * * bash /path/to/cron_start_training.sh
```

## Tagging commits in git repo
Precautions:
- Make sure you are on the right branch - the same as specified in the script
- Use the lightweight tag, it's enough

For details see: https://git-scm.com/book/en/v2/Git-Basics-Tagging

Tagging cheat-sheet:

Create lightweight tag for the last commit:
```bash
git tag RUN001
```

Create lightweight tag for a commit from history:
List tags:
```bash
git log --pretty=oneline
```
```bash
# git tag <tag_name> <commit_hash>
git tag RUN001 9fceb02
```

### Cherry-picking commits from your branch
TBD