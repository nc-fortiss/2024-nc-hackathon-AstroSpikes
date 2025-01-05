# `cron_start_training.sh` Wiki

## Table of Contents
1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Parameters](#parameters)
4. [Installation](#installation)


## Overview
The `cron_start_training.sh` script automates the detection and processing of new Git tags with the pattern `RUNxxx` (where `xxx` are alphanumeric characters and underscores). Make sure to make your tag unique! It is designed to monitor a specific Git branch and trigger training or other predefined tasks whenever it encounters an unprocessed tag.

This script is intended to be run periodically using `crontab`.

---

## How It Works
1. **Repository Setup**:
   - The script checks if the repository is already cloned. If not, it clones the repository from the configured remote URL.
   - It then switches to the specified branch (`BRANCHNAME`) and pulls the latest changes.

2. **Tag Detection**:
   - It lists all tags in the repository matching the pattern `RUNXXX`.
   - For each tag, it checks if the tag has already been processed by looking for it in the `processed_tags.txt` file.

3. **Tag Processing**:
   - If a tag is new, the script:
     1. Checks out the tag.
     2. Executes the training command (`COMMAND`) while logging its output to `output.log`.
     3. Marks the tag as processed by appending it to `processed_tags.txt`.

4. **Logging**:
   - The script logs key operations and outputs to specified files:
     - `output.log`: Stores the output of the training command.
     - `training.log`: Records metadata about processed tags.

5. **Repetition**:
   - The script is intended to run periodically (e.g., every 10 minutes) using a `crontab` entry.

---

## Parameters
Below are the key parameters and variables in the script:

### Configurable Variables
| **Variable**       | **Description**                                                                 |
|---------------------|---------------------------------------------------------------------------------|
| `REPOURL`          | The SSH URL of the Git repository to monitor.                                   |
| `REPOPATH`         | Absolute path to where the repository will be cloned locally.                   |
| `BRANCHNAME`       | The branch name in the repository to monitor for new tags.                      |
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
The script uses `/tmp/` directories for temporary storage. To avoid triggering unnecessary training due to `/tmp` being erased, update these paths in the [configuration variables](#configurable-variables) to a non-volatile location.

### 2. Make the Script Executable
Ensure the script is executable by running:
```bash
chmod +x /path/to/cron_start_training.sh
```

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