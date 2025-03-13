import os
import glob


# Rename all synthetic trajectories to seq_RTXXX.csv to STXXX.csv
def rename_syn_csv_files(folder_path):
    """Renames all 'seq_RTXXX.csv' files to 'STXXX.csv' in the given folder."""
    for file_path in glob.glob(os.path.join(folder_path, "seq_RT*.csv")):
        filename = os.path.basename(file_path)  # Extract filename
        new_filename = filename.replace("seq_RT", "RT")  # Rename pattern
        new_path = os.path.join(folder_path, new_filename)

        os.rename(file_path, new_path)  # Rename file
        print(f"Renamed: {filename} ➝ {new_filename}")


def rename_syn_folders(directory_path):
    """Renames all folders 'seq_RTXXX' to 'RTXXX' in the given directory."""
    # Find all directories that match the pattern 'seq_RT*'
    for folder_path in glob.glob(os.path.join(directory_path, "seq_RT*")):
        # Get the folder name
        folder_name = os.path.basename(folder_path)

        # Check if it's a folder and matches the pattern
        if os.path.isdir(folder_path):
            new_folder_name = folder_name.replace("seq_", "")  # Remove 'seq_' prefix
            new_folder_path = os.path.join(directory_path, new_folder_name)

            # Rename the folder
            os.rename(folder_path, new_folder_path)
            print(f"Renamed: {folder_name} ➝ {new_folder_name}")


if __name__ == '__main__':
    folder_path = "/home/ak/datasets/Event_Dataset/SPADES/synthetic/events"
    rename_syn_csv_files(folder_path)

    folder_path = "/home/ak/datasets/Event_Dataset/SPADES/synthetic/labels"
    rename_syn_csv_files(folder_path)

    # directory_path = "/home/arunkumar/datasets/SPADES/synthetic/lnes"
    # rename_syn_folders(directory_path)
