import os

def rename_files(folder_path, naming_pattern, start_index=1):
    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return
    
    # Get a list of files in the folder
    files = os.listdir(folder_path)
    
    # Loop through the files and rename them
    for index, filename in enumerate(files, start=start_index):
        # Create the new filename based on the pattern
        new_filename = naming_pattern.format(index)
        
        # Construct the full paths for old and new filenames
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        
        print(f"Renamed: {filename} => {new_filename}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path: ")
    naming_pattern = input("Enter the naming pattern (use '{}' as a placeholder for the index): ")
    start_index = int(input("Enter the starting index (default is 1): ") or 1)
    
    rename_files(folder_path, naming_pattern, start_index)
