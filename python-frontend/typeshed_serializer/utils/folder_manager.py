import os
import shutil
import sys

CURRENT_PATH = os.path.dirname(__file__)


class FolderManager:
    output_folder: str = os.path.join(CURRENT_PATH, "../../src/main/resources/org/sonar/python/types")
    license_file = os.path.join(output_folder, "LICENSE")

    # Mapping of serializer names to their output directories
    SERIALIZER_FOLDERS = {
        "stdlib": "stdlib_protobuf",
        "third_party": "third_party_protobuf",
        "custom": "custom_protobuf",
        "importer": "third_party_protobuf_mypy",
        "microsoft": "third_party_protobuf_microsoft"
    }

    def cleanup_output_folder(self, selective_folders=None):
        """
        Clean up output folder. If selective_folders is provided, only clean those folders.
        Otherwise, clean everything except LICENSE.

        Args:
            selective_folders: List of serializer names (e.g., ['custom', 'stdlib'])
        """
        if selective_folders:
            # Selective cleanup - only remove specified folders
            for serializer_name in selective_folders:
                if serializer_name in self.SERIALIZER_FOLDERS:
                    folder_name = self.SERIALIZER_FOLDERS[serializer_name]
                    folder_path = os.path.join(self.output_folder, folder_name)
                    if os.path.exists(folder_path):
                        print(f"Cleaning up {folder_name} folder...")
                        shutil.rmtree(folder_path)
                else:
                    print(f"Warning: Unknown serializer '{serializer_name}'")
        else:
            # Full cleanup - remove everything except LICENSE
            for filename in os.listdir(self.output_folder):
                filepath = os.path.join(self.output_folder, filename)
                if filepath != self.license_file:
                    try:
                        shutil.rmtree(filepath)
                    except OSError:
                        os.remove(filepath)


def main():
    # Check if specific folders were provided as command line arguments
    if len(sys.argv) > 1:
        # Selective cleanup mode
        serializer_names = sys.argv[1].split(",")
        FolderManager().cleanup_output_folder(serializer_names)
    else:
        # Full cleanup mode
        FolderManager().cleanup_output_folder()


if __name__ == '__main__':
    main()
