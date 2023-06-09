import os
import shutil

CURRENT_PATH = os.path.dirname(__file__)


class FolderManager:
     
    output_folder: str = os.path.join(CURRENT_PATH, "../../src/main/resources/org/sonar/python/types")
    license_file = os.path.join(output_folder, "LICENSE")

    def cleanup_output_folder(self):
        for filename in os.listdir(self.output_folder):
            filepath = os.path.join(self.output_folder, filename)
            if filepath != self.license_file:
                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)


def main():
    FolderManager().cleanup_output_folder()


if __name__ == '__main__':
    main()
