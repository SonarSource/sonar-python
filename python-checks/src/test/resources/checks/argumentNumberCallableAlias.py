import tarfile
from tarfile import open as open_archive


def archive_calls():
    archive = tarfile.open("example.tar", "w")
    archive.add("file.txt", exclude=lambda name: False)  # Noncompliant
    archive.add("file.txt", nonexistent=True)  # Noncompliant
    archive.add("file.txt", recursive=False)
    archive.close()
    other = open_archive("example.tar")
    other.add("file.txt", nonexistent=True)  # Noncompliant
    other.close()
