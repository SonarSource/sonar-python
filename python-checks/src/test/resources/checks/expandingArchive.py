import zipfile

archive_file = zipfile.ZipFile('attachment.zip', 'r')
archive_file.extractall('.') # Noncompliant
archive_file.close()

import tarfile

tar = tarfile.open("attachment.tar.gz") # Noncompliant
tar.extractall()
tar.close()
