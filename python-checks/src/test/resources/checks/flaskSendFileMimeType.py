from flask import send_file
from io import BytesIO, StringIO

def download_with_open_file():
    file_obj = open('data.txt', 'rb')
    return send_file(file_obj)  # Noncompliant {{Provide "mimetype" or "download_name" when calling "send_file" with a file-like object.}}
#          ^^^^^^^^^

def download_with_bytesio():
    csv_data = BytesIO(b'name,age\nJohn,30')
    return send_file(csv_data)  # Noncompliant

def download_with_stringio():
    log_data = StringIO('INFO: Application started')
    return send_file(log_data)  # Noncompliant

def compliant_with_mimetype():
    file_obj = open('data.txt', 'rb')
    return send_file(file_obj, mimetype='text/plain')

def compliant_with_download_name():
    csv_data = BytesIO(b'name,age\nJohn,30')
    return send_file(csv_data, download_name='data.csv', as_attachment=True)

def compliant_with_attachment_filename():
    csv_data = BytesIO(b'name,age\nJohn,30')
    return send_file(csv_data, attachment_filename='data.csv')

def compliant_with_both():
    log_data = StringIO('INFO: Application started')
    return send_file(log_data, mimetype='text/plain', download_name='app.log')

def compliant_with_string_path():
    return send_file('data.txt')

def compliant_with_path_object():
    from pathlib import Path
    return send_file(Path('data.txt'))

def no_issue_for_non_flask_send_file():
    def send_file(file_obj):
        pass
    
    file_obj = open('data.txt', 'rb')
    send_file(file_obj)

def no_issue_when_no_arguments():
    send_file()
