from pathlib import Path
import hashlib
import io
import zipfile
import pytest
from drugshift.prepare import download_resume,unzip_transactional,checksum

def test_zip_traversal_rejected(tmp_path):
    path=tmp_path/'test.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('../attack.txt','not allowed')
    with pytest.raises(ValueError):unzip_transactional(path,tmp_path/'out')

def test_existing_download_checksum(tmp_path):
    f=tmp_path/'data.zip';f.write_bytes(b'correct offline data')
    md=hashlib.md5(f.read_bytes()).hexdigest()
    assert download_resume('https://invalid.example.test/file',f,md)==f
