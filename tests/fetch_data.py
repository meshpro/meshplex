import hashlib
import os
import requests
import shutil


# The tests files are located on sourceforge.
def download_mesh(name, md5):

    filename = os.path.join('/tmp', name)
    if not os.path.exists(filename):
        print('Downloading %s...' % name)
        url = 'https://sourceforge.net/projects/meshzoo-data/files/'
        r = requests.get(url + name + '/download', stream=True)
        if not r.ok:
            raise RuntimeError(
                'Download error (%s, return code %s).' % (r.url, r.status_code)
                )

        # save the mesh in /tmp
        with open(filename, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

    # check MD5
    file_md5 = hashlib.md5(open(filename, 'rb').read()).hexdigest()

    if file_md5 != md5:
        raise RuntimeError(
            'Checksums not matching (%s != %s).' % (file_md5, md5)
            )

    return filename
