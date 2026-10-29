"""Download processed The Way Up videos and labels without the 4K originals."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import ssl
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib

URL = 'https://zenodo.org/api/records/15196867/files/dataset.zip/content'


class RemoteArchive(io.RawIOBase):
    def __init__(self):
        import certifi
        self.context = ssl.create_default_context(cafile=certifi.where())
        self.position = 0
        self.size = None
        self.cache = None
        self.fetch(0, 1)

    def fetch(self, start, count):
        if count <= 0:
            return b''
        request = urllib.request.Request(URL, headers={
            'Range': f'bytes={start}-{start+count-1}',
            'User-Agent': 'BetaTranslatorResearch/0.1'})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, context=self.context, timeout=120) as response:
                    if response.status != 206:
                        raise IOError('Server must honor byte ranges.')
                    content_range = response.headers['Content-Range']
                    if not content_range.startswith(f'bytes {start}-'):
                        raise IOError('Unexpected byte range.')
                    self.size = int(content_range.split('/')[-1])
                    data = response.read()
                    if len(data) != min(count, self.size-start):
                        raise IOError('Incomplete response.')
                    return data
            except (urllib.error.URLError, TimeoutError, IOError) as error:
                if attempt == 4:
                    raise
                print(f'Retrying request: {type(error).__name__}', flush=True)
                time.sleep(min(30, 3 * 2**attempt))

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        count = self.size-self.position if count < 0 else count
        if self.cache is None or not (self.cache[0] <= self.position and self.position+count <= self.cache[0]+len(self.cache[1])):
            start = max(0, min(self.position, self.size-131072))
            self.cache = (start, self.fetch(start, min(self.size-start, max(count, 131072))))
        offset = self.position-self.cache[0]
        result = self.cache[1][offset:offset+count]
        self.position += len(result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    remote = RemoteArchive()
    with zipfile.ZipFile(remote) as archive:
        selected = [entry for entry in archive.infolist() if not entry.is_dir()
                    and '_large.' not in entry.filename and not entry.filename.startswith('__MACOSX/')]
    print(f'Selected {len(selected)} files, {sum(e.file_size for e in selected)/1e6:.1f} MB', flush=True)
    manifest = []
    for i, entry in enumerate(selected):
        path = args.output/entry.filename
        if args.output.resolve() not in path.resolve().parents:
            raise ValueError('Unsafe archive path.')
        if not path.exists() or path.stat().st_size != entry.file_size:
            packet = remote.fetch(entry.header_offset, min(entry.compress_size+65536, remote.size-entry.header_offset))
            if packet[:4] != b'PK\x03\x04':
                raise IOError('Invalid ZIP local header.')
            name_len, extra_len = struct.unpack_from('<HH', packet, 26)
            start = 30+name_len+extra_len
            compressed = packet[start:start+entry.compress_size]
            if len(compressed) != entry.compress_size:
                raise IOError('Truncated archive member.')
            data = zlib.decompress(compressed, -15) if entry.compress_type == 8 else compressed
            if entry.compress_type not in (0, 8) or len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
                raise IOError('Archive member integrity check failed.')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix+'.part')
            temporary.write_bytes(data)
            temporary.replace(path)
        data = path.read_bytes()
        if zlib.crc32(data) != entry.CRC:
            raise IOError(f'Cached member CRC mismatch: {entry.filename}')
        manifest.append(dict(path=entry.filename, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), zip_crc32=entry.CRC))
        print(f'[{i+1}/{len(selected)}] {entry.filename}', flush=True)
    (args.output/'download_manifest.json').write_text(json.dumps(dict(
        source=URL, doi='10.5281/zenodo.15196867', license='CC-BY-4.0',
        authors=['Anna Maschek', 'David C. Schedl'], archive_bytes=remote.size,
        selection='All members except original large videos and macOS metadata', files=manifest), indent=2)+'\n')


if __name__ == '__main__':
    main()
