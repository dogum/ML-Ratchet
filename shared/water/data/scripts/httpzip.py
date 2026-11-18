"""Read members of a remote zip with HTTP Range requests (no full download). Data only; nothing is executed."""
import io, requests


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.s = url, 0, requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        self.url = r.url
        self.size = int(r.headers['Content-Length'])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for i in range(5):
            try:
                r = self.s.get(self.url, headers={'Range': f'bytes={self.pos}-{self.pos + n - 1}'}, timeout=(30, 300))
                r.raise_for_status(); data = r.content; break
            except Exception:
                if i == 4: raise
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)
