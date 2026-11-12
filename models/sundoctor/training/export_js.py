"""Pack the numpy weights into one float32 blob + manifest for the in-browser network."""
import sys, json, base64, numpy as np
def pack(npz):
    z = np.load(npz); meta = json.loads(str(z['meta'])); parts, man, off = [], [], 0
    for k in sorted(k for k in z.files if k != 'meta'):
        a = np.ascontiguousarray(z[k], np.float32); parts.append(a.ravel()); man.append({'k': k, 'shape': list(a.shape), 'off': off}); off += a.size
    return {'tensors': man, 'n': off, 'meta': meta}, base64.b64encode(np.concatenate(parts).astype('<f4').tobytes()).decode()
if __name__ == '__main__':
    man, b64 = pack(sys.argv[1]); json.dump({'man': man, 'b64': b64}, open(sys.argv[2], 'w')); print('params', man['n'], 'MB %.2f' % (len(b64) / 1e6))
