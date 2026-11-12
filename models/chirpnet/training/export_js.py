"""Pack fused weights into one float32 blob + manifest for the in-browser network."""
import sys, json, base64, numpy as np
def pack(npz_path):
    z = np.load(npz_path); meta = json.loads(str(z['meta']))
    names = [f'w{i}' for i in range(len(meta['conv']))]
    parts, man, off = [], [], 0
    def add(key, arr):
        nonlocal off
        a = np.ascontiguousarray(arr, np.float32).ravel(); parts.append(a)
        man.append({'k': key, 'shape': list(arr.shape), 'off': off}); off += a.size
    for i, (s, p) in enumerate(meta['conv']):
        add(f'w{i}', z[f'w{i}']); add(f'b{i}', z[f'b{i}'])
    if 'rw' in z.files: add('rw', z['rw']); add('rb', z['rb'])
    for k in ('mixw', 'mixb', 'p1w', 'p1b', 'p2w', 'p2b', 'psw', 'psb'):
        if k in z.files: add(k, z[k])
    for k in ('lw0', 'lb0', 'lw1', 'lb1'): add(k, z[k])
    blob = np.concatenate(parts).astype('<f4').tobytes()
    return {'conv': meta['conv'], 'tensors': man, 'n': off, 'arch': meta.get('arch', 'cnn'), 'veto': meta.get('veto')}, base64.b64encode(blob).decode()
if __name__ == '__main__':
    man, b64 = pack(sys.argv[1])
    json.dump({'man': man, 'b64': b64}, open(sys.argv[2], 'w'))
    print('params', man['n'], 'b64 MB %.2f' % (len(b64) / 1e6))
