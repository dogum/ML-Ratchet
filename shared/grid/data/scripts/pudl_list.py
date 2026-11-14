import urllib.request, re, urllib.parse, sys
base="https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/?list-type=2&prefix=stable/&max-keys=1000"
tok=None; out=[]
while True:
    u=base+("&continuation-token="+urllib.parse.quote(tok) if tok else "")
    x=urllib.request.urlopen(u,timeout=60).read().decode()
    for k,s in re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>",x): out.append((k,int(s)))
    m=re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>",x)
    if not m: break
    tok=m.group(1)
with open(sys.argv[1],'w') as f:
    for k,s in out: f.write(f"{s}\t{k}\n")
print(len(out))
