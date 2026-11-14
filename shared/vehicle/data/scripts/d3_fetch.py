"""Argonne D3 (Downloadable Dynamometer Database): list and download the public test data.

D3's vehicle pages (anl.gov/taps/...) embed one public Box folder per vehicle; Box is Argonne's distribution channel
for the data. anl.gov itself refuses scripted requests (HTTP 403), so the vehicle -> Box folder table below was read
once from the vehicle pages in a browser (13 Nov 2026). Box serves the folders to scripts.

Terms (anl.gov/taps/downloadable-dynamometer-database): data can only be used with the attribution "This data is
from the Downloadable Dynamometer Database and was generated at the Advanced Mobility Technology Laboratory (AMTL) at
Argonne National Laboratory under the funding and guidance of the U.S. Department of Energy (DOE)" or a standard
bibliographic reference. Files stay in data/raw/d3/ (git-ignored); only results derived from them are published.

usage:
    python d3_fetch.py list                       # walk every folder, write raw/d3/listing.json, print sizes
    python d3_fetch.py get [--cat hybrid,plugin] [--max-gb 60] [--vehicles "Sonata Hybrid,Accord Hybrid"]
"""
import os, re, sys, json, time, argparse, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

_VEH = os.environ.get('ML_RATCHET_VEHICLE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
RAW = os.path.join(_VEH, 'data', 'raw', 'd3')
UA = {'User-Agent': 'Mozilla/5.0 (research data download)'}

# (category, vehicle, D3 page slug, box host, shared folder id)
VEHICLES = [
    ('alternative', '2016 Toyota Mirai', 'd3-2016-toyota-mirai', 'app', 'igeyaop4lgw2lx62lkocvo1a4hry75di'),
    ('alternative', '2015 Chevrolet Bi-Fuel Impala', 'd3-2015-chevrolet-bifuel-impala', 'app', 'hpsov0jd02nv4u05m3w3aog55z1wd150'),
    ('alternative', '2010 Honda Civic GX', 'd3-2012-honda-civic-gx', 'app', 'mj352urkca0z6aamrlcsmn9steozc5oi'),
    ('alternative', '2010 Hyundai LPI Hybrid', 'd3-2010-hyundai-lpi-hybrid', 'app', 'ciieb8ro6nqhdx8tgt91i094m0lvt5sh'),
    ('conventional', '2020 Chrysler Pacifica', 'd3-2020-chrysler-pacifica', 'anl', 'iwkh4xc5cfzcdz0uil4k1rbv9qbdnvpd'),
    ('conventional', '2019 Chevrolet Blazer', 'd3-2019-chevrolet-blazer', 'anl', 'qc7324oek427s6o7olypubrb9vd2m5xc'),
    ('conventional', '2019 Chevrolet Silverado', 'd3-2019-chevrolet-silverado', 'anl', 'pebnmrtlczlqmsdz0ebuycxxx7xlecl5'),
    ('conventional', '2019 Infiniti QX50', 'd3-2019-infiniti-qx50', 'anl', '27zy83andh3bi8kasnb5r8ym5ae6ptch'),
    ('conventional', '2018 Toyota Camry XLE', 'd3-2018-toyota-camry-xle', 'anl', 'k39kpkwrxcbte0j6340a2ysiz6tvdp44'),
    ('conventional', '2018 Honda Accord LX', 'd3-2018-honda-accord-lx', 'anl', 'o72fkaps280txvq88wv6k4noo0x14qwx'),
    ('conventional', '2017 Ford F150', 'd3-2017-ford-f150', 'app', 'xqw41hnmyynjj08a3wwd9retn43kqcu3'),
    ('conventional', '2016 Mazda CX9', 'd3-2016-mazda-cx9', 'app', 'w7k4tfrc3hejtf02bflre0p8l5jflpsm'),
    ('conventional', '2014 Chevrolet Cruze Diesel', 'd3-2014-chevrolet-cruze-diesel', 'app', 'tdksakbjmuczg1txju5fydil0y1w3q8d'),
    ('conventional', '2014 Mazda 3 iEloop', 'd3-2014-mazda-3-ieloop', 'app', 'h47v6isntx40tt0kappmkie7dnxdztvq'),
    ('conventional', '2013 Dodge Ram 1500 HFE', 'd3-2013-dodge-ram-1500-hfe', 'app', 'px7oclo38t37fzzbn65vurreqd3wye3h'),
    ('conventional', '2013 Hyundai Sonata', 'd3-2013-hyundai-sonata', 'app', 'xrpr0mxtxch7hx66v02kd6z9x24e3wh0'),
    ('conventional', '2013 Nissan Altima', 'd3-2013-nissan-altima', 'app', 'le6u53n2kvkoy3o1bmahwpmml4nh1fjp'),
    ('conventional', '2013 Volkswagen Jetta TDI', 'd3-2013-volkswagen-jetta-tdi', 'app', 'kczc4v7y83o56zmgs7g7p0w8gpozaxpc'),
    ('conventional', '2012 Chrysler 300', 'd3-2012-chrysler-300', 'app', '1dbuvdn2xzqodpzfb8ckymmskv1homzl'),
    ('conventional', '2012 Fiat 500 Sport', 'd3-2012-fiat-500-sport', 'app', 'g3g8vp1xbcjz9uk3dd5aiiosr7bmf13x'),
    ('conventional', '2012 Ford F150 Ecoboost', 'd3-2012-ford-f150-ecoboost', 'app', 'w21b77h4fux6uniqh4mpcjuuikr6ycjo'),
    ('conventional', '2012 Ford Focus', 'd3-2012-ford-focus', 'app', '6tzwtbzznfr8fz8chx4us7c10ob6909x'),
    ('conventional', '2012 Ford Fusion V6', 'd3-2012-ford-fusion-v6', 'app', 'p9ml81nftkrtqk3bdej45z9y13rzgwoc'),
    ('conventional', '2010 Mazda 3 i-stop', 'd3-2010-mazda-3-istop', 'app', 'wgu1butx8dtvzcek285euzuo38ecdbmk'),
    ('conventional', '2010 SmartCar MHD', 'd3-2010-smartcar-mhd', 'app', '7gweq1n7smelti7jvk2rv4mm77bmhfcw'),
    ('conventional', '2010 VW Golf TDI Bluemotion', 'd3-2010-vw-golf-tdi-bluemotion', 'app', 'yg7i3ar1bsh5o98vh5vdp941jstyyseh'),
    ('conventional', '2009 Volkswagen Jetta TDI', 'd3-2009-volkswagen-jetta-tdi', 'app', 'uqbq20spg0v1n7ss50rfl9gwuhr4z4ow'),
    ('electric', '2023 Hyundai Ioniq5', 'd3-2023-hyundai-ioniq5', 'anl', 'b8xuadhau857u3a2wdpy5x2qsuqqmsn3'),
    ('electric', '2020 Tesla Model 3', 'd3-2020-tesla-model-3', 'anl', '6ybbqmq5orgq6m47nt8zjvfwmfohph6i'),
    ('electric', '2020 Chevrolet Bolt', 'd3-2020-chevrolet-bolt', 'anl', '3algypjxuq224pbq7jzo6b95199oxzu7'),
    ('electric', '2019 Nissan Leaf ePlus', 'd3-2019-nissan-leaf-eplus', 'anl', '2esdlz7wiqpu5odqk90jn2myqz64u6gf'),
    ('electric', '2019 Chevrolet Bolt', 'd3-2019-chevrolet-bolt', 'anl', 'cc41akjh9k50gjfx4l9ibhj20ba9glrj'),
    ('electric', '2017 Chevrolet Bolt', 'd3-2017-chevrolet-bolt', 'anl', 'rpx3ss7ceabzrhgd9k644vka67rx9ce2'),
    ('electric', '2015 Chevrolet Spark EV', 'd3-2015-chevrolet-spark-ev', 'app', '0lzzhnr4yrdfbiwnoxeqea208bkrdzyv'),
    ('electric', '2015 Kia Soul Electric', 'd3-2015-kia-soul-electric', 'app', 'xz4kutflu5oxrvjb2j6otrv4he33rwlz'),
    ('electric', '2015 Mercedes-Benz B-Class Electric Drive', 'd3-2015-mercedesbenz-bclass-electric-drive', 'app', 'v80x4rdck3oplt30cjc2pcckxjpb3vs9'),
    ('electric', '2015 Volkswagen e-Golf', 'd3-2015-volkswagen-egolf', 'app', 'dh840rhp4u3b1btopg4p8iiqqox1m2ee'),
    ('electric', '2014 BMW i3BEV', 'd3-2014-bmw-i3bev', 'app', 'p26vff1rf2yu2bft5s8v2vh5lxy2d5hb'),
    ('electric', '2014 Smart Electric Drive', 'd3-2014-smart-electric-drive', 'app', '77keah06jwpcdq5rg6c843mw5ofajbws'),
    ('electric', '2013 Ford Focus Electric', 'd3-2013-ford-focus-electric', 'app', 'tt76liy88h2kg6vcrupv662vvyl6pst6'),
    ('electric', '2013 Nissan Leaf SV', 'd3-2013-nissan-leaf-sv', 'app', 'qoi43in143szma036o3uab63pnvgprn6'),
    ('electric', '2012 Mitsubishi I-MiEV', 'd3-2012-mitsubishi-imiev', 'app', '7ss458xp3sawl1xpr9q0ianmfjzo2p24'),
    ('electric', '2012 Nissan Leaf', 'd3-2012-nissan-leaf', 'app', 'zfq1vydzrk7i1h4oicc7tb62pc8082rm'),
    ('hybrid', '2021 Hyundai Sonata Hybrid', 'd3-2021-hyundai-sonata-hybrid', 'anl', 'bbo0chvvlpxijd0qvyphzutwkeoi5ocv'),
    ('hybrid', '2019 Acura MDX Sport Hybrid', 'd3-2019-acura-mdx-sport-hybrid', 'anl', 'nlve32pjd80hmggn0ln1j9pglmxndbm3'),
    ('hybrid', '2015 Honda Accord Hybrid', 'd3-2015-honda-accord-hybrid', 'app', '6ieutyro6mmps59t3eelcdbjdenj39rn'),
    ('hybrid', '2013 Chevrolet Malibu Eco', 'd3-2013-chevrolet-malibu-eco', 'app', 'jwh38u9wonkso55u019np6gqqxqmjrpb'),
    ('hybrid', '2013 Ford Cmax Hybrid', 'd3-2013-ford-cmax-hybrid', 'app', '923z5xo9o3dhlp6y9npgn8k9ex78j6n6'),
    ('hybrid', '2013 Honda Civic Hybrid', 'd3-2013-honda-civic-hybrid', 'app', 'obi9wua454apg02iewrgr2t8pb8pzx6t'),
    ('hybrid', '2013 Volkswagen Jetta Hybrid', 'd3-2013-volkswagen-jetta-hybrid', 'app', '2jzpggiib0mfvgxf5z02oo2xaag9fui2'),
    ('hybrid', '2011 Hyundai Sonata Hybrid', 'd3-2011-hyundai-sonata', 'app', 'cjs8r6ffo31s3dy5uryewop9zk863tib'),
    ('hybrid', '2010 Ford Fusion Hybrid', 'd3-2010-ford-fusion-hybrid', 'app', 'lhbuweeav3nlzmxj4b90ab8ee8eoz33m'),
    ('hybrid', '2010 Honda CR-Z', 'd3-2010-honda-crz', 'app', 'wvpc65g0de1q5z09nxreu92zbs6z3cmn'),
    ('hybrid', '2010 Honda Insight', 'd3-2010-honda-insight', 'app', 'erdcrylmuys968cgqc2282k2wjkydm1t'),
    ('hybrid', '2010 Toyota Prius', 'd3-2010-toyota-prius', 'app', '56wv6abwb9tcww8ko881up2dyqtj02z9'),
    ('hybrid', '2010 Mercedes S400h BlueHybrid', 'd3-2010-mercedes-s400h-bluehybrid', 'app', 'q6knl35a3u9l7eb6bk31gho7v6vchec9'),
    ('plugin', '2021 Toyota RAV4 Prime', 'd3-2021-toyota-rav-4-prime', 'anl', 'o2dmm6p8llgrodug2k7pwy5nor9q1wfm'),
    ('plugin', '2017 Toyota Prius Prime', 'd3-2017-toyota-prius-prime', 'anl', 'bisti1zkntiglmnjotikzon1h3atfv5m'),
    ('plugin', '2016 Chevrolet Volt', 'd3-2016-chevrolet-volt', 'app', 's5p8d7sbo0haouiihivt7f3s6o5nuzox'),
    ('plugin', '2014 BMW i3-REX', 'd3-2014-bmw-i3rex', 'app', 'gjqa5np90a4fhvk7cy1kfyvksvirlmmj'),
    ('plugin', '2013 Ford Cmax Energi', 'd3-2013-ford-cmax-energi', 'app', '87a4wdm15vis5gdm6fs4fy7283e4nwks'),
    ('plugin', '2013 Ford Fusion Energi', 'd3-ford-fusion-energi', 'app', '2ircoqc0qy1lol7i82yp1o6w7i7pg3gs'),
    ('plugin', '2013 Chevrolet Volt', 'd3-2013-chevrolet-volt', 'app', 'f1xvnc165i59opmbav1nwg4ola1opnav'),
    ('plugin', '2012 Chevrolet Volt', 'd3-2012-chevrolet-volt', 'app', 'qp3rybooe8gfb6c8g9jrxx4s48hbix2e'),
    ('plugin', '2013 Toyota Prius Plug-in', 'd3-2013-toyota-prius', 'app', 'p9i55s01f3g69xb1qfpgsjz673enfhmy'),
    ('plugin', '2012 Toyota Prius Plug-in', 'd3-2012-toyota-prius', 'app', '6gvrktg25swo0cx0l99s4q88ibcr82qt'),
]


def _host(h):
    return 'https://anl.app.box.com' if h == 'anl' else 'https://app.box.com'


def _get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def _folder(host, share, folder_id=None, page=1):
    url = f'{_host(host)}/s/{share}' + (f'/folder/{folder_id}' if folder_id else '') + f'?page={page}'
    h = _get(url).decode('utf-8', 'replace')
    m = re.search(r'Box\.postStreamData\s*=\s*(\{.*?\});\s*</script>', h, re.S)
    if not m:
        raise RuntimeError(f'no folder data at {url}')
    return json.loads(m.group(1))['/app-api/enduserapp/shared-folder']


def walk(v):
    cat, name, slug, host, share = v
    files, stack = [], [(None, '')]
    while stack:
        fid, path = stack.pop()
        page, pages = 1, 1
        while page <= pages:
            d = _folder(host, share, fid, page)
            pages = d.get('pageCount', 1)
            for it in d['items']:
                p = f'{path}/{it["name"]}'.strip('/')
                if it['type'] == 'folder':
                    stack.append((it['id'], p))
                else:
                    files.append({'path': p, 'id': it['id'], 'size': it.get('itemSize', 0)})
            page += 1
    return {'cat': cat, 'name': name, 'page': f'https://www.anl.gov/taps/{slug}', 'host': host, 'share': share,
            'files': files, 'bytes': sum(f['size'] for f in files)}


def cmd_list():
    os.makedirs(RAW, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(walk, VEHICLES))
    json.dump(out, open(os.path.join(RAW, 'listing.json'), 'w', encoding='utf-8'), indent=1)
    tot = 0
    for v in out:
        tot += v['bytes']
        print(f'{v["cat"]:<12} {v["name"]:<42} {len(v["files"]):>4} files {v["bytes"] / 1e9:7.2f} GB')
    print(f'total {sum(len(v["files"]) for v in out)} files, {tot / 1e9:.1f} GB')


def _download(job):
    v, f = job
    dest = os.path.join(RAW, re.sub(r'[^\w.-]+', '_', v['name']), *[re.sub(r'[<>:"|?*]', '_', s) for s in f['path'].split('/')])
    if os.path.exists(dest) and os.path.getsize(dest) == f['size']:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f'{_host(v["host"])}/index.php?rm=box_download_shared_file&shared_name={v["share"]}&file_id=f_{f["id"]}'
    tmp = dest + '.part'
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, 'wb') as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            if os.path.getsize(tmp) != f['size'] and f['size']:
                raise IOError(f'size {os.path.getsize(tmp)} != {f["size"]}')
            os.replace(tmp, dest)
            return f['size']
        except Exception as e:
            if k == 4:
                print('FAILED', v['name'], f['path'], e, flush=True)
                return 0
            time.sleep(3 * (k + 1))


def cmd_get(cats, max_gb, names):
    listing = json.load(open(os.path.join(RAW, 'listing.json'), encoding='utf-8'))
    sel = [v for v in listing if (not cats or v['cat'] in cats) and (not names or any(n.lower() in v['name'].lower() for n in names))]
    jobs, budget = [], max_gb * 1e9
    for v in sel:                                   # whole vehicles, in listing order, until the budget is spent
        if v['bytes'] > budget:
            print('skip (budget)', v['name'], f'{v["bytes"] / 1e9:.1f} GB'); continue
        budget -= v['bytes']
        jobs += [(v, f) for f in v['files']]
    print(f'{len(jobs)} files from {len({id(j[0]) for j in jobs})} vehicles, {sum(j[1]["size"] for j in jobs) / 1e9:.1f} GB', flush=True)
    done = 0
    with ThreadPoolExecutor(6) as ex:
        for n in ex.map(_download, jobs):
            done += n
    print(f'downloaded {done / 1e9:.2f} GB new', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'get'])
    ap.add_argument('--cat', default='')
    ap.add_argument('--vehicles', default='')
    ap.add_argument('--max-gb', type=float, default=1e9)
    a = ap.parse_args()
    if a.cmd == 'list':
        cmd_list()
    else:
        cmd_get([c for c in a.cat.split(',') if c], a.max_gb, [n.strip() for n in a.vehicles.split(',') if n.strip()])
