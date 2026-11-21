#!/usr/bin/env python3
"""
CLIP KATARNAK  -  YouTube video / live stream  ->  9:16 split-screen clips
  upar  : gameplay / screen
  neeche: streamer ka face (auto face detect)

Run:   python app.py        ->  http://127.0.0.1:5000
Needs: pip install flask yt-dlp opencv-python-headless   +   ffmpeg
"""
import os, re, sys, json, time, uuid, shutil, zipfile, threading, subprocess, statistics, traceback, urllib.request, urllib.parse
from collections import defaultdict
from pathlib import Path
from flask import Flask, request, jsonify, send_file, abort, Response

BASE = Path(__file__).parent.resolve()
WORK = BASE / "work"; WORK.mkdir(exist_ok=True)
OUT_W, HALF = 1080, 960
ASPECT = OUT_W / HALF  # 1.125  (har half 1080x960)
JOBS = {}
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MAX_CLIPS, MIN_DUR, MAX_DUR = 10, 5, 180

# ----------------------------------------------------------------- helpers
def parse_time(s):
    """'01:27:02' | '1:27:02' | '87:02' | '5237' | '1h27m02s'  ->  seconds"""
    s = (s or "").strip().lower().lstrip("-–— ").strip()
    if not s: return None
    if re.fullmatch(r"\d+(:\d+){1,2}(\.\d+)?", s):
        sec = 0.0
        for p in s.split(":"): sec = sec * 60 + float(p)
        return sec
    if re.fullmatch(r"\d+(\.\d+)?s?", s): return float(s.rstrip("s"))
    m = re.fullmatch(r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?", s)
    if m and any(m.groups()):
        h, mi, se = [int(x or 0) for x in m.groups()]; return h * 3600 + mi * 60 + se
    raise ValueError("Time format galat hai. Aise likh: 01:27:02")

def fmt_time(sec):
    sec = int(round(sec)); return "%d:%02d:%02d" % (sec // 3600, sec % 3600 // 60, sec % 60)

def time_from_url(url):
    try:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        for k in ("t", "start"):
            if k in q: return parse_time(q[k][0])
    except Exception: pass
    return None

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", str(path)], capture_output=True, text=True).stdout
    j = json.loads(out)
    return int(j["streams"][0]["width"]), int(j["streams"][0]["height"]), float(j["format"]["duration"])

def clamp(v, a, b): return max(a, min(b, v))

# ----------------------------------------------------------------- face / facecam
def _make_detector(log):
    import cv2
    try:
        p = BASE / "models" / "yunet.onnx"
        if not p.exists():
            p.parent.mkdir(exist_ok=True); urllib.request.urlretrieve(YUNET_URL, p)
        det = cv2.FaceDetectorYN.create(str(p), "", (320, 320), 0.7, 0.3, 5000)
        def run(fr):
            h0, w0 = fr.shape[:2]; sc = min(1.0, 960.0 / w0)
            sm = cv2.resize(fr, (int(w0 * sc), int(h0 * sc))) if sc < 1 else fr
            det.setInputSize((sm.shape[1], sm.shape[0])); _, faces = det.detect(sm)
            return [] if faces is None else [(f[0] / sc, f[1] / sc, f[2] / sc, f[3] / sc) for f in faces]
        return run, "YuNet"
    except Exception as e:
        log("YuNet nahi chala (%s) - Haar try kar raha hu" % str(e)[:60])
    try:
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if casc.empty(): raise RuntimeError("cascade missing")
        def run(fr):
            h0, w0 = fr.shape[:2]; sc = min(1.0, 960.0 / w0)
            g = cv2.cvtColor(cv2.resize(fr, (int(w0 * sc), int(h0 * sc))) if sc < 1 else fr, cv2.COLOR_BGR2GRAY)
            g = cv2.equalizeHist(g)
            return [(x / sc, y / sc, w / sc, h / sc) for x, y, w, h in casc.detectMultiScale(g, 1.1, 5, minSize=(28, 28))]
        return run, "Haar"
    except Exception as e:
        log("Face detector available nahi: %s" % str(e)[:60]); return None, None

def fit_rect(cx, cy, cw, ch, W, H):
    s = min(1.0, W / cw, H / ch); cw, ch = cw * s, ch * s
    x = clamp(cx - cw / 2, 0, W - cw); y = clamp(cy - ch / 2, 0, H - ch)
    return int(x) // 2 * 2, int(y) // 2 * 2, int(cw) // 2 * 2, int(ch) // 2 * 2

def face_crop(face, W, H):
    x, y, w, h = face; cw = w * 2.6; ch = cw / ASPECT
    return fit_rect(x + w / 2, y + h / 2 + h * .08, cw, ch, W, H)

PRESETS = {  # facecam corner: (x, y, w, h) fraction of frame
    "br": (.68, .58, .32, .42), "bl": (0, .58, .32, .42), "tr": (.68, 0, .32, .42), "tl": (0, 0, .32, .42), "center": (.25, .1, .5, .8)}

def preset_crop(name, W, H):
    fx, fy, fw, fh = PRESETS.get(name, PRESETS["br"]); w, h = fw * W, fh * H
    if w / h < ASPECT: w = h * ASPECT
    else: h = w / ASPECT
    return fit_rect(fx * W + fw * W / 2, fy * H + fh * H / 2, w, h, W, H)

# ----------------------------------------------------------------- ffmpeg
def build_graph(mode, crop, divider):
    cx, cy, cw, ch = crop
    if mode == "zoom":
        top = ("[0:v]split=2[a][c];[a]crop='min(iw,ih*%s)':'min(ih,iw/%s)',scale=%d:%d:flags=lanczos[top];" % (ASPECT, ASPECT, OUT_W, HALF))
    else:
        top = ("[0:v]split=3[a][b][c];"
               "[a]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,gblur=sigma=40,eq=brightness=-0.18:saturation=1.25[bg];"
               "[b]scale=%d:%d:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[top];" % (OUT_W, HALF, OUT_W, HALF, OUT_W, HALF))
    g = top + "[c]crop=%d:%d:%d:%d,scale=%d:%d:flags=lanczos[bot];[top][bot]vstack=inputs=2[s];" % (cw, ch, cx, cy, OUT_W, HALF)
    g += "[s]" + ("drawbox=x=0:y=%d:w=%d:h=8:color=0xFF2E93:t=fill," % (HALF - 4, OUT_W) if divider else "") + "fps=30,setsar=1,format=yuv420p[v]"
    return g

def make_clip(src, t0, d, out, crop, mode, divider):
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % t0, "-t", "%.3f" % d, "-i", str(src), "-filter_complex", build_graph(mode, crop, divider),
           "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
           "-movflags", "+faststart", "-t", "%.3f" % d, str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not Path(out).exists(): raise RuntimeError("ffmpeg error: " + r.stderr[-400:])

# ----------------------------------------------------------------- job
class Job:
    def __init__(self): self.id = uuid.uuid4().hex[:10]; self.status = "running"; self.progress = 0; self.logs = []; self.clips = []; self.error = None; self.title = ""
    def log(self, m): self.logs.append(m); print("[%s] %s" % (self.id, m), flush=True)
    def public(self):
        return dict(id=self.id, status=self.status, progress=self.progress, logs=self.logs[-40:], clips=self.clips, error=self.error, title=self.title,
                    zip=("api/zip/%s" % self.id) if self.status == "done" and self.clips else None)

def friendly(e):
    s = str(e)
    if "Sign in to confirm" in s or "not a bot" in s:
        return "YouTube ne bot samajh ke roka. 'Advanced' me cookies.txt upload kar (YouTube login wale browser se export) aur dobara try kar."
    if "Private video" in s or "members-only" in s.lower(): return "Ye video private / members-only hai, download nahi ho sakti."
    if "Requested format is not available" in s: return "Is video ka format nahi mila. yt-dlp update kar: pip install -U yt-dlp"
    if "Video unavailable" in s: return "Video available nahi hai (link check kar)."
    return s[:300]

def run_job(job, p):
    wd = WORK / job.id; wd.mkdir(exist_ok=True)
    try:
        url, start, d, n = p["url"], p["start"], p["dur"], p["count"]; mode, face_pos, divider = p["mode"], p["face"], p["divider"]
        end = start + d * n; off = 0.0
        if os.path.isfile(url):                                     # local file bhi chalegi
            src = Path(url); off = start; job.title = src.name; job.log("Local file use kar raha hu"); W, H, full = probe(src)
            actual = full - start
        else:
            import yt_dlp
            from yt_dlp.utils import download_range_func
            base = {"quiet": True, "no_warnings": True, "noplaylist": True, "retries": 5}
            if p.get("cookies"): base["cookiefile"] = p["cookies"]
            job.log("Video ki info la raha hu..."); job.progress = 3
            with yt_dlp.YoutubeDL(dict(base, skip_download=True, live_from_start=p["live"])) as y: info = y.extract_info(url, download=False)
            job.title = info.get("title", ""); length = info.get("duration"); is_live = bool(info.get("is_live"))
            job.log("Title: %s" % job.title[:80])
            if length and not is_live:
                if start >= length: raise ValueError("Start time (%s) video ki length (%s) se zyada hai." % (fmt_time(start), fmt_time(length)))
                if end > length:
                    n = int((length - start) / d + 0.05)
                    if n < 1: raise ValueError("Start ke baad itna video bacha hi nahi (%s)." % fmt_time(length - start))
                    job.log("Video %s pe khatam ho jati hai, isliye %d clip ban payengi" % (fmt_time(length), n)); end = start + d * n
            job.log("Download: %s -> %s" % (fmt_time(start), fmt_time(end)))

            def hook(h):
                if h.get("status") == "downloading":
                    tot = h.get("total_bytes") or h.get("total_bytes_estimate")
                    if tot: job.progress = max(job.progress, int(6 + 44 * h["downloaded_bytes"] / tot))
            opts = dict(base, format="bv*[height<=1080]+ba/b[height<=1080]/b", outtmpl=str(wd / "src.%(ext)s"), merge_output_format="mp4",
                        download_ranges=download_range_func(None, [(start, end)]), force_keyframes_at_cuts=True,
                        progress_hooks=[hook], concurrent_fragment_downloads=4, live_from_start=p["live"])
            with yt_dlp.YoutubeDL(opts) as y: y.download([url])
            cands = [f for f in wd.glob("src.*") if f.suffix not in (".part", ".ytdl")]
            if not cands: raise RuntimeError("Download fail ho gaya (file nahi bani).")
            src = sorted(cands, key=lambda f: f.suffix != ".mp4")[0]; W, H, actual = probe(src)
        job.progress = 52; job.log("Source %dx%d, %s" % (W, H, fmt_time(actual)))
        n = min(n, int((actual + 1.5) // d))
        if n < 1: raise ValueError("Itne time ka video mila hi nahi.")

        # --- facecam crop
        crop = None
        if face_pos == "auto":
            job.log("Face dhoond raha hu...")
            try: face = find_face_window(src, W, H, off, actual, job.log)
            except Exception as e: face = None; job.log("Face detect error: %s" % str(e)[:80])
            if face: crop = face_crop(face, W, H)
            else: job.log("Face nahi mila - bottom-right corner facecam maan raha hu (Facecam position se badal sakta hai)"); crop = preset_crop("br", W, H)
        else: crop = preset_crop(face_pos, W, H)
        job.progress = 58

        # --- clips
        for i in range(n):
            a = i * d; name = "clip_%02d.mp4" % (i + 1); out = wd / name
            job.log("Clip %d/%d bana raha hu (%s)" % (i + 1, n, fmt_time(start + a)))
            make_clip(src, off + a, d, out, crop, mode, divider)
            th = wd / ("clip_%02d.jpg" % (i + 1))
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(out), "-frames:v", "1", "-vf", "scale=360:-2", str(th)], capture_output=True)
            job.clips.append(dict(name=name, label="Clip %d  |  %s - %s  |  %ds" % (i + 1, fmt_time(start + a), fmt_time(start + a + d), d),
                                  url="api/file/%s/%s" % (job.id, name), dl="api/file/%s/%s?dl=1" % (job.id, name),
                                  thumb="api/file/%s/%s" % (job.id, th.name) if th.exists() else "", size=round(out.stat().st_size / 1e6, 1)))
            job.progress = 58 + int(40 * (i + 1) / n)
        with zipfile.ZipFile(wd / "clips.zip", "w", zipfile.ZIP_STORED) as z:
            for c in job.clips: z.write(wd / c["name"], c["name"])
        if off == 0:
            for f in wd.glob("src.*"): f.unlink(missing_ok=True)
        job.progress = 100; job.status = "done"; job.log("Ho gaya! %d clip ready" % n)
    except Exception as e:
        traceback.print_exc(); job.status = "error"; job.error = friendly(e); job.log("ERROR: " + job.error)

def find_face_window(src, W, H, off, length, log):
    import cv2
    run, name = _make_detector(log)
    if run is None: return None
    cap = cv2.VideoCapture(str(src)); hits = []; n = 14
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (off + min(length, 600) * (k + .5) / n) * 1000); ok, fr = cap.read()
        if ok: hits += [f for f in run(fr) if f[2] >= W * 0.025]
    cap.release()
    if len(hits) < 2: return None
    b = defaultdict(list)
    for f in hits: b[(int((f[0] + f[2] / 2) // (W * .1)), int((f[1] + f[3] / 2) // (H * .1)))].append(f)
    best = max(b.values(), key=lambda v: (len(v), sum(f[2] for f in v)))
    if len(best) < 2: return None
    log("Face mila (%s): %d/%d frames me" % (name, len(best), n))
    return tuple(statistics.median(v) for v in zip(*best))

def cleanup_old():
    for d in WORK.iterdir():
        try:
            if d.is_dir() and time.time() - d.stat().st_mtime > 3 * 3600: shutil.rmtree(d, ignore_errors=True)
        except Exception: pass

# ----------------------------------------------------------------- web
app = Flask(__name__)

@app.get("/")
def index(): return Response(PAGE, mimetype="text/html")

@app.post("/api/start")
def api_start():
    f = request.form; url = f.get("url", "").strip()
    try:
        if not url: raise ValueError("YouTube link daal.")
        start = parse_time(f.get("start", ""))
        if start is None: start = time_from_url(url) or 0.0
        d = int(float(f.get("dur", 30))); n = int(float(f.get("count", 5)))
        if not (MIN_DUR <= d <= MAX_DUR): raise ValueError("Clip %d se %d second ke beech honi chahiye." % (MIN_DUR, MAX_DUR))
        if not (1 <= n <= MAX_CLIPS): raise ValueError("Clips 1 se %d ke beech rakh." % MAX_CLIPS)
    except ValueError as e: return jsonify(error=str(e)), 400
    cleanup_old(); job = Job(); JOBS[job.id] = job
    ck = None
    if "cookies" in request.files and request.files["cookies"].filename:
        (WORK / job.id).mkdir(exist_ok=True); ck = str(WORK / job.id / "cookies.txt"); request.files["cookies"].save(ck)
    p = dict(url=url, start=start, dur=d, count=n, mode=f.get("mode", "fit") if f.get("mode") in ("fit", "zoom") else "fit",
             face=f.get("face", "auto"), live=f.get("live") == "1", divider=f.get("divider", "1") == "1", cookies=ck)
    threading.Thread(target=run_job, args=(job, p), daemon=True).start()
    return jsonify(id=job.id)

@app.get("/api/status/<jid>")
def api_status(jid):
    j = JOBS.get(jid); return jsonify(j.public()) if j else abort(404)

@app.get("/api/file/<jid>/<name>")
def api_file(jid, name):
    if not re.fullmatch(r"[0-9a-f]{10}", jid) or not re.fullmatch(r"clip_\d\d\.(mp4|jpg)", name): abort(404)
    p = WORK / jid / name
    if not p.exists(): abort(404)
    return send_file(p, as_attachment=request.args.get("dl") == "1", download_name=name, conditional=True)

@app.get("/api/zip/<jid>")
def api_zip(jid):
    if not re.fullmatch(r"[0-9a-f]{10}", jid): abort(404)
    p = WORK / jid / "clips.zip"
    if not p.exists(): abort(404)
    return send_file(p, as_attachment=True, download_name="clips_%s.zip" % jid)

PAGE = r"""<!doctype html><html lang="hi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>CLIP KATARNAK</title><style>
:root{--bg:#0b0319;--card:#17102e;--card2:#1f1640;--line:#3a2a6b;--pink:#ff2e93;--cyan:#19e3ff;--yel:#ffe14d;--grn:#2dff8f;--txt:#fff;--mut:#a99cd6}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;font-family:Poppins,Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--txt);min-height:100vh;
background:radial-gradient(900px 500px at 10% -10%,#5b1a8c55,transparent),radial-gradient(800px 500px at 100% 10%,#ff2e9344,transparent),linear-gradient(#0b0319,#12062a);
padding:max(18px,env(safe-area-inset-top)) 16px 60px}
.wrap{max-width:720px;margin:0 auto}
h1{font-size:34px;line-height:1.05;margin:6px 0 4px;letter-spacing:-.5px;background:linear-gradient(90deg,var(--pink),var(--yel),var(--cyan));-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--mut);margin:0 0 18px;font-size:14px}
.card{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:22px;padding:18px;margin-bottom:16px;box-shadow:0 18px 40px #0008,0 0 0 1px #ffffff08 inset}
label{display:block;font-weight:600;font-size:13px;color:var(--cyan);margin:14px 0 7px;letter-spacing:.4px;text-transform:uppercase}label:first-child{margin-top:0}
input[type=text],select,input[type=number]{width:100%;background:#0e0822;border:1.5px solid var(--line);color:#fff;border-radius:14px;padding:14px;font-size:16px;font-family:inherit;outline:none}
input:focus,select:focus{border-color:var(--pink);box-shadow:0 0 0 3px #ff2e9333}
.hint{font-size:12px;color:var(--mut);margin-top:6px}
.chips{display:flex;gap:8px;flex-wrap:wrap}.chip{flex:1 1 auto;min-width:92px;text-align:center;padding:13px 10px;border-radius:14px;background:#0e0822;border:1.5px solid var(--line);cursor:pointer;font-weight:700;font-size:15px;user-select:none;transition:.15s}
.chip.on{background:linear-gradient(135deg,var(--pink),#ff7a3d);border-color:#fff;box-shadow:0 8px 22px #ff2e9355;transform:translateY(-1px)}
.row{display:flex;gap:10px}.row>*{flex:1}
.seg{display:flex;background:#0e0822;border:1.5px solid var(--line);border-radius:14px;padding:4px;gap:4px}.seg div{flex:1;text-align:center;padding:11px 6px;border-radius:10px;font-size:13px;font-weight:600;cursor:pointer;color:var(--mut)}
.seg div.on{background:linear-gradient(135deg,#7a3dff,var(--cyan));color:#fff}
.sum{margin-top:14px;padding:12px 14px;border-radius:14px;background:#0e0822;border:1px dashed var(--cyan);font-size:14px;color:var(--cyan)}
.go{width:100%;border:0;border-radius:18px;padding:18px;font-size:19px;font-weight:800;color:#fff;font-family:inherit;cursor:pointer;margin-top:16px;letter-spacing:.5px;
background:linear-gradient(135deg,var(--pink),#ff7a3d);box-shadow:0 14px 34px #ff2e9366,0 0 0 1px #fff3 inset;transition:.15s}.go:active{transform:scale(.98)}.go[disabled]{opacity:.5;filter:grayscale(.6)}
.layout{display:flex;gap:16px;align-items:center}.phone{width:86px;height:152px;border-radius:14px;border:3px solid #6d55c9;overflow:hidden;flex:none;display:flex;flex-direction:column;box-shadow:0 8px 22px #0008}
.phone div{flex:1;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800;letter-spacing:.6px}.phone .a{background:linear-gradient(#5b1a8c,#ff2e93)}.phone .b{background:linear-gradient(#19e3ff,#4a2bb5);border-top:3px solid var(--pink)}
.layout p{margin:0;font-size:13px;color:var(--mut);line-height:1.5}.layout b{color:#fff}
details{margin-top:14px}summary{cursor:pointer;color:var(--mut);font-size:13px;font-weight:600}
.bar{height:16px;border-radius:99px;background:#0e0822;border:1px solid var(--line);overflow:hidden}.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--pink),var(--yel),var(--grn));transition:width .5s;box-shadow:0 0 14px var(--pink)}
.log{margin-top:12px;font:12px/1.55 ui-monospace,Menlo,Consolas,monospace;color:#cbbff3;max-height:150px;overflow:auto;background:#0b0620;border-radius:12px;padding:10px;white-space:pre-wrap}
.err{color:#ff7a8a;font-weight:700;margin-top:10px}
.clips{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px;margin-top:14px}
.clip{background:#0e0822;border:1px solid var(--line);border-radius:18px;padding:10px;box-shadow:0 12px 26px #0007}
.clip video{width:100%;aspect-ratio:9/16;border-radius:12px;background:#000;display:block}.clip small{display:block;color:var(--mut);margin:8px 2px;font-size:12px;line-height:1.4}
a.dl{display:block;text-align:center;text-decoration:none;color:#06121a;font-weight:800;padding:12px;border-radius:12px;background:linear-gradient(135deg,var(--grn),var(--cyan))}
a.zip{display:block;text-align:center;text-decoration:none;color:#fff;font-weight:800;padding:15px;border-radius:14px;background:linear-gradient(135deg,#7a3dff,var(--pink));margin-top:14px}
.hide{display:none}
</style></head><body><div class="wrap">
<h1>CLIP<br>KATARNAK &#9889;</h1><p class="sub">YouTube video ya live stream se 9:16 split-screen clips - gameplay upar, face neeche.</p>

<div class="card">
 <label>YouTube link</label><input type="text" id="url" placeholder="https://youtube.com/watch?v=..." autocapitalize="off" autocorrect="off" inputmode="url">
 <label>Clip kahan se shuru kare? (start time)</label><input type="text" id="start" placeholder="01:27:02" inputmode="numeric">
 <div class="hint">Format: <b>HH:MM:SS</b> (jaise 01:27:02), ya MM:SS, ya seconds. Khali chhodega to link ka &amp;t= use hoga, warna 0:00.</div>

 <label>Kitni clip chahiye?</label>
 <div class="chips" id="presets"><div class="chip on" data-d="30" data-n="5">30s &times; 5</div><div class="chip" data-d="60" data-n="5">60s &times; 5</div><div class="chip" data-d="15" data-n="2">15s &times; 2</div><div class="chip" data-d="0" data-n="0">Custom</div></div>
 <div class="row hide" id="custom" style="margin-top:10px"><div><label style="margin-top:0">Clip length (sec)</label><input type="number" id="cd" value="45" min="5" max="180"></div><div><label style="margin-top:0">Kitni clips</label><input type="number" id="cn" value="3" min="1" max="10"></div></div>
 <div class="sum" id="sum"></div>

 <label>Gameplay (upar wala half)</label>
 <div class="seg" id="mode"><div class="on" data-v="fit">Poori screen + blur</div><div data-v="zoom">Zoom (fill)</div></div>
 <label>Facecam position (neeche wala half)</label>
 <select id="face"><option value="auto">Auto - face khud dhoondo</option><option value="br">Bottom-right corner</option><option value="bl">Bottom-left corner</option><option value="tr">Top-right corner</option><option value="tl">Top-left corner</option><option value="center">Center</option></select>
 <details><summary>Advanced</summary>
  <label><input type="checkbox" id="live" style="width:auto;margin-right:8px">Stream abhi live chal rahi hai (live-from-start)</label>
  <label><input type="checkbox" id="divider" checked style="width:auto;margin-right:8px">Beech me pink divider line</label>
  <label>cookies.txt (optional - YouTube bot-check aaye tab)</label><input type="file" id="cookies" accept=".txt">
 </details>
 <button class="go" id="go">&#9889; CLIPS BANAO</button>
</div>

<div class="card"><div class="layout"><div class="phone"><div class="a">GAMEPLAY</div><div class="b">FACE</div></div>
<p><b>Output:</b> 1080&times;1920 (9:16). Upar half me gameplay/screen, neeche half me streamer ka face (auto zoom). Har clip tere bataye time ki exact hogi, ek ke baad ek peeche se.</p></div></div>

<div class="card hide" id="prog"><div class="bar"><i id="pb"></i></div><div class="hint" id="pt">Shuru...</div><div class="log" id="log"></div><div class="err" id="err"></div></div>
<div class="card hide" id="res"><div style="font-weight:800;font-size:18px" id="rt">Clips ready</div><div class="clips" id="clips"></div><a class="zip hide" id="zip">&#11015; Saari clips ek saath (ZIP)</a></div>
</div>
<script>
const $=id=>document.getElementById(id);let D=30,N=5,MODE='fit';
function pt(s){s=(s||'').trim().replace(/^[-\u2013\u2014\s]+/,'');if(!s)return null;if(/^\d+(:\d+){1,2}$/.test(s)){return s.split(':').reduce((a,p)=>a*60+(+p),0)}if(/^\d+$/.test(s))return +s;return NaN}
function ft(x){x=Math.round(x);const h=Math.floor(x/3600),m=Math.floor(x%3600/60),s=x%60;return h+':'+String(m).padStart(2,'0')+':'+String(s).padStart(2,'0')}
function upd(){const custom=document.querySelector('#presets .on').dataset.d==='0';if(custom){D=+$('cd').value||0;N=+$('cn').value||0}
 const st=pt($('start').value);let t='';if(isNaN(st))t='Start time ka format galat hai (01:27:02 aisa likh)';else{const s=st||0;t=N+' clip &times; '+D+'s = '+ft(D*N)+' total  |  '+ft(s)+' se '+ft(s+D*N)+' tak'}$('sum').innerHTML=t}
document.querySelectorAll('#presets .chip').forEach(c=>c.onclick=()=>{document.querySelectorAll('#presets .chip').forEach(x=>x.classList.remove('on'));c.classList.add('on');if(c.dataset.d==='0'){$('custom').classList.remove('hide')}else{$('custom').classList.add('hide');D=+c.dataset.d;N=+c.dataset.n}upd()});
document.querySelectorAll('#mode div').forEach(c=>c.onclick=()=>{document.querySelectorAll('#mode div').forEach(x=>x.classList.remove('on'));c.classList.add('on');MODE=c.dataset.v});
['start','cd','cn'].forEach(i=>$(i).addEventListener('input',upd));upd();
let timer=null;
$('go').onclick=async()=>{upd();const fd=new FormData();fd.append('url',$('url').value);fd.append('start',$('start').value);fd.append('dur',D);fd.append('count',N);fd.append('mode',MODE);fd.append('face',$('face').value);
 fd.append('live',$('live').checked?'1':'0');fd.append('divider',$('divider').checked?'1':'0');if($('cookies').files[0])fd.append('cookies',$('cookies').files[0]);
 $('err').textContent='';$('res').classList.add('hide');$('clips').innerHTML='';$('prog').classList.remove('hide');$('go').disabled=true;$('pb').style.width='2%';$('log').textContent='';
 const r=await fetch('api/start',{method:'POST',body:fd});const j=await r.json();
 if(!r.ok){$('err').textContent=j.error;$('go').disabled=false;return}
 clearInterval(timer);timer=setInterval(()=>poll(j.id),1200);poll(j.id);$('prog').scrollIntoView({behavior:'smooth'})};
async function poll(id){const r=await fetch('api/status/'+id);if(!r.ok)return;const s=await r.json();$('pb').style.width=Math.max(2,s.progress)+'%';$('pt').textContent=s.progress+'%  '+(s.title?('- '+s.title):'');
 const L=$('log');L.textContent=s.logs.join('\n');L.scrollTop=L.scrollHeight;
 if(s.status==='error'){clearInterval(timer);$('err').textContent=s.error;$('go').disabled=false}
 if(s.status==='done'){clearInterval(timer);$('go').disabled=false;$('res').classList.remove('hide');$('rt').textContent=s.clips.length+' clips ready \u2705';
  $('clips').innerHTML=s.clips.map(c=>'<div class="clip"><video src="'+c.url+'" poster="'+c.thumb+'" controls playsinline preload="metadata"></video><small>'+c.label+' ('+c.size+' MB)</small><a class="dl" href="'+c.dl+'" download>\u2B07 Download</a></div>').join('');
  if(s.zip){$('zip').href=s.zip;$('zip').classList.remove('hide')}$('res').scrollIntoView({behavior:'smooth'})}}
</script></body></html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n  CLIP KATARNAK chal raha hai ->  http://127.0.0.1:%d\n" % port)
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, threaded=True)
