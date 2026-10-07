"""The writer: a training script imports this and a run folder appears.

    import tracker
    run = tracker.start("runs/f8192", config=vars(args), total=steps)
    run.log(step, loss=0.31, sky={"feature": 1649, "responds_to": [" sky", " skies"]})
    run.save("runs/f8192/step_0000488.pt")
    run.finish()

This file stands alone. It uses only the standard library, opens no network connection, and runs
on Python 3.8 and later, so it can be copied next to a training script on a cluster as it is.

Nothing here is allowed to stop the training. Every write is wrapped: if the disk is full or a
file cannot be opened, a warning goes to stderr and the call returns.

The run folder it writes is described in docs/format.md.
"""
import atexit, hashlib, json, math, os, shlex, signal, socket, subprocess, sys, threading, time

FORMAT = 1
_warned = set()


def _warn(what, err):
    """Say once per kind of failure that something could not be written, and carry on."""
    key = (what, type(err).__name__)
    if key in _warned:
        return
    _warned.add(key)
    try:
        sys.stderr.write("tracker: could not %s (%s: %s); training continues\n" % (what, type(err).__name__, err))
        sys.stderr.flush()
    except Exception:
        pass


def _plain(v, depth=0):
    """Turn a value into something json can write: numbers, strings, lists and dicts of those.

    Tensors and numpy values become numbers or lists. A number that is not finite becomes the
    string "NaN", "Infinity" or "-Infinity", so every line stays strict JSON.
    """
    if v is None or isinstance(v, (bool, str, int)):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return v
    if depth > 8:
        return str(v)
    if isinstance(v, dict):
        return {str(k): _plain(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_plain(x, depth + 1) for x in v]
    for attr in ("item", "tolist"):                       # numpy and torch scalars, then arrays
        f = getattr(v, attr, None)
        if callable(f):
            try:
                return _plain(f(), depth + 1)
            except Exception:
                pass
    return str(v)


def _append(path, obj):
    """Add one line to a file with a single write, so a reader never sees half of it mixed with another."""
    data = (json.dumps(_plain(obj), ensure_ascii=False, separators=(", ", ": ")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _replace(path, obj):
    """Overwrite a file in one move, so a reader sees the old version or the new one and nothing between."""
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_plain(obj), f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def _read(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _out(cmd, cwd=None, timeout=10):
    try:
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
        return r.stdout.decode("utf-8", "replace").strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git(code_dir):
    """The commit the code is at, and whether anything was changed since."""
    top = _out(["git", "rev-parse", "--show-toplevel"], cwd=code_dir)
    if not top:
        return {"commit": None}
    changed = _out(["git", "status", "--porcelain", "--untracked-files=no"], cwd=top)
    return {"commit": _out(["git", "rev-parse", "HEAD"], cwd=top),
            "branch": _out(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=top),
            "dirty": bool(changed),
            "changed_files": [l.split(None, 1)[-1] for l in changed.splitlines()][:50] if changed else [],
            "remote": _out(["git", "config", "--get", "remote.origin.url"], cwd=top),
            "root": top,
            "path": os.path.relpath(code_dir, top)}


def _job():
    """The scheduler's number for this job, if the script was started by one."""
    for name, var in (("lsf", "LSB_JOBID"), ("slurm", "SLURM_JOB_ID"), ("pbs", "PBS_JOBID"), ("sge", "JOB_ID")):
        if os.environ.get(var):
            return {"scheduler": name, "job_id": os.environ[var],
                    "job_name": os.environ.get("LSB_JOBNAME") or os.environ.get("SLURM_JOB_NAME") or os.environ.get("PBS_JOBNAME"),
                    "queue": os.environ.get("LSB_QUEUE") or os.environ.get("SLURM_JOB_PARTITION")}
    return {"scheduler": None, "job_id": None}


GPU_QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


def _gpus():
    """One dict per GPU that nvidia-smi can see, or an empty list where there is none."""
    out = _out(["nvidia-smi", "--query-gpu=" + GPU_QUERY, "--format=csv,noheader,nounits"], timeout=5)
    if not out:
        return []
    def num(s):
        try:
            return float(s)
        except Exception:
            return None
    rows = []
    for line in out.splitlines():
        c = [x.strip() for x in line.split(",")]
        if len(c) >= 7:
            rows.append({"index": int(num(c[0]) or 0), "name": c[1], "util": num(c[2]), "mem_mb": num(c[3]),
                         "mem_total_mb": num(c[4]), "temp_c": num(c[5]), "power_w": num(c[6])})
    return rows


def _cpu_mem_mb():
    """Memory this process holds, in megabytes."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss       # the peak, where the current figure is not on offer
        return round(peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0, 1)
    except Exception:
        return None


class Run:
    """One run folder being written. Made by start()."""

    def __init__(self, folder, config=None, total=None, prediction=None, name=None,
                 system_every=5.0, heartbeat_every=20.0, code_dir=None, project=None, group=None):
        self.folder = os.path.abspath(folder)
        self.total = total
        self.step = None
        self.state = "running"
        self._system_every, self._heartbeat_every = system_every, heartbeat_every
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._last_log = None
        self._started = time.time()
        self._error = None
        try:
            os.makedirs(self.folder, exist_ok=True)
        except Exception as e:
            _warn("make the run folder", e)
        self._start_files(config, prediction, name, code_dir, project, group)
        self._write_status()
        try:
            self._thread = threading.Thread(target=self._background, name="tracker", daemon=True)
            self._thread.start()
        except Exception as e:
            _warn("start the sampling thread", e)
        self._hooks()

    def _p(self, name):
        return os.path.join(self.folder, name)

    # ---------- written once, at the start ----------
    def _start_files(self, config, prediction, name, code_dir, project=None, group=None):
        try:
            code_dir = code_dir or os.path.dirname(os.path.abspath(sys.argv[0] or ".")) or os.getcwd()
            gpus = _gpus()
            here = {"time": self._started, "command": " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv),
                    "cwd": os.getcwd(), "host": socket.gethostname(), "user": os.environ.get("USER"),
                    "python": sys.version.split()[0], "pid": os.getpid(),
                    "gpus": [{"name": g["name"], "mem_total_mb": g["mem_total_mb"]} for g in gpus],
                    "git": _git(code_dir)}
            here.update(_job())
            old = _read(self._p("meta.json"))
            if old and os.path.exists(self._p("log.jsonl")):
                # A restart: the first start stays as it was, and this one is added to the list beneath it.
                entry = dict(here)
                if config is not None and _plain(config) != _read(self._p("config.json")):
                    entry["config"] = config                            # the settings this time, where they differ
                old.setdefault("restarts", []).append(entry)
                meta = old
            else:
                meta = dict(here)
                meta.update({"format": FORMAT, "name": name or os.path.basename(self.folder), "restarts": []})
                if prediction:
                    meta["prediction"] = prediction
                if project:
                    meta["project"] = project
                if group:
                    meta["group"] = group
                _replace(self._p("config.json"), config if config is not None else {})
            _replace(self._p("meta.json"), meta)
            self._first_started = meta.get("time", self._started)
            self._restarts = len(meta.get("restarts", []))
        except Exception as e:
            self._first_started, self._restarts = self._started, 0
            _warn("write meta.json", e)

    # ---------- the common case ----------
    def log(self, step, **fields):
        """Add one line to log.jsonl. Fields can be numbers, short strings, lists or nested dicts."""
        try:
            now = time.time()
            rec = {"step": step, "_t": round(now, 3)}
            rec.update(fields)
            with self._lock:
                _append(self._p("log.jsonl"), rec)
                self.step, self._last_log = step, now
        except Exception as e:
            _warn("write to log.jsonl", e)

    def save(self, path, kind=None, checksum=True, **extra):
        """Register a file the run has saved, such as a snapshot of the weights. The file stays where it is.

        The entry carries the last logged step; pass step=... where the file belongs to a different one.
        """
        try:
            path = os.path.abspath(path)
            rel = os.path.relpath(path, self.folder)
            entry = {"path": path if rel.startswith("..") else rel, "size": os.path.getsize(path),
                     "step": self.step, "_t": round(time.time(), 3)}
            if kind:
                entry["kind"] = kind
            if checksum:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(1 << 22), b""):
                        h.update(block)
                entry["sha256"] = h.hexdigest()
            entry.update(extra)
            with self._lock:
                _append(self._p("artifacts.jsonl"), entry)
        except Exception as e:
            _warn("register a saved file", e)

    def describe(self, texts=None, **more):
        """Say in a sentence what each logged name means, so the viewer can show it under the chart.

            run.describe(not_rebuilt="The share of the input the network failed to rebuild. Lower is better.",
                         sky="The strongest feature for the word sky, and what it responds to.")

        A name can be a field ("not_rebuilt"), a nested field ("sky.fired") or the object that holds it ("sky").
        """
        try:
            texts = dict(texts or {}, **more)
            with self._lock:
                meta = _read(self._p("meta.json"), {}) or {}
                meta.setdefault("about", {}).update({str(k): str(v) for k, v in texts.items()})
                _replace(self._p("meta.json"), meta)
        except Exception as e:
            _warn("write the descriptions to meta.json", e)

    def finish(self):
        """Mark the run finished. Called for you when the script ends normally."""
        self._close("finished")

    def fail(self, error=None):
        """Mark the run failed. Called for you when the script ends on an exception or is told to stop."""
        self._close("failed", error)

    # ---------- status and the background thread ----------
    def _write_status(self):
        try:
            now = time.time()
            s = {"state": self.state, "step": self.step, "total": self.total, "heartbeat": round(now, 3),
                 "heartbeat_every": self._heartbeat_every, "last_log": self._last_log,
                 "started": self._first_started, "restarts": self._restarts}
            if self.state != "running":
                s["ended"] = round(now, 3)
            if self._error:
                s["error"] = self._error
            with self._lock:
                _replace(self._p("status.json"), s)
        except Exception as e:
            _warn("write status.json", e)

    def _sample(self):
        try:
            rec = {"_t": round(time.time(), 3), "step": self.step, "cpu_mem_mb": _cpu_mem_mb()}
            try:
                rec["load"] = round(os.getloadavg()[0], 2)
            except Exception:
                pass
            gpus = _gpus()
            if gpus:
                rec["gpu"] = [{k: g[k] for k in ("index", "util", "mem_mb", "temp_c", "power_w")} for g in gpus]
            with self._lock:
                _append(self._p("system.jsonl"), rec)
        except Exception as e:
            _warn("write to system.jsonl", e)

    def _background(self):
        next_beat = 0.0
        while not self._stop.is_set():
            if self._system_every:
                self._sample()
            if time.time() >= next_beat:
                self._write_status()
                next_beat = time.time() + self._heartbeat_every
            self._stop.wait(self._system_every or self._heartbeat_every)

    def _close(self, state, error=None):
        if self._closed:
            return
        self._closed = True
        self.state = state
        if error is not None:
            self._error = str(error)[:2000]
        self._stop.set()
        self._write_status()

    # ---------- noticing how the script ends ----------
    def _hooks(self):
        try:
            atexit.register(self._at_exit)
            previous = sys.excepthook
            def on_error(kind, value, trace):
                if kind is KeyboardInterrupt:
                    self.fail("interrupted")
                else:
                    self.fail("%s: %s" % (kind.__name__, value))
                previous(kind, value, trace)
            sys.excepthook = on_error
            if threading.current_thread() is threading.main_thread():
                # A scheduler ends a job that has run out of time with a signal. Note it, then let it take effect.
                for sig in (signal.SIGTERM, getattr(signal, "SIGUSR2", None)):
                    if sig is not None and signal.getsignal(sig) is signal.SIG_DFL:
                        signal.signal(sig, self._on_signal)
        except Exception as e:
            _warn("set up the end-of-run hooks", e)

    def _on_signal(self, sig, frame):
        self.fail("stopped by signal %d" % sig)
        signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)

    def _at_exit(self):
        self._close("finished")

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        if kind is None:
            self.finish()
        else:
            self.fail("interrupted" if kind is KeyboardInterrupt else "%s: %s" % (kind.__name__, value))
        return False


def start(folder, config=None, **options):
    """Begin (or continue) the run in this folder and return the object to log to.

    config      every setting the run was started with; written to config.json
    total       the last step the run will reach, if known, so the viewer can say how far along it is
    prediction  what you expect to see, written down before the run; shown beside the result afterwards
    name        a name for the run; the folder's name if left out
    project     which project the run belongs to; the viewer keeps projects apart
    system_every, heartbeat_every   seconds between GPU samples and between heartbeats
    """
    return Run(folder, config, **options)
