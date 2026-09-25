"""Run the entity resolution work on the NITJ H100 cluster (PBS, workq, one MIG slice).

Nothing heavy runs on the laptop. The laptop only writes code, sends it, queues jobs and pulls results.

  python run_h100.py push                       send run_h100.py + code/ to the cluster
  python run_h100.py submit NAME [opts] -- CMD  push, then queue a GPU job that runs CMD in the project folder
        opts: --hours 4  --ncpus 8  --mem 32gb  --after JOBID  --expect MINUTES (expected run time, default 30)
              --env K=V (repeatable; e.g. --env ER_NORM=v2 --env ER_CACHE=data/cache_v2)
        example: python run_h100.py submit eda --hours 1 -- python -u code/eda/eda.py
  python run_h100.py status                     my jobs in the queue + last lines of recent job logs
  python run_h100.py log NAME [N]               last N lines of a job log
  python run_h100.py pull [SUBDIR]              copy results/ (or results/SUBDIR) back here
  python run_h100.py wait JOBID [JOBID...]       block until these jobs leave the queue
  python run_h100.py pick-mig                   (cluster side) print a free MIG slice id
"""
import os
import re
import shlex
import subprocess
import sys
import time

HOST = "10.10.11.201"
USER = "ai_25901334"
RDIR = "/tmp/ai_25901334/AmazonML"          # root disk; /Data3 is almost full
CONDA_SH = "/apps/compilers/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV = "refused_h100"                   # torch 2.5.1+cu121; extra packages go to RDIR/pkgs
HERE = os.path.dirname(os.path.abspath(__file__))


def ssh_base():
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=20", f"{USER}@{HOST}"]


def tar_exe():
    win = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tar.exe")
    return win if os.name == "nt" and os.path.exists(win) else "tar"


def rsh(script, capture=False):
    """Run a bash script on the login node (only light commands: qsub, qstat, tail, tar)."""
    # bytes, not text: text mode on Windows turns \n into \r\n and bash then fails on '/bin/bash^M'
    r = subprocess.run(ssh_base() + ["bash -s"], input=("export PATH=/opt/pbs/bin:$PATH\n" + script).encode(),
                       capture_output=capture)
    if capture:
        r.stdout = r.stdout.decode("utf-8", "replace")
        r.stderr = r.stderr.decode("utf-8", "replace")
    return r


def cmd_push():
    members = [m for m in ["run_h100.py", "code"] if os.path.exists(os.path.join(HERE, m))]
    tar = subprocess.Popen([tar_exe(), "-cf", "-", "--exclude=__pycache__"] + members, cwd=HERE, stdout=subprocess.PIPE)
    ssh = subprocess.Popen(ssh_base() + [f"mkdir -p {RDIR}/results/_jobs && tar -xf - -C {RDIR}"], stdin=tar.stdout)
    tar.stdout.close()
    rc = ssh.wait() or tar.wait()
    if rc:
        sys.exit("push failed")
    print("pushed", ", ".join(members))


def pbs_script(name, cmd, hours, ncpus, mem, expect=30, env=()):
    out = f"{RDIR}/results/_jobs/{name}.out"
    return "\n".join([
        "#!/bin/bash", f"#PBS -N er_{name}"[:17], "#PBS -q workq",
        f"#PBS -l select=1:ncpus={ncpus}:ngpus=1:mem={mem}", f"#PBS -l walltime={int(hours):02d}:{int(hours % 1 * 60):02d}:00",
        "#PBS -j oe", f"#PBS -o {out}", "",
        f"exec >> {RDIR}/results/_jobs/{name}.log 2>&1",
        f"cd {RDIR}", "export PYTHONNOUSERSITE=1", f"export NCPUS={ncpus}", f"export OMP_NUM_THREADS={ncpus}",
        f"export POLARS_MAX_THREADS={ncpus}", f"export RAYON_NUM_THREADS={ncpus}",
        f"export PYTHONPATH={RDIR}/pkgs:{RDIR}/code", f"export HF_HOME={RDIR}/hf_cache", "export TOKENIZERS_PARALLELISM=false",
        f"source {CONDA_SH}", f"conda activate {CONDA_ENV}",
        *[f"export {e}" for e in env],
        'echo "job $PBS_JOBID on $(hostname) at $(date -Iseconds); PBS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
        'export PBS_CUDA_ORIG="${CUDA_VISIBLE_DEVICES:-unset}"',
        'MIG=""',
        'sleep $((RANDOM % 20)); for try in $(seq 1 120); do MIG=$(python run_h100.py pick-mig 2>/dev/null) && break; MIG=""; '
        'echo "no free MIG slice yet (try $try), waiting 60 s"; sleep 60; done',
        '[ -z "$MIG" ] && { echo "no free MIG slice after 2 hours"; exit 3; }',
        'export CUDA_VISIBLE_DEVICES="$MIG"', 'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"',
        'trap "rm -f ' + RDIR + '/results/_jobs/mig_locks/$MIG" EXIT',
        "# all our jobs were SIGKILLed at 10:45:01 and 12:44:02 (every 2 h at ~:44, cause not visible to us):",
        "# if the expected run would cross the next such window, wait until it has passed",
        f"EXPECT={expect}",
        'M=$(( 10#$(date +%H) * 60 + 10#$(date +%M) )); TO=$(( (44 - M % 120 + 120) % 120 ))',
        'if [ $TO -lt $((EXPECT + 2)) ]; then W=$(( (TO + 3) * 60 )); echo "waiting ${W}s to skip the :44 window"; sleep $W; fi',
        "T0=$(date +%s)", cmd, "RC=$?",
        'echo "EXIT rc=$RC after $(( $(date +%s) - T0 )) s at $(date -Iseconds)"', "exit $RC", ""])


def cmd_submit(argv):
    if "--" not in argv:
        sys.exit("usage: submit NAME [--hours H --ncpus N --mem M --after JOBID] -- CMD ...")
    i = argv.index("--")
    opts, cmd = argv[:i], argv[i + 1:]
    name = opts[0]
    get = lambda k, d: opts[opts.index(k) + 1] if k in opts else d
    hours, ncpus, mem, after = float(get("--hours", 4)), int(get("--ncpus", 8)), get("--mem", "32gb"), get("--after", None)
    expect = int(get("--expect", 30))
    env = [opts[i + 1] for i, o in enumerate(opts) if o == "--env"]
    cmd_push()
    script = pbs_script(name, " ".join(shlex.quote(c) for c in cmd), hours, ncpus, mem, expect, env)
    dep = f"-W depend=afterok:{after} " if after else ""
    r = rsh(f"cat > {RDIR}/results/_jobs/{name}.pbs <<'PBSEOF'\n{script}PBSEOF\n"
            f"rm -f {RDIR}/results/_jobs/{name}.out {RDIR}/results/_jobs/{name}.log; qsub {dep}{RDIR}/results/_jobs/{name}.pbs", capture=True)
    if r.returncode:
        sys.exit("qsub failed: " + r.stderr)
    jid = r.stdout.strip().splitlines()[-1]
    with open(os.path.join(HERE, "results", "_jobs.log"), "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{jid}\t{name}\t{' '.join(cmd)}\n")
    print(jid)


def cmd_status():
    rsh(f"qstat -u {USER} 2>/dev/null; cd {RDIR}/results/_jobs 2>/dev/null && for f in $(ls -t *.log 2>/dev/null | head -4); "
        f"do echo \"== $f\"; tail -n 4 $f; done")


def cmd_log(name, n=60):
    rsh(f"cd {RDIR}/results/_jobs; f={name}.log; [ -f $f ] || f={name}.out; tail -n {n} $f")


def cmd_wait(jids):
    while True:
        r = rsh(f"qstat {' '.join(jids)} 2>/dev/null | tail -n +3", capture=True)
        alive = [l for l in r.stdout.splitlines() if l.strip() and not l.strip().endswith(" F")]
        if r.returncode == 0 and not alive:
            return
        if r.returncode not in (0, 35, 153) and not alive:
            return
        time.sleep(30)


def cmd_pull(sub=""):
    src = "results" + ("/" + sub if sub else "")
    ssh = subprocess.Popen(ssh_base() + [f"tar -cf - --exclude='*.parquet' --exclude='*.npy' --exclude='*.pt' -C {RDIR} {src}"],
                           stdout=subprocess.PIPE)
    tar = subprocess.Popen([tar_exe(), "-xf", "-"], cwd=HERE, stdin=ssh.stdout)
    ssh.stdout.close()
    rc = tar.wait() or ssh.wait()
    print("pull", "done" if rc == 0 else "FAILED", src)


def cmd_pick_mig():
    """Print the UUID of a MIG slice with no process and almost no memory used. Largest slice first."""
    L = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    full = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    devs, gpu = [], None
    for line in L.splitlines():
        m = re.match(r"\s*GPU (\d+):", line)
        if m:
            gpu = int(m.group(1))
            continue
        m = re.search(r"MIG\s+(\S+)\s+Device\s+(\d+):\s*\(UUID:\s*(MIG-[^)\s]+)\)", line)
        if m and gpu is not None:
            devs.append({"gpu": gpu, "dev": int(m.group(2)), "uuid": m.group(3)})
    mem, gici, busy, section = {}, {}, set(), None
    for line in full.splitlines():
        if "MIG devices" in line:
            section = "mig"
        elif "Processes:" in line:
            section = "proc"
        if section == "mig":
            m = re.match(r"\|\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\|\s+(\d+)MiB\s*/\s*(\d+)MiB", line)
            if m:
                g, gi, ci, d, used, total = map(int, m.groups())
                mem[(g, d)] = (used, total)
                gici[(g, gi, ci)] = d
        elif section == "proc":
            m = re.match(r"\|\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+(C|G|C\+G)\s", line)
            if m and (int(m.group(1)), int(m.group(2)), int(m.group(3))) in gici:
                busy.add((int(m.group(1)), gici[(int(m.group(1)), int(m.group(2)), int(m.group(3)))]))
    free = [d for d in devs if (d["gpu"], d["dev"]) not in busy and mem.get((d["gpu"], d["dev"]), (999, 0))[0] < 200]
    # lock files so two of our jobs that start at the same moment never take the same slice
    lockdir = os.path.join(RDIR, "results", "_jobs", "mig_locks")
    os.makedirs(lockdir, exist_ok=True)
    try:
        q = subprocess.run(["/opt/pbs/bin/qstat", "-u", USER], capture_output=True, text=True).stdout
        running = set(re.findall(r"^(\d+)\.", q, flags=re.M))
    except OSError:
        running = None                     # cannot tell: keep every lock
    for fn in os.listdir(lockdir) if running is not None else []:
        try:
            owner = open(os.path.join(lockdir, fn)).read().strip().split(".")[0]
        except OSError:
            continue
        if owner not in running:
            try:
                os.remove(os.path.join(lockdir, fn))
            except OSError:
                pass
    me = os.environ.get("PBS_JOBID", "manual")
    free.sort(key=lambda d: -mem[(d["gpu"], d["dev"])][1])
    for d in free:
        try:
            fd = os.open(os.path.join(lockdir, d["uuid"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, me.encode())
        os.close(fd)
        print(d["uuid"])
        return
    sys.exit(3)


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return
    c, rest = a[0], a[1:]
    if c == "push":
        cmd_push()
    elif c == "submit":
        cmd_submit(rest)
    elif c == "status":
        cmd_status()
    elif c == "log":
        cmd_log(rest[0], int(rest[1]) if len(rest) > 1 else 60)
    elif c == "wait":
        cmd_wait(rest)
    elif c == "pull":
        cmd_pull(rest[0] if rest else "")
    elif c == "pick-mig":
        cmd_pick_mig()
    else:
        sys.exit(f"unknown command {c}")


if __name__ == "__main__":
    main()
