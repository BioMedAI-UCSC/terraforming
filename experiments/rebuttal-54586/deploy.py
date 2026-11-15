"""Submit deadline-free managed Jobs and retain records until manual review."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--staged", type=Path, required=True)
p.add_argument("--ref", required=True)
p.add_argument("--tag", required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
kube = ["rtk", "proxy", "kubectl", "--context", "nautilus", "-n", "razvanlab"]
spec = importlib.util.spec_from_file_location("jobs", root / "apps/nautalis/jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)

def apply(body):
    subprocess.run([*kube, "apply", "-f", "-"], input=json.dumps(body), text=True, check=True)

def read(*args):
    return json.loads(subprocess.check_output([*kube, *args, "-o", "json"], text=True))

campaigns, mounts, volumes = [], [], []
for task in ("projected", "unprojected", "spatial", "radiation", "heating", "stability"):
    name = f"aegis-54586-{task}-{a.tag}"
    parsed = jobs.parser().parse_args(["--name", name, "--ref", a.ref, "--experiments",
        "experiments/rebuttal-54586", "--run-id", a.tag, "--storage", "64Gi",
        "--cpu", "4", "--memory", "32Gi", "--gpu", "--", task])
    body = jobs.manifests(parsed)
    job = body["items"][1]
    job["spec"]["backoffLimit"] = 0
    job["metadata"]["labels"] = {"campaign": "aegis-54586", "task": task}
    job["spec"]["template"]["metadata"]["labels"].update(job["metadata"]["labels"])
    job["spec"]["template"]["spec"].pop("affinity")
    apply(body["items"][0])
    # Nautilus injects Job TTLs and six-hour deadlines on standalone Pods.
    # Managed Pods have no deadline; this finalizer prevents timed Job deletion.
    job["metadata"]["finalizers"] = ["aegis.terraforming/retain-results"]
    campaigns.append(job)
    volumes.append(dict(name=task, persistentVolumeClaim=dict(claimName=name)))
    mounts.append(dict(name=task, mountPath=f"/data/{task}"))

stage_name = f"aegis-54586-stage-{a.tag}-recovery"
stage = dict(apiVersion="v1", kind="Pod", metadata=dict(name=stage_name), spec=dict(
    restartPolicy="Never", automountServiceAccountToken=False,
        securityContext=dict(runAsNonRoot=True, runAsUser=10001, runAsGroup=10001, fsGroup=10001),
        containers=[dict(name="stage", image=jobs.IMAGE, command=["sh", "-c",
            "until test -f /data/projected/transfer-complete; do sleep 1; done"],
            resources=dict(requests=dict(cpu="1", memory="1Gi"), limits=dict(cpu="1", memory="1Gi")),
            securityContext=dict(allowPrivilegeEscalation=False, readOnlyRootFilesystem=True,
                capabilities=dict(drop=["ALL"])), volumeMounts=mounts)], volumes=volumes))
record = dict(source=a.ref, staged=str(a.staged), stage=stage, jobs=campaigns,
    reason="Managed Pods have no run deadline. The manual retention finalizer prevents deletion by the cluster-injected completed-Job TTL. PVCs retain all inputs and outputs.")
destination = Path(__file__).with_name("deployment.json")
destination.write_text(json.dumps(record, indent=2)+"\n")
apply(stage)
while True:
    state = read("get", "pod", stage_name)
    if state["status"]["phase"] == "Running":
        pod = stage_name
        break
    if state["status"]["phase"] == "Failed":
        raise RuntimeError(state["status"])
    time.sleep(2)
manifest = json.loads((a.staged / "staged-sha256.json").read_text())
for campaign in campaigns:
    task = campaign["metadata"]["labels"]["task"]
    inspect = "import json,hashlib;from pathlib import Path;m=json.loads("+repr(json.dumps(manifest))+");print(json.dumps([p for p,h in m.items() if not (f:=Path(p.replace('/work/', '/data/"+task+"/',1))).is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=h]))"
    missing = json.loads(subprocess.check_output([*kube, "exec", pod, "--", "python", "-c", inspect], text=True))
    for remote in missing:
        relative = remote.removeprefix("/work/")
        remote_destination = f"/data/{task}/{relative}"
        subprocess.run([*kube, "exec", pod, "--", "mkdir", "-p", str(Path(remote_destination).parent)], check=True)
        subprocess.run([*kube, "cp", str(a.staged/relative), f"{pod}:{remote_destination}"], check=True)
    subprocess.run([*kube, "cp", str(a.staged/"staged-sha256.json"),
        f"{pod}:/data/{task}/staged-sha256.json"], check=True)
    checks = "".join(f"{digest}  /data/{task}/{path.removeprefix('/work/')}\n" for path, digest in manifest.items())
    verified = subprocess.run([*kube, "exec", "-i", pod, "--", "sha256sum", "-c", "-"],
        input=checks, text=True, capture_output=True, check=True)
    print(f"Verified {len(manifest)} input hashes for {task}; recovered {len(missing)} files", flush=True)
subprocess.run([*kube, "exec", pod, "--", "touch", "/data/projected/transfer-complete"], check=True)
while read("get", "pod", pod)["status"]["phase"] not in ("Succeeded", "Failed"):
    time.sleep(2)
if read("get", "pod", pod)["status"]["phase"] != "Succeeded":
    raise RuntimeError("Input staging failed")
for campaign in campaigns:
    apply(campaign)
record["submitted"] = [j["metadata"]["name"] for j in campaigns]
destination.write_text(json.dumps(record, indent=2)+"\n")
print("Submitted six deadline-free Jobs with manual retention finalizers", flush=True)
