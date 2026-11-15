"""Stage hashed inputs and submit the six immutable rebuttal jobs to Nautilus."""
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
    campaigns.append(job)
    volumes.append(dict(name=task, persistentVolumeClaim=dict(claimName=name)))
    mounts.append(dict(name=task, mountPath=f"/data/{task}"))

stage_name = f"aegis-54586-stage-{a.tag}"
stage = dict(apiVersion="batch/v1", kind="Job", metadata=dict(name=stage_name), spec=dict(
    backoffLimit=0, template=dict(spec=dict(restartPolicy="Never", automountServiceAccountToken=False,
        securityContext=dict(runAsNonRoot=True, runAsUser=10001, runAsGroup=10001, fsGroup=10001),
        containers=[dict(name="stage", image="busybox:1.37.0", command=["sh", "-c",
            "until test -f /data/projected/transfer-complete; do sleep 1; done"],
            resources=dict(requests=dict(cpu="1", memory="1Gi"), limits=dict(cpu="1", memory="1Gi")),
            securityContext=dict(allowPrivilegeEscalation=False, readOnlyRootFilesystem=True,
                capabilities=dict(drop=["ALL"])), volumeMounts=mounts)], volumes=volumes))))
record = dict(source=a.ref, staged=str(a.staged), stage=stage, jobs=campaigns)
destination = Path(__file__).with_name("deployment.json")
destination.write_text(json.dumps(record, indent=2)+"\n")
apply(stage)
while True:
    pods = read("get", "pods", "-l", f"job-name={stage_name}")["items"]
    if pods and pods[0]["status"]["phase"] == "Running":
        pod = pods[0]["metadata"]["name"]
        break
    if pods and pods[0]["status"]["phase"] == "Failed":
        raise RuntimeError(pods[0]["status"])
    time.sleep(2)
manifest = json.loads((a.staged / "staged-sha256.json").read_text())
for campaign in campaigns:
    task = campaign["metadata"]["labels"]["task"]
    for folder in ("reviewer-inputs", "reviewer-windows", "rebuttal-references"):
        subprocess.run([*kube, "cp", str(a.staged/folder), f"{pod}:/data/{task}/{folder}"], check=True)
    subprocess.run([*kube, "cp", str(a.staged/"staged-sha256.json"),
        f"{pod}:/data/{task}/staged-sha256.json"], check=True)
    checks = "".join(f"{digest}  /data/{task}/{path.removeprefix('/work/')}\n" for path, digest in manifest.items())
    verified = subprocess.run([*kube, "exec", "-i", pod, "--", "sha256sum", "-c", "-"],
        input=checks, text=True, capture_output=True, check=True)
    print(f"Verified {len(manifest)} input hashes for {task}", flush=True)
subprocess.run([*kube, "exec", pod, "--", "touch", "/data/projected/transfer-complete"], check=True)
while read("get", "pod", pod)["status"]["phase"] not in ("Succeeded", "Failed"):
    time.sleep(2)
if read("get", "pod", pod)["status"]["phase"] != "Succeeded":
    raise RuntimeError("Input staging failed")
for campaign in campaigns:
    apply(campaign)
record["submitted"] = [j["metadata"]["name"] for j in campaigns]
destination.write_text(json.dumps(record, indent=2)+"\n")
print("Submitted all six jobs; no deadlines or timed deletion", flush=True)
