"""Print Kubernetes JSON (also valid YAML); never changes the active context."""
import argparse
import json
from run import experiments


def resources(gpu):
    requests = {"cpu": "8", "memory": "64Gi", "ephemeral-storage": "2Gi"}
    limits = {"cpu": "8", "memory": "64Gi", "ephemeral-storage": "8Gi"}
    if gpu:
        requests["nvidia.com/a100"] = limits["nvidia.com/a100"] = "1"
    return {"requests": requests, "limits": limits}


def manifests(task, image, run_id):
    gpu = task != "stage"
    affinity = [{"key": "topology.kubernetes.io/region", "operator": "In", "values": ["us-west"]}]
    if gpu:
        affinity.append({"key": "nvidia.com/cuda.driver.major", "operator": "Gt", "values": ["524"]})
    pod = {
        "restartPolicy": "Never", "automountServiceAccountToken": False,
        "terminationGracePeriodSeconds": 120,
        "securityContext": {"runAsNonRoot": True, "runAsUser": 10001, "runAsGroup": 10001, "fsGroup": 10001, "fsGroupChangePolicy": "OnRootMismatch", "seccompProfile": {"type": "RuntimeDefault"}},
        "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{"matchExpressions": affinity}]}}},
        "containers": [{"name": "paper", "image": image, "imagePullPolicy": "Always", "args": [task, "--run-id", run_id], "resources": resources(gpu),
                        "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True, "capabilities": {"drop": ["ALL"]}},
                        "volumeMounts": [{"name": "work", "mountPath": "/work"}, {"name": "tmp", "mountPath": "/tmp"}, {"name": "shm", "mountPath": "/dev/shm"}]}],
        "volumes": [{"name": "work", "persistentVolumeClaim": {"claimName": "paper-experiments"}}, {"name": "tmp", "emptyDir": {"sizeLimit": "4Gi"}}, {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}}],
    }
    return {"apiVersion": "v1", "kind": "List", "items": [
        {"apiVersion": "v1", "kind": "PersistentVolumeClaim", "metadata": {"name": "paper-experiments", "namespace": "razvanlab"}, "spec": {"storageClassName": "rook-ceph-block", "accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": "500Gi"}}}},
        {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": f"paper-{task}", "namespace": "razvanlab"}, "spec": {"backoffLimit": 2, "activeDeadlineSeconds": 604800, "template": {"metadata": {"labels": {"app": "paper-experiments"}}, "spec": pod}}},
    ]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["stage", "all", *experiments()])
    parser.add_argument("--image", default="samkash/terraforming-paper@sha256:47ef0396b05394d0d19395c99ac54f058356c486b8def7213669641cfd17a7f3")
    parser.add_argument("--run-id", default="paper-v1")
    args = parser.parse_args()
    print(json.dumps(manifests(args.task, args.image, args.run_id), indent=2))
