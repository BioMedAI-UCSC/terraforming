# Deployments

Deployment methods describe how to select experiment code, provide resources,
run it, and retrieve its outputs. Experiment folders own their scientific
protocols and dependencies; deployment tooling owns execution and storage.

| Deployment option | Use |
| --- | --- |
| [Nautalis](nautalis.md) | Run branch-selected experiment folders on the Nautilus Kubernetes cluster, or use the same image with mounted local source. |

Each deployment guide documents its inputs, experiment entry point, output
layout, and download command. Additional deployment methods can be added here
without changing the experiment folders.
