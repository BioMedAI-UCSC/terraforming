"""Compile unchanged pinned Ames routines behind small validation adapters."""
import hashlib
from pathlib import Path
import re
import subprocess


def compile_routines(source, expected_blob, names, header, adapter, destination, compiler):
    raw = Path(source).read_bytes()
    blob = hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if blob != expected_blob:
        raise ValueError("Ames reference differs from the pinned v3.2.1 source")
    text = raw.decode()
    routines = []
    for name in names:
        start = re.search(r"(?im)^(?:real(?:\*8)?\s+)?(subroutine|function)\s+"+
            re.escape(name)+r"\s*\(",text)
        if start is None:
            raise ValueError(f"missing reference routine {name}")
        end = re.search(r"(?im)^end\s+"+start.group(1)+r"\s+"+re.escape(name)+r"\b",
            text[start.start():])
        if end is None:
            raise ValueError(f"missing end of reference routine {name}")
        routines.append(text[start.start():start.start()+end.end()])
    destination = Path(destination)
    code = destination/"reference.f90"
    code.write_text(header+"\n".join(routines)+adapter)
    library = destination/"reference.so"
    flags = ["-O3","-cpp","-fdefault-real-8","-fdefault-double-8","-fPIC","-shared"]
    result = subprocess.run([compiler,*flags,str(code),"-o",str(library)],cwd=destination,
        capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f"Fortran reference compilation failed:\n{result.stderr}")
    return library,dict(source_git_blob=blob,
        compiled_source_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
        routine_sha256={name:hashlib.sha256(body.encode()).hexdigest()
            for name,body in zip(names,routines,strict=True)},
        compiler=subprocess.check_output([compiler,"--version"],text=True).splitlines()[0],
        compiler_flags=flags,real_precision_bits=64,
        adaptation="module dependency declarations and C array adapters; routine bodies unchanged")
