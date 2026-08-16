"""Execute a notebook in place with jupyter_client (nbclient is not installed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from jupyter_client.manager import KernelManager


def run(path: Path, cwd: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    km = KernelManager(kernel_name="python3")
    km.start_kernel(cwd=str(cwd))
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=120)
    failures = 0
    count = 0
    try:
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            count += 1
            source = "".join(cell["source"])
            msg_id = kc.execute(source)
            outputs = []
            while True:
                msg = kc.get_iopub_msg(timeout=600)
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                kind, content = msg["msg_type"], msg["content"]
                if kind == "status" and content["execution_state"] == "idle":
                    break
                if kind == "stream":
                    if outputs and outputs[-1].get("output_type") == "stream" \
                            and outputs[-1]["name"] == content["name"]:
                        outputs[-1]["text"] += content["text"]
                    else:
                        outputs.append({"output_type": "stream",
                                        "name": content["name"],
                                        "text": content["text"]})
                elif kind in ("display_data", "execute_result"):
                    out = {"output_type": kind,
                           "data": content["data"],
                           "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        out["execution_count"] = count
                    outputs.append(out)
                elif kind == "error":
                    failures += 1
                    outputs.append({"output_type": "error",
                                    "ename": content["ename"],
                                    "evalue": content["evalue"],
                                    "traceback": content["traceback"]})
                    print(f"CELL {index} FAILED: {content['ename']}: {content['evalue']}",
                          file=sys.stderr)
                    print("\n".join(content["traceback"])[-2500:], file=sys.stderr)
            cell["outputs"] = outputs
            cell["execution_count"] = count
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"executed {count} code cells, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
