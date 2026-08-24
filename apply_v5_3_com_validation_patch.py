"""Patch V5.3 environment for payload CoM validation.

Run from the project root that contains the `source/` directory:

    python apply_v5_3_com_validation_patch.py

This patch does not change the experiment difficulty. It only:
1) adds debug config flags for payload CoM printing;
2) rewrites `_apply_payload_com_offset()` to Set USD MassAPI attributes robustly;
3) prints requested/read-back mass and centerOfMass before and after cloning.

Use `set_payload_com_offset.py` to switch offsets for A/B tests.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()
ENV_PATH = ROOT / "source/agv_transport/agv_transport/tasks/direct/agv_transport/agv_transport_env.py"
CFG_PATH = ROOT / "source/agv_transport/agv_transport/tasks/direct/agv_transport/agv_transport_env_cfg.py"


def read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def backup(path: Path, suffix: str) -> None:
    bak = path.with_name(path.name + suffix)
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        print(f"[backup] {bak}")


def patch_cfg() -> None:
    text = read(CFG_PATH)
    backup(CFG_PATH, ".bak_com_validate")

    block = '''\n\n    # V5.3-COM-VERIFY: print USD MassAPI values for CoM validation.\n    # This does not change reward, contact, wall, or payload geometry.\n    enable_payload_com_debug_print = True\n    payload_com_debug_print_once = True\n'''

    # Remove old validation block if it exists.
    text = re.sub(
        r"\n\s*# V5\.3-COM-VERIFY: print USD MassAPI values for CoM validation\.[\s\S]*?payload_com_debug_print_once\s*=\s*(?:True|False)\s*\n",
        "\n",
        text,
        count=1,
    )

    pattern = r"(\n\s*payload_com_offset\s*=\s*\([^\n]+\)\s*)"
    if not re.search(pattern, text):
        raise RuntimeError("Could not find `payload_com_offset = (...)` in cfg.py")
    text = re.sub(pattern, r"\1" + block, text, count=1)

    write(CFG_PATH, text)
    print(f"[patched] {CFG_PATH}")


def patch_env() -> None:
    text = read(ENV_PATH)
    backup(ENV_PATH, ".bak_com_validate")

    # Ensure imports include UsdPhysics and Gf.
    if "from pxr import" in text:
        text = re.sub(
            r"from pxr import ([^\n]+)",
            lambda m: "from pxr import " + ", ".join(
                dict.fromkeys([x.strip() for x in m.group(1).split(",")] + ["UsdPhysics", "Gf"])
            ),
            text,
            count=1,
        )
    else:
        text = text.replace("import torch\n", "import torch\nfrom pxr import UsdPhysics, Gf\n", 1)

    new_method = r'''    def _apply_payload_com_offset(self) -> None:
        """Apply and debug-print Payload root CoM offset.

        This method is intentionally limited to USD Physics MassAPI attributes
        on /World/envs/env_0/Payload.  It does not change payload collision
        geometry, contact envelope, wall geometry, reward terms, soft escape,
        or wall-stuck prevention.

        The debug print proves that USD attributes are written/read back.  It
        does not by itself prove that PhysX dynamics uses the offset; that must
        be checked by comparing zero-shot rollouts under different offsets.
        """
        if not bool(getattr(self.cfg, "enable_payload_com_offset", False)):
            if bool(getattr(self.cfg, "enable_payload_com_debug_print", False)):
                print("[V5.3-COM] enable_payload_com_offset=False; no CoM offset applied.")
            return

        stage = sim_utils.get_current_stage()
        payload_path = "/World/envs/env_0/Payload"
        payload_prim = stage.GetPrimAtPath(payload_path)
        if not payload_prim.IsValid():
            raise RuntimeError(f"Cannot apply payload CoM offset: {payload_path} does not exist.")

        if payload_prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(payload_prim)
        else:
            mass_api = UsdPhysics.MassAPI.Apply(payload_prim)

        com = tuple(float(v) for v in getattr(self.cfg, "payload_com_offset", (0.0, 0.0, 0.0)))
        if len(com) != 3:
            raise ValueError("payload_com_offset must be a 3D tuple: (x, y, z).")

        mass = float(getattr(self.cfg, "payload_mass", 24.0))

        mass_attr = mass_api.GetMassAttr()
        if not mass_attr:
            mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(mass)

        com_attr = mass_api.GetCenterOfMassAttr()
        if not com_attr:
            com_attr = mass_api.CreateCenterOfMassAttr()
        com_attr.Set(Gf.Vec3f(com[0], com[1], com[2]))

        if bool(getattr(self.cfg, "enable_payload_com_debug_print", False)):
            print(
                "[V5.3-COM] before_clone path="
                f"{payload_path}, requested_com={com}, "
                f"usd_mass={mass_attr.Get()}, usd_centerOfMass={com_attr.Get()}"
            )

    def _debug_print_payload_mass_api(self, label: str = "after_clone") -> None:
        """Print Payload MassAPI values from env_0 and, if available, env_1.

        This is a diagnostic only.  It verifies authored USD attributes on the
        cloned prims; rollout comparison is still required to verify effective
        PhysX dynamics.
        """
        if not bool(getattr(self.cfg, "enable_payload_com_debug_print", False)):
            return
        if bool(getattr(self.cfg, "payload_com_debug_print_once", True)) and getattr(
            self, "_payload_com_debug_printed", False
        ):
            return

        stage = sim_utils.get_current_stage()
        max_envs = min(int(getattr(self, "num_envs", 1)), 2)
        for env_id in range(max_envs):
            payload_path = f"/World/envs/env_{env_id}/Payload"
            prim = stage.GetPrimAtPath(payload_path)
            if not prim.IsValid():
                print(f"[V5.3-COM] {label} path={payload_path}: prim not found")
                continue
            if not prim.HasAPI(UsdPhysics.MassAPI):
                print(f"[V5.3-COM] {label} path={payload_path}: no MassAPI")
                continue
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            com_attr = mass_api.GetCenterOfMassAttr()
            mass = mass_attr.Get() if mass_attr else None
            com = com_attr.Get() if com_attr else None
            print(f"[V5.3-COM] {label} path={payload_path}, usd_mass={mass}, usd_centerOfMass={com}")

        self._payload_com_debug_printed = True

'''

    pattern = r"    def _apply_payload_com_offset\(self\) -> None:\n[\s\S]*?\n    def _get_manual_boundary_world_points\(self\)"
    if not re.search(pattern, text):
        raise RuntimeError("Could not find `_apply_payload_com_offset` block in env.py")
    text = re.sub(pattern, new_method + "    def _get_manual_boundary_world_points(self)", text, count=1)

    # Add after-clone debug call if not already present.
    if "self._debug_print_payload_mass_api(" not in text:
        text = text.replace(
            "        # 克隆多环境\n        self.scene.clone_environments(copy_from_source=False)\n",
            "        # 克隆多环境\n        self.scene.clone_environments(copy_from_source=False)\n\n"
            "        # V5.3-COM-VERIFY: print cloned Payload MassAPI values.\n"
            "        self._debug_print_payload_mass_api(\"after_clone\")\n",
            1,
        )

    write(ENV_PATH, text)
    print(f"[patched] {ENV_PATH}")


def main() -> None:
    patch_cfg()
    patch_env()
    print("\nDone. Now run py_compile on env.py and cfg.py before evaluation.")


if __name__ == "__main__":
    main()
