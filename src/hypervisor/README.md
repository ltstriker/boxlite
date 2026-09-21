# boxlite-hypervisor

Host backends for BoxLite's VMM: HVF on macOS arm64 and KVM on Linux x86_64
and arm64, with WHP on Windows x86_64 reserved for M10. The `Vm`, `Vcpu` and
`VcpuHandle` traits, with `VcpuExit`, `MemoryRegion` and `Error`, are the
shared contract. `kvm::KvmVm` implements it: it runs a guest, dispatches its
exits and can be forced out of the guest from another thread, but it creates
no interrupt controller and cannot boot a kernel yet. M1 adds the controller,
the vCPU register access boot needs, the HVF `CPU_ON` exit and the HVF
backend. The rules each backend keeps are in the
[VMM design](../../docs/architecture/vmm/README.md#hypervisor-backend-interface).

This crate owns host VM/vCPU handles, memory registration, interrupt
injection, and decoding host exits. It is a leaf among the BoxLite crates;
machine layout and device emulation belong to `boxlite-vmm`.

| Module | Responsibility |
| --- | --- |
| `vm` | `Vm`: memory mapping, vCPU creation, and interrupt lines |
| `vcpu` | `Vcpu` and `VcpuHandle`: running a vCPU, completing pending I/O before stop, and kicking it from another thread |
| `exit` | `VcpuExit`: decoded exits and the I/O completion contract |
| `memory` | `MemoryRegion`: host memory mapped into the guest |
| `error` | `Error`: the failed operation, its resource, and the host cause |
| `hvf` / `hvf::syndrome` | HVF operations and ARM exception decoding (not implemented yet) |
| `kvm` | `KvmVm`, `KvmVcpu` and `KvmVcpuHandle` on Linux, with no interrupt controller yet |
| `kvm::memory` | Private memory-slot allocation, reusing a slot once its region is unmapped |
| `kvm::vcpu` / `kvm::kick` | Running a vCPU and decoding its exits; forcing one out through `kvm_run.immediate_exit` and a real-time signal |
| `whp` / `whp::emulator` | WHP operations and x86 instruction decoding for memory-access exits (reserved for M10) |

Three boundaries decide placement when a case is ambiguous: KVM memory-slot
indices stay inside `kvm`, because HVF has no such concept; ARM syndrome
decoding stays inside `hvf`, so `boxlite-vmm` never sees a raw `ESR_EL2`; and
x86 instruction decoding stays inside `whp`, so `boxlite-vmm` never sees raw
instruction bytes.

The backend modules are selected by host OS and architecture, and a host with
no backend fails the build rather than producing a library that exposes no VM
operations. The traits are exported from the crate root; each backend's
concrete type gets its own path, so the KVM one is `kvm::KvmVm` and `HvfVm` and
`WhpVm` will follow.

From the repository root, run `make vmm`, `make clippy:vmm` or
`make test:unit:vmm`. See the [VMM README](../vmm/README.md) for cross-target
build commands.

The KVM tests that run guest instructions need a usable `/dev/kvm`. They print
`SKIP` and pass without one, which is what every GitHub-hosted runner is; a
host that is supposed to provide KVM sets `BOXLITE_TEST_REQUIRE_KVM=1`, which
turns the skip into a failure.
