// Copyright 2026 BoxLite Contributors
// SPDX-License-Identifier: Apache-2.0

//! Running a KVM vCPU and decoding why it left the guest.

use std::{io, marker::PhantomData, sync::Arc};

use kvm_bindings::{KVM_SYSTEM_EVENT_RESET, KVM_SYSTEM_EVENT_SHUTDOWN};
use kvm_ioctls::{VcpuExit as KvmExit, VcpuFd};

use super::kick::{ImmediateExit, KickState};
use crate::{Error, Result, Vcpu, VcpuExit, VcpuHandle};

/// What `KVM_RUN` reports: an exit, or the errno the ioctl failed with.
type KvmOutcome<'a> = std::result::Result<KvmExit<'a>, kvm_ioctls::Error>;

/// A KVM vCPU, bound to the thread that created it.
///
/// `VcpuFd` is `Send`, but the trait requires backend vCPUs not to be: HVF
/// accepts vCPU calls only from the creating thread, and this type records that
/// thread so another can kick it. `PhantomData<*const ()>` is what makes the
/// compiler enforce the binding on Linux too.
#[derive(Debug)]
pub struct KvmVcpu {
    id: u32,
    /// Readable by the smoke tests, which set boot registers directly until the
    /// register API arrives with kernel boot.
    pub(super) fd: VcpuFd,
    flag: ImmediateExit,
    kick: Arc<KickState>,
    _thread_bound: PhantomData<*const ()>,
}

impl KvmVcpu {
    /// Wraps a freshly created vCPU, arming kicks on the calling thread.
    pub(super) fn new(id: u32, mut fd: VcpuFd) -> Result<Self> {
        let flag = ImmediateExit::new(&mut fd.get_kvm_run().immediate_exit);
        let kick = KickState::register(flag).map_err(|source| Error::CreateVcpu { id, source })?;
        Ok(Self {
            id,
            fd,
            flag,
            kick: Arc::new(kick),
            _thread_bound: PhantomData,
        })
    }
}

impl Vcpu for KvmVcpu {
    type Handle = KvmVcpuHandle;

    fn run(&mut self) -> Result<VcpuExit<'_>> {
        // The kick flag is never cleared before entry: a kick that arrived
        // between two runs must still be reported by this one.
        let outcome = self.fd.run();
        if is_interrupt(&outcome) {
            self.flag.acknowledge_kick();
        }
        decode_exit(self.id, outcome)
    }

    fn complete_pending_io(&mut self) -> Result<()> {
        // KVM finishes a pending access before it tests `immediate_exit`, so
        // this entry completes the access without running a guest instruction.
        // With nothing pending it simply returns.
        self.flag.begin_completion();
        let outcome = self.fd.run();
        self.flag.end_completion();
        completion_result(self.id, outcome)
    }

    fn handle(&self) -> Self::Handle {
        KvmVcpuHandle {
            id: self.id,
            kick: Arc::clone(&self.kick),
        }
    }
}

impl Drop for KvmVcpu {
    fn drop(&mut self) {
        // Before `fd` unmaps `kvm_run`, so no handle can reach the freed byte.
        self.kick.deregister();
    }
}

/// Forces one KVM vCPU out of the guest, from any thread.
#[derive(Debug, Clone)]
pub struct KvmVcpuHandle {
    id: u32,
    kick: Arc<KickState>,
}

impl VcpuHandle for KvmVcpuHandle {
    fn kick(&self) -> Result<()> {
        self.kick.kick().map_err(|source| Error::KickVcpu {
            id: self.id,
            source,
        })
    }
}

/// Reports whether the guest was left because something interrupted it.
///
/// KVM reports an interruption two ways: as `EINTR` from the ioctl, and as a
/// decoded `Intr` exit. libkrun handles only the first
/// (`libkrun/src/vmm/src/linux/vstate.rs:1551-1556`), which leaves the second
/// to its catch-all; both mean the same thing here.
fn is_interrupt(outcome: &KvmOutcome<'_>) -> bool {
    match outcome {
        Ok(KvmExit::Intr) => true,
        Err(error) => error.errno() == libc::EINTR,
        Ok(_) => false,
    }
}

/// Translates one KVM outcome into the VMM's exit contract.
fn decode_exit<'a>(id: u32, outcome: KvmOutcome<'a>) -> Result<VcpuExit<'a>> {
    let exit = match outcome {
        Ok(exit) => exit,
        Err(error) if error.errno() == libc::EINTR => return Ok(VcpuExit::Interrupted),
        // KVM asks for a plain re-entry. Reporting it as an interruption sends
        // the VMM through its control-request check and back into the guest,
        // which is the same retry without a hidden loop here.
        Err(error) if error.errno() == libc::EAGAIN => return Ok(VcpuExit::Interrupted),
        Err(error) => {
            return Err(Error::RunVcpu {
                id,
                source: error.into(),
            });
        }
    };

    let unhandled = |reason: String| Err(Error::UnhandledExit { id, reason });

    match exit {
        KvmExit::MmioRead(guest_addr, bytes) => Ok(VcpuExit::MmioRead { guest_addr, bytes }),
        KvmExit::MmioWrite(guest_addr, bytes) => Ok(VcpuExit::MmioWrite { guest_addr, bytes }),
        #[cfg(target_arch = "x86_64")]
        KvmExit::IoIn(port, bytes) => match access_width(bytes.len()) {
            Ok(()) => Ok(VcpuExit::IoIn { port, bytes }),
            Err(reason) => unhandled(format!("{reason} on port {port:#x}")),
        },
        #[cfg(target_arch = "x86_64")]
        KvmExit::IoOut(port, bytes) => match access_width(bytes.len()) {
            Ok(()) => Ok(VcpuExit::IoOut { port, bytes }),
            Err(reason) => unhandled(format!("{reason} on port {port:#x}")),
        },
        #[cfg(not(target_arch = "x86_64"))]
        KvmExit::IoIn(port, _) | KvmExit::IoOut(port, _) => {
            unhandled(format!("port I/O on port {port:#x} on a non-x86 host"))
        }
        KvmExit::Intr => Ok(VcpuExit::Interrupted),
        KvmExit::Hlt => Ok(VcpuExit::Halted),
        // On x86 a triple fault surfaces as KVM_EXIT_SHUTDOWN, and the guest
        // has reset itself; other hosts report a reset as a system event.
        KvmExit::Shutdown => Ok(VcpuExit::Reset),
        KvmExit::SystemEvent(KVM_SYSTEM_EVENT_SHUTDOWN, _) => Ok(VcpuExit::Shutdown),
        KvmExit::SystemEvent(KVM_SYSTEM_EVENT_RESET, _) => Ok(VcpuExit::Reset),
        KvmExit::SystemEvent(event, _) => unhandled(format!("system event {event}")),
        KvmExit::FailEntry(reason, cpu) => unhandled(format!(
            "hardware entry failure {reason:#x} on host CPU {cpu}"
        )),
        // kvm-ioctls does not surface the suberror that names the fault.
        KvmExit::InternalError => unhandled("KVM internal error".to_owned()),
        KvmExit::Unsupported(reason) => unhandled(format!("unsupported KVM exit reason {reason}")),
        other => unhandled(format!("unexpected KVM exit {other:?}")),
    }
}

/// Accepts the access widths a single port instruction can produce.
///
/// kvm-ioctls hands over `count * size` bytes without the count, so a string
/// instruction is only visible as an unusual width. That is enough here: the
/// exit contract makes string port I/O unhandled, and BoxLite's guests drive
/// the 8250, RTC and i8042 with single accesses. A `rep outsw` of exactly two
/// words is indistinguishable from one 4-byte access and would be dispatched as
/// one; closing that gap needs `kvm_run.io.count`, which kvm-ioctls hides.
#[cfg(target_arch = "x86_64")]
fn access_width(len: usize) -> std::result::Result<(), String> {
    match len {
        1 | 2 | 4 => Ok(()),
        other => Err(format!("string port I/O of {other} bytes")),
    }
}

/// Checks that a completion entry finished without running the guest.
fn completion_result(id: u32, outcome: KvmOutcome<'_>) -> Result<()> {
    match outcome {
        Ok(KvmExit::Intr) => Ok(()),
        Err(error) if error.errno() == libc::EINTR => Ok(()),
        // KVM tests `immediate_exit` before it enters the guest, so any other
        // exit means a guest instruction ran and the caller's state is stale.
        Ok(exit) => Err(Error::CompletePendingIo {
            id,
            source: io::Error::other(format!("the guest ran and exited with {exit:?}")),
        }),
        Err(error) => Err(Error::CompletePendingIo {
            id,
            source: error.into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use kvm_bindings::{KVM_SYSTEM_EVENT_CRASH, KVM_SYSTEM_EVENT_RESET, KVM_SYSTEM_EVENT_SHUTDOWN};
    use kvm_ioctls::VcpuExit as KvmExit;
    use vmm_sys_util::errno;

    use super::{KvmOutcome, KvmVcpu, completion_result, decode_exit, is_interrupt};
    use crate::{Error, VcpuExit};

    static_assertions::assert_not_impl_any!(KvmVcpu: Send);
    static_assertions::assert_impl_all!(super::KvmVcpuHandle: Send, Sync, Clone);

    const VCPU: u32 = 7;

    fn failing(errno: i32) -> KvmOutcome<'static> {
        Err(errno::Error::new(errno))
    }

    #[test]
    fn guest_exits_decode_to_the_vmm_contract() {
        let mut mmio_read = [0u8; 4];
        let mmio_write = [1u8, 2, 3, 4];

        assert!(matches!(
            decode_exit(VCPU, Ok(KvmExit::MmioRead(0x1000, &mut mmio_read))),
            Ok(VcpuExit::MmioRead {
                guest_addr: 0x1000,
                bytes
            }) if bytes.len() == 4
        ));
        assert!(matches!(
            decode_exit(VCPU, Ok(KvmExit::MmioWrite(0x2000, &mmio_write))),
            Ok(VcpuExit::MmioWrite {
                guest_addr: 0x2000,
                bytes
            }) if bytes == [1, 2, 3, 4]
        ));
        assert!(matches!(
            decode_exit(VCPU, Ok(KvmExit::Hlt)),
            Ok(VcpuExit::Halted)
        ));
        assert!(
            matches!(
                decode_exit(VCPU, Ok(KvmExit::Shutdown)),
                Ok(VcpuExit::Reset)
            ),
            "an x86 triple fault is a guest reset"
        );
        assert!(matches!(
            decode_exit(
                VCPU,
                Ok(KvmExit::SystemEvent(KVM_SYSTEM_EVENT_SHUTDOWN, &[]))
            ),
            Ok(VcpuExit::Shutdown)
        ));
        assert!(matches!(
            decode_exit(VCPU, Ok(KvmExit::SystemEvent(KVM_SYSTEM_EVENT_RESET, &[]))),
            Ok(VcpuExit::Reset)
        ));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn single_port_accesses_decode_and_string_accesses_do_not() {
        let mut inbound = [0u8; 2];
        assert!(matches!(
            decode_exit(VCPU, Ok(KvmExit::IoIn(0x3f8, &mut inbound))),
            Ok(VcpuExit::IoIn { port: 0x3f8, bytes }) if bytes.len() == 2
        ));

        let outbound = *b"x";
        assert!(matches!(
            decode_exit(VCPU, Ok(KvmExit::IoOut(0x3f8, &outbound))),
            Ok(VcpuExit::IoOut { port: 0x3f8, bytes }) if bytes == b"x"
        ));

        let string_access = [0u8; 8];
        let error = decode_exit(VCPU, Ok(KvmExit::IoOut(0x3f8, &string_access)))
            .expect_err("a string access is not one access");
        let Error::UnhandledExit { id, reason } = error else {
            panic!("expected an unhandled exit");
        };
        assert_eq!(id, VCPU);
        assert!(
            reason.contains("string port I/O of 8 bytes") && reason.contains("0x3f8"),
            "the reason names the width and the port: {reason}"
        );
    }

    #[test]
    fn interruptions_are_reported_without_an_error() {
        for outcome in [Ok(KvmExit::Intr), failing(libc::EINTR)] {
            assert!(is_interrupt(&outcome), "the kick must be acknowledged");
            assert!(matches!(
                decode_exit(VCPU, outcome),
                Ok(VcpuExit::Interrupted)
            ));
        }

        let retry = failing(libc::EAGAIN);
        assert!(
            !is_interrupt(&retry),
            "a bare retry acknowledges no kick, or a pending kick would be lost"
        );
        assert!(matches!(
            decode_exit(VCPU, retry),
            Ok(VcpuExit::Interrupted)
        ));
    }

    #[test]
    fn unexplained_exits_carry_a_diagnostic_reason() {
        let crashed = format!("system event {KVM_SYSTEM_EVENT_CRASH}");
        let cases: [(KvmOutcome<'static>, &str); 4] = [
            (
                Ok(KvmExit::FailEntry(0x8, 3)),
                "hardware entry failure 0x8 on host CPU 3",
            ),
            (Ok(KvmExit::InternalError), "KVM internal error"),
            (
                Ok(KvmExit::Unsupported(42)),
                "unsupported KVM exit reason 42",
            ),
            (
                Ok(KvmExit::SystemEvent(KVM_SYSTEM_EVENT_CRASH, &[])),
                &crashed,
            ),
        ];

        for (outcome, expected) in cases {
            let error = decode_exit(VCPU, outcome).expect_err("the backend cannot handle this");
            let Error::UnhandledExit { id, reason } = error else {
                panic!("expected an unhandled exit, got {error:?}");
            };
            assert_eq!(id, VCPU);
            assert_eq!(reason, expected);
        }

        let error = decode_exit(VCPU, Ok(KvmExit::IrqWindowOpen))
            .expect_err("an exit the backend never asked for");
        let Error::UnhandledExit { reason, .. } = error else {
            panic!("expected an unhandled exit");
        };
        assert!(
            reason.starts_with("unexpected KVM exit"),
            "the catch-all names the exit: {reason}"
        );
    }

    #[test]
    fn a_failed_entry_keeps_the_host_errno() {
        let error = decode_exit(VCPU, failing(libc::EFAULT)).expect_err("the ioctl failed");
        let Error::RunVcpu { id, source } = error else {
            panic!("expected a run failure, got {error:?}");
        };
        assert_eq!(id, VCPU);
        assert_eq!(
            source.raw_os_error(),
            Some(libc::EFAULT),
            "the VMM maps unsupported and permission failures by errno"
        );
    }

    #[test]
    fn completion_accepts_only_an_entry_that_ran_no_instruction() {
        for outcome in [Ok(KvmExit::Intr), failing(libc::EINTR)] {
            completion_result(VCPU, outcome).expect("the access completed without the guest");
        }

        let error = completion_result(VCPU, Ok(KvmExit::Hlt))
            .expect_err("a guest exit means an instruction ran");
        let Error::CompletePendingIo { id, source } = error else {
            panic!("expected a completion failure, got {error:?}");
        };
        assert_eq!(id, VCPU);
        assert!(
            source.to_string().contains("Hlt"),
            "names the exit: {source}"
        );

        let error =
            completion_result(VCPU, failing(libc::EFAULT)).expect_err("the ioctl failed outright");
        let Error::CompletePendingIo { source, .. } = error else {
            panic!("expected a completion failure");
        };
        assert_eq!(source.raw_os_error(), Some(libc::EFAULT));
    }
}
