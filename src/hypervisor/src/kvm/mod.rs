// Copyright 2026 BoxLite Contributors
// SPDX-License-Identifier: Apache-2.0

//! KVM operations on Linux x86_64 and arm64.
//!
//! [`KvmVm`] implements the crate's [`Vm`] contract on top of `kvm-ioctls`,
//! which wraps each ioctl at fixed arity. That avoids the variadic-FFI trouble
//! that forced BoxLite's existing probe (`src/boxlite/src/kvm_smoke.c`) to be
//! written in C.
//!
//! **This is the first step of the backend, and it creates no interrupt
//! controller yet.** The design has the constructor create one
//! (`docs/architecture/vmm/README.md`), but an in-kernel irqchip handles `HLT`
//! inside KVM, so a guest could not be observed halting until devices exist to
//! exit for. Booting a kernel needs the controller, and the change that adds it
//! replaces the halt test with exits that survive it.

mod kick;
mod memory;
mod vcpu;

#[cfg(test)]
mod tests;

use std::{
    ffi::CStr,
    io::{self, ErrorKind},
    sync::Mutex,
};

use kvm_ioctls::{Kvm, VmFd};

use self::memory::SlotAllocator;
use crate::{Error, MemoryRegion, Result, Vm};

pub use self::vcpu::{KvmVcpu, KvmVcpuHandle};

/// The KVM character device every Linux host exposes when KVM is available.
const KVM_DEVICE: &CStr = c"/dev/kvm";

/// A VM on the host's KVM.
#[derive(Debug)]
pub struct KvmVm {
    fd: VmFd,
    /// Slot numbers are backend state that `map_memory(&self)` mutates, so the
    /// allocator carries its own lock rather than making the trait take `&mut`.
    slots: Mutex<SlotAllocator>,
}

impl KvmVm {
    /// Creates an empty VM on the host's KVM.
    ///
    /// The VM has no interrupt controller and no vCPUs. See the module docs for
    /// why the controller is not created here yet.
    pub fn new() -> Result<Self> {
        Self::open_at(KVM_DEVICE)
    }

    /// Creates an empty VM through the KVM device at `path`.
    ///
    /// Taking the path makes both failure branches reachable in tests on a host
    /// that has no KVM at all.
    fn open_at(path: &CStr) -> Result<Self> {
        let kvm = Kvm::new_with_path(path)
            .map_err(|source| Error::CreateVm(classify_host_failure(source.into())))?;
        let fd = kvm
            .create_vm()
            .map_err(|source| Error::CreateVm(classify_host_failure(source.into())))?;
        let max_slots = u32::try_from(kvm.get_nr_memslots()).unwrap_or(u32::MAX);
        Ok(Self {
            fd,
            slots: Mutex::new(SlotAllocator::new(max_slots)),
        })
    }
}

impl Vm for KvmVm {
    type Vcpu = KvmVcpu;

    unsafe fn map_memory(&self, region: &MemoryRegion) -> Result<()> {
        memory::map_region(&self.slots, region, |mapping| {
            // SAFETY: the caller of `map_memory` guarantees the host range
            // stays mapped and backs nothing else for as long as the guest
            // mapping lives; `mapping` describes exactly that range.
            unsafe { self.fd.set_user_memory_region(mapping) }.map_err(io::Error::from)
        })
    }

    fn unmap_memory(&self, region: &MemoryRegion) -> Result<()> {
        memory::unmap_region(&self.slots, region, |mapping| {
            // SAFETY: the mapping has size zero, which removes a slot rather
            // than registering host memory, so no aliasing obligation applies.
            unsafe { self.fd.set_user_memory_region(mapping) }.map_err(io::Error::from)
        })
    }

    fn create_vcpu(&self, id: u32) -> Result<Self::Vcpu> {
        let fd = self
            .fd
            .create_vcpu(id.into())
            .map_err(|source| Error::CreateVcpu {
                id,
                source: source.into(),
            })?;
        KvmVcpu::new(id, fd)
    }

    fn set_irq_line(&self, line: u32, level: bool) -> Result<()> {
        let encoded = encode_irq_line(line).map_err(|source| Error::SetIrqLine { line, source })?;
        self.fd
            .set_irq_line(encoded, level)
            .map_err(|source| Error::SetIrqLine {
                line,
                source: source.into(),
            })
    }
}

/// Encodes an interrupt line the way `KVM_IRQ_LINE` expects it.
///
/// On x86_64 the line is the GSI itself. On arm64 KVM packs an interrupt type
/// into the top byte, so the bare SPI INTID the trait takes would address a
/// CPU interrupt on vCPU 0 instead of the device's line.
#[cfg(target_arch = "x86_64")]
fn encode_irq_line(line: u32) -> io::Result<u32> {
    Ok(line)
}

#[cfg(target_arch = "aarch64")]
fn encode_irq_line(line: u32) -> io::Result<u32> {
    use kvm_bindings::{KVM_ARM_IRQ_NUM_MASK, KVM_ARM_IRQ_TYPE_SHIFT, KVM_ARM_IRQ_TYPE_SPI};

    // An INTID wider than the number field would overflow into the vCPU index
    // and the type, quietly targeting a different interrupt.
    if line > KVM_ARM_IRQ_NUM_MASK {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!("interrupt {line} is beyond the largest GIC INTID KVM accepts"),
        ));
    }
    Ok((KVM_ARM_IRQ_TYPE_SPI << KVM_ARM_IRQ_TYPE_SHIFT) | line)
}

/// Marks the errno values that mean "this host cannot run VMs".
///
/// The crate's error contract asks for [`ErrorKind::Unsupported`] on a host
/// without working virtualization and [`ErrorKind::PermissionDenied`] when
/// access is missing, which is the distinction the runtime already reports to
/// users (`src/boxlite/src/system_check.rs`). `io::Error` already decodes
/// `EACCES` and `EPERM`; the rest need naming.
fn classify_host_failure(source: io::Error) -> io::Error {
    match source.raw_os_error() {
        // No /dev/kvm, no such device, or a device that does not speak KVM.
        Some(libc::ENOENT | libc::ENXIO | libc::ENODEV | libc::ENOTTY) => {
            io::Error::new(ErrorKind::Unsupported, source)
        }
        _ => source,
    }
}

#[cfg(test)]
mod host_failure_tests {
    use std::io::{self, ErrorKind};

    use super::{KvmVm, classify_host_failure, encode_irq_line};
    use crate::Error;

    #[test]
    fn a_host_without_kvm_is_unsupported() {
        let error = KvmVm::open_at(c"/nonexistent/kvm").expect_err("there is no device there");
        let Error::CreateVm(source) = error else {
            panic!("expected a VM creation failure, got {error:?}");
        };

        assert_eq!(
            source.kind(),
            ErrorKind::Unsupported,
            "the VMM maps an unsupported host to a distinct client error"
        );
        let cause = source
            .into_inner()
            .expect("the host errno stays reachable")
            .downcast::<io::Error>()
            .expect("the cause is an OS error");
        assert_eq!(cause.raw_os_error(), Some(libc::ENOENT));
    }

    #[test]
    fn a_device_that_does_not_speak_kvm_is_unsupported() {
        let error = KvmVm::open_at(c"/dev/null").expect_err("/dev/null answers no KVM ioctl");
        let Error::CreateVm(source) = error else {
            panic!("expected a VM creation failure, got {error:?}");
        };
        assert_eq!(source.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn host_failures_classify_by_errno() {
        let unsupported = [libc::ENOENT, libc::ENXIO, libc::ENODEV, libc::ENOTTY];
        for errno in unsupported {
            let classified = classify_host_failure(io::Error::from_raw_os_error(errno));
            assert_eq!(
                classified.kind(),
                ErrorKind::Unsupported,
                "errno {errno} means this host cannot run VMs"
            );
        }

        let denied = classify_host_failure(io::Error::from_raw_os_error(libc::EACCES));
        assert_eq!(denied.kind(), ErrorKind::PermissionDenied);
        assert_eq!(
            denied.raw_os_error(),
            Some(libc::EACCES),
            "a permission failure keeps its errno for the operator log"
        );

        let other = classify_host_failure(io::Error::from_raw_os_error(libc::ENOMEM));
        assert_eq!(other.raw_os_error(), Some(libc::ENOMEM));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn an_interrupt_line_is_the_gsi_itself() {
        assert_eq!(
            encode_irq_line(5).expect("a GSI needs no encoding"),
            5,
            "x86 takes the GSI as it stands"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn an_interrupt_line_carries_its_gic_type() {
        // The first SPI: KVM reads the top byte as the interrupt type, so a
        // bare 32 would raise a CPU interrupt on vCPU 0 instead.
        assert_eq!(
            encode_irq_line(32).expect("the first SPI is in range"),
            0x0100_0020,
            "an SPI INTID must carry KVM_ARM_IRQ_TYPE_SPI"
        );

        let error = encode_irq_line(kvm_bindings::KVM_ARM_IRQ_NUM_MASK + 1)
            .expect_err("an INTID that wide would overflow into the type field");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}
