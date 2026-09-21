// Copyright 2026 BoxLite Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tests that run real guest instructions, so they need a usable `/dev/kvm`.
//!
//! They skip on a host without one, which every GitHub-hosted runner is.
//! Setting `BOXLITE_TEST_REQUIRE_KVM` to any value turns the skip into a
//! failure, so a runner that is supposed to have KVM cannot pass by skipping.
//!
//! x86_64 only: the instructions and the real-mode boot state are x86. The arm64
//! equivalents arrive with `KVM_ARM_VCPU_INIT` and the register API.
#![cfg(target_arch = "x86_64")]

use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    io::ErrorKind,
    ptr::NonNull,
    sync::mpsc,
    thread,
    time::Duration,
};

use super::KvmVm;
use crate::{Error, Vcpu, VcpuExit, VcpuHandle, Vm};

/// Set on a host that must be able to run VMs, turning a skip into a failure.
const REQUIRE_KVM_ENV: &str = "BOXLITE_TEST_REQUIRE_KVM";

/// One page of guest RAM, which on x86_64 is always 4 KiB.
const PAGE_SIZE: usize = 4096;

/// `HLT`: halts until an interrupt. Without an in-kernel interrupt controller
/// KVM hands the halt to the VMM instead of waiting inside the kernel.
const HLT: &[u8] = &[0xf4];

/// `jmp $`: an unconditional two-byte branch to itself, so the guest never
/// leaves on its own and only a kick can end the run.
const SPIN: &[u8] = &[0xeb, 0xfe];

/// Opens a VM, or explains why this host cannot and skips.
///
/// Both a missing device and a denied one skip: the hosted runners that lack
/// `/dev/kvm` and the ones that expose it to the `kvm` group only are the same
/// situation for this suite. Any other failure is a real defect and panics.
fn vm_or_skip(test: &str) -> Option<KvmVm> {
    match KvmVm::new() {
        Ok(vm) => Some(vm),
        Err(Error::CreateVm(source))
            if matches!(
                source.kind(),
                ErrorKind::Unsupported | ErrorKind::PermissionDenied
            ) =>
        {
            assert!(
                std::env::var_os(REQUIRE_KVM_ENV).is_none(),
                "{test}: {source}; {REQUIRE_KVM_ENV} is set, so this host must provide KVM"
            );
            eprintln!("SKIP {test}: {source}");
            None
        }
        Err(other) => panic!("{test}: opening a VM failed unexpectedly: {other}"),
    }
}

/// A page of host memory to back guest RAM.
///
/// Guest memory must outlive every mapping and every vCPU that can reach it.
/// Locals drop in reverse declaration order, so each test declares its page
/// before the VM. A page moved into a closure that captures the VM does not get
/// that ordering, because captures drop after the closure's own locals; the
/// threaded test releases its mapping explicitly instead.
struct GuestPage {
    memory: NonNull<u8>,
}

impl GuestPage {
    fn with(program: &[u8]) -> Self {
        // SAFETY: the layout has a non-zero size, and the allocation is checked
        // below before it is used.
        let memory = unsafe { alloc_zeroed(Self::layout()) };
        let memory = NonNull::new(memory).expect("one page of zeroed host memory");
        // SAFETY: the program fits in the page, and nothing else refers to the
        // fresh allocation yet.
        unsafe {
            assert!(program.len() <= PAGE_SIZE, "the program must fit in a page");
            std::ptr::copy_nonoverlapping(program.as_ptr(), memory.as_ptr(), program.len());
        }
        Self { memory }
    }

    fn layout() -> Layout {
        Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).expect("a page-aligned page")
    }

    /// Describes the page as guest RAM at `guest_addr`.
    fn region(&self, guest_addr: u64) -> crate::MemoryRegion {
        crate::MemoryRegion {
            guest_addr,
            host_addr: self.memory,
            size: PAGE_SIZE,
        }
    }
}

impl Drop for GuestPage {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `alloc_zeroed` with this same layout,
        // and the VM that mapped it has been dropped by now.
        unsafe { dealloc(self.memory.as_ptr(), Self::layout()) };
    }
}

/// Points a vCPU at guest physical address zero in real mode.
///
/// A reset x86 vCPU fetches from `0xFFFFFFF0`, which a one-page VM does not
/// map. Clearing the code segment puts `CS:IP` at zero instead. `RFLAGS` bit 1
/// is architecturally always set, so a zeroed value is rejected. This is the
/// state the C probe sets (`src/boxlite/src/kvm_smoke.c`).
fn enter_at_zero(vcpu: &mut super::KvmVcpu) {
    let mut sregs = vcpu.fd.get_sregs().expect("read the segment registers");
    sregs.cs.base = 0;
    sregs.cs.selector = 0;
    vcpu.fd.set_sregs(&sregs).expect("point CS at address zero");

    let mut regs = vcpu.fd.get_regs().expect("read the general registers");
    regs.rip = 0;
    regs.rflags = 0x2;
    vcpu.fd.set_regs(&regs).expect("start execution at zero");
}

#[test]
fn a_guest_that_halts_reports_a_halt() {
    let page = GuestPage::with(HLT);
    let Some(vm) = vm_or_skip("a_guest_that_halts_reports_a_halt") else {
        return;
    };
    let region = page.region(0);

    // SAFETY: `page` is declared before the VM, so it is dropped after it,
    // and nothing else uses its memory.
    unsafe { vm.map_memory(&region) }.expect("map the guest's only page");
    let mut vcpu = vm.create_vcpu(0).expect("create the boot vCPU");
    enter_at_zero(&mut vcpu);

    let exit = vcpu.run().expect("run the guest");

    assert!(
        matches!(exit, VcpuExit::Halted),
        "a halted guest without an interrupt controller exits to the VMM, got {exit:?}"
    );

    drop(vcpu);
    vm.unmap_memory(&region).expect("release the guest's page");
}

#[test]
fn a_kick_before_the_guest_runs_interrupts_the_next_entry() {
    let page = GuestPage::with(SPIN);
    let Some(vm) = vm_or_skip("a_kick_before_the_guest_runs_interrupts_the_next_entry") else {
        return;
    };
    let region = page.region(0);

    // SAFETY: `page` is declared before the VM, so it is dropped after it,
    // and nothing else uses its memory.
    unsafe { vm.map_memory(&region) }.expect("map the guest's only page");
    let mut vcpu = vm.create_vcpu(0).expect("create the boot vCPU");
    enter_at_zero(&mut vcpu);

    vcpu.handle()
        .kick()
        .expect("kick before entering the guest");
    let exit = vcpu.run().expect("run the guest");

    assert!(
        matches!(exit, VcpuExit::Interrupted),
        "a kick that lands outside the guest must end the next entry at once, got {exit:?}"
    );
}

#[test]
fn a_kick_interrupts_a_spinning_guest() {
    let Some(vm) = vm_or_skip("a_kick_interrupts_a_spinning_guest") else {
        return;
    };
    let (handles, exits) = (mpsc::channel(), mpsc::channel());

    // The vCPU is bound to its thread, and so is the page: a guest that ignored
    // the kick would otherwise keep running over freed memory.
    let guest = thread::spawn(move || {
        let page = GuestPage::with(SPIN);
        let region = page.region(0);
        // SAFETY: the mapping is released below, before `page` is dropped at
        // the end of this closure.
        unsafe { vm.map_memory(&region) }.expect("map the guest's only page");
        let mut vcpu = vm.create_vcpu(0).expect("create the boot vCPU");
        enter_at_zero(&mut vcpu);

        handles.0.send(vcpu.handle()).expect("hand over the handle");
        let outcome = vcpu.run().map(|exit| matches!(exit, VcpuExit::Interrupted));

        // `vm` is captured by the closure, so it would otherwise drop after
        // `page`, a local, leaving KVM holding a mapping of freed memory.
        drop(vcpu);
        vm.unmap_memory(&region).expect("release the guest's page");
        drop(vm);

        exits.0.send(outcome).expect("report the exit");
    });

    let handle = handles
        .1
        .recv_timeout(Duration::from_secs(10))
        .expect("the guest thread produced a handle");
    // Widens the window so the kick usually lands inside the guest; both
    // interleavings must end the run either way.
    thread::sleep(Duration::from_millis(20));
    handle.kick().expect("kick the spinning guest");

    let interrupted = exits
        .1
        .recv_timeout(Duration::from_secs(10))
        .expect("the guest thread left the guest")
        .expect("run the guest");
    assert!(
        interrupted,
        "a kick must end a guest that never exits itself"
    );
    guest.join().expect("join the guest thread");
}

#[test]
fn completing_nothing_leaves_the_guest_untouched() {
    let page = GuestPage::with(HLT);
    let Some(vm) = vm_or_skip("completing_nothing_leaves_the_guest_untouched") else {
        return;
    };
    let region = page.region(0);

    // SAFETY: `page` is declared before the VM, so it is dropped after it,
    // and nothing else uses its memory.
    unsafe { vm.map_memory(&region) }.expect("map the guest's only page");
    let mut vcpu = vm.create_vcpu(0).expect("create the boot vCPU");
    enter_at_zero(&mut vcpu);

    vcpu.complete_pending_io()
        .expect("completing with nothing pending succeeds without entering the guest");

    let exit = vcpu.run().expect("run the guest");
    assert!(
        matches!(exit, VcpuExit::Halted),
        "the guest must still be at its first instruction, got {exit:?}"
    );
}

#[test]
fn raising_a_line_without_an_interrupt_controller_fails() {
    let Some(vm) = vm_or_skip("raising_a_line_without_an_interrupt_controller_fails") else {
        return;
    };

    let error = vm
        .set_irq_line(5, true)
        .expect_err("this step of the backend creates no interrupt controller");

    let Error::SetIrqLine { line, source } = error else {
        panic!("expected an interrupt-line failure, got {error:?}");
    };
    assert_eq!(line, 5);
    assert_eq!(
        source.raw_os_error(),
        Some(libc::ENXIO),
        "KVM reports a missing interrupt controller as ENXIO"
    );
}
