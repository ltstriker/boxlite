// Copyright 2026 BoxLite Contributors
// SPDX-License-Identifier: Apache-2.0

//! Forcing a running vCPU back out to the VMM, from any thread.
//!
//! KVM leaves the guest when `kvm_run.immediate_exit` is non-zero at entry, and
//! `KVM_RUN` fails with `EINTR` when a signal arrives while the vCPU is inside
//! the guest. Neither alone is enough: the flag does not disturb a vCPU that is
//! already running, and a signal that lands just before entry is lost. The
//! kicking thread therefore sets the flag and then signals, as Firecracker
//! does ([`vstate/vcpu.rs:391-403`][firecracker-kick]).
//!
//! libkrun arrives at the same place from the other side: it only signals, and
//! its handler sets the flag on the vCPU thread through a thread-local pointer
//! to the vCPU (`libkrun/src/vmm/src/linux/vstate.rs:1023-1037`). Setting it
//! from the kicking thread needs no such pointer, and the flag is an atomic
//! whose release in [`KickState::kick`] pairs with the acquire in
//! [`ImmediateExit::acknowledge_kick`], so the handler here does nothing at
//! all.
//!
//! [firecracker-kick]: https://github.com/firecracker-microvm/firecracker/blob/68698adfee9b252df130b7a98e3ba04eb81f0f54/src/vmm/src/vstate/vcpu.rs#L391-L403

use std::{
    io,
    ptr::NonNull,
    sync::{Mutex, OnceLock, atomic::AtomicU8, atomic::Ordering},
};

use libc::{c_int, pthread_t};

/// Offset from `SIGRTMIN` for the signal that interrupts `KVM_RUN`.
///
/// One, not zero: libkrun kicks its own vCPUs with `SIGRTMIN` and is linked
/// into the same shim process until the native VMM becomes the default
/// (`libkrun/src/vmm/src/linux/vstate.rs:75`). Two handlers on one signal would
/// fight over the same `sigaction`.
const KICK_SIGNAL_OFFSET: c_int = 1;

/// `immediate_exit` bit claimed by [`KickState::kick`].
const KICK: u8 = 0b01;

/// `immediate_exit` bit claimed by a pending-I/O completion.
///
/// A separate bit from [`KICK`] so that completing an access and kicking cannot
/// erase each other: KVM only tests the byte for non-zero, so both can be set.
const COMPLETION: u8 = 0b10;

/// The `immediate_exit` byte inside a vCPU's `kvm_run` mapping.
///
/// Sending it across threads is what makes a kick possible at all. Every access
/// goes through an atomic, because the vCPU thread reads and writes the same
/// byte, and kvm-ioctls holds a `&mut kvm_run` while it decodes an exit.
#[derive(Debug, Clone, Copy)]
pub(super) struct ImmediateExit(NonNull<u8>);

// SAFETY: the byte lives in the vCPU's `kvm_run` mapping, which outlives every
// `ImmediateExit` that can reach it: `KvmVcpu::drop` clears the pointer out of
// the shared `KickState` under its mutex before the mapping is unmapped, and
// `KickState` hands the pointer out only while holding that mutex. Accesses are
// atomic, so concurrent reads and writes are defined.
unsafe impl Send for ImmediateExit {}

impl ImmediateExit {
    /// Borrows the `immediate_exit` byte of a live `kvm_run` mapping.
    pub(super) fn new(flag: &mut u8) -> Self {
        Self(NonNull::from(flag))
    }

    /// Reads the byte as KVM would at guest entry.
    #[cfg(test)]
    fn requested(&self) -> u8 {
        self.atomic().load(Ordering::Relaxed)
    }

    /// Returns the byte as an atomic.
    fn atomic(&self) -> &AtomicU8 {
        // SAFETY: the pointer comes from a live `&mut u8` in the `kvm_run`
        // mapping (see the `Send` justification above), and `AtomicU8` has the
        // same layout and alignment as `u8`.
        unsafe { AtomicU8::from_ptr(self.0.as_ptr()) }
    }

    /// Requests an exit for `bits`, leaving the other bits alone.
    fn set(&self, bits: u8, order: Ordering) {
        self.atomic().fetch_or(bits, order);
    }

    /// Withdraws the request for `bits`.
    fn clear(&self, bits: u8, order: Ordering) {
        self.atomic().fetch_and(!bits, order);
    }

    /// Acknowledges a kick once the vCPU has left the guest because of it.
    ///
    /// `Acquire` pairs with the `Release` in [`KickState::kick`], so whatever
    /// the kicking thread wrote before kicking, typically a stop request, is
    /// visible to the vCPU thread by the time `run` reports the interruption.
    pub(super) fn acknowledge_kick(&self) {
        self.clear(KICK, Ordering::Acquire);
    }

    /// Asks KVM to leave the guest without executing another instruction.
    pub(super) fn begin_completion(&self) {
        self.set(COMPLETION, Ordering::Relaxed);
    }

    /// Withdraws the completion request made by [`begin_completion`].
    ///
    /// [`begin_completion`]: Self::begin_completion
    pub(super) fn end_completion(&self) {
        self.clear(COMPLETION, Ordering::Relaxed);
    }
}

/// What a kicking thread needs to reach a live vCPU.
#[derive(Debug)]
struct KickTarget {
    flag: ImmediateExit,
    thread: pthread_t,
}

/// The rendezvous between a vCPU and the handles that can kick it.
///
/// The target is `None` once the vCPU is dropped, which is what makes kicking a
/// dropped vCPU a no-op rather than a use-after-free or a signal to a thread id
/// the host has since reused.
#[derive(Debug)]
pub(super) struct KickState {
    /// Held across the whole kick, so a vCPU being dropped cannot unmap
    /// `kvm_run` between the pointer being read and the byte being written.
    target: Mutex<Option<KickTarget>>,
}

impl KickState {
    /// Arms kicks for the calling thread's vCPU.
    ///
    /// Must run on the thread that will call `KVM_RUN`: it records that thread
    /// as the signal's destination and unblocks the signal there.
    pub(super) fn register(flag: ImmediateExit) -> io::Result<Self> {
        install_kick_handler()?;
        unblock_kick_signal()?;
        // SAFETY: `pthread_self` is always safe to call.
        let thread = unsafe { libc::pthread_self() };
        Ok(Self {
            target: Mutex::new(Some(KickTarget { flag, thread })),
        })
    }

    /// Stops kicks from reaching this vCPU, before its `kvm_run` is unmapped.
    pub(super) fn deregister(&self) {
        *self.target.lock().unwrap_or_else(|poisoned| {
            // A kick panicking would leave no state to repair: the target is
            // either present or not.
            self.target.clear_poison();
            poisoned.into_inner()
        }) = None;
    }

    /// Forces the vCPU out of the guest, from any thread.
    pub(super) fn kick(&self) -> io::Result<()> {
        let target = self.target.lock().unwrap_or_else(|poisoned| {
            self.target.clear_poison();
            poisoned.into_inner()
        });
        let Some(target) = target.as_ref() else {
            return Ok(());
        };

        // Order matters: a vCPU that is about to enter the guest must see the
        // flag, and one already inside it must get the signal.
        target.flag.set(KICK, Ordering::Release);
        // SAFETY: the target is present, so `KvmVcpu::drop` has not run and the
        // vCPU thread's id is still valid.
        let failure = unsafe { libc::pthread_kill(target.thread, kick_signal()) };
        if failure != 0 {
            // pthread_kill returns the error number instead of setting errno.
            return Err(io::Error::from_raw_os_error(failure));
        }
        Ok(())
    }
}

/// The signal that interrupts `KVM_RUN`.
fn kick_signal() -> c_int {
    vmm_sys_util::signal::SIGRTMIN() + KICK_SIGNAL_OFFSET
}

/// Installs the kick handler once per process.
///
/// The handler does nothing. Interrupting the `KVM_RUN` ioctl is the whole
/// point, and the flag that says why has already been set by the kicker.
fn install_kick_handler() -> io::Result<()> {
    static INSTALLED: OnceLock<Result<(), i32>> = OnceLock::new();

    extern "C" fn ignore_kick(
        _signal: c_int,
        _info: *mut libc::siginfo_t,
        _context: *mut libc::c_void,
    ) {
    }

    let outcome = INSTALLED.get_or_init(|| {
        // The handler touches nothing, so it is async-signal-safe.
        vmm_sys_util::signal::register_signal_handler(kick_signal(), ignore_kick)
            .map_err(|error| error.errno())
    });
    outcome.map_err(io::Error::from_raw_os_error)
}

/// Unblocks the kick signal on the calling thread.
///
/// Signal masks are inherited, and BoxLite runs inside host processes that may
/// block real-time signals. A blocked kick never interrupts `KVM_RUN`, so every
/// stop would hang.
fn unblock_kick_signal() -> io::Result<()> {
    // SAFETY: the set is initialised by `sigemptyset` before use, and
    // `pthread_sigmask` only reads it.
    unsafe {
        let mut mask = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
        if libc::sigemptyset(mask.as_mut_ptr()) != 0 {
            return Err(io::Error::last_os_error());
        }
        let mut mask = mask.assume_init();
        if libc::sigaddset(&mut mask, kick_signal()) != 0 {
            return Err(io::Error::last_os_error());
        }
        let failure = libc::pthread_sigmask(libc::SIG_UNBLOCK, &mask, std::ptr::null_mut());
        if failure != 0 {
            // pthread_sigmask returns the error number instead of setting errno.
            return Err(io::Error::from_raw_os_error(failure));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::{COMPLETION, ImmediateExit, KICK, KickState, kick_signal};

    /// A standalone byte standing in for the `kvm_run` field, owned for as long
    /// as the flag that points at it. Reads go through the flag, exactly as
    /// production code does, so no reference aliases that pointer.
    struct FakeRun {
        byte: Box<u8>,
    }

    impl FakeRun {
        fn new() -> Self {
            Self { byte: Box::new(0) }
        }

        fn flag(&mut self) -> ImmediateExit {
            ImmediateExit::new(&mut self.byte)
        }
    }

    #[test]
    fn a_kick_and_a_completion_do_not_erase_each_other() {
        let mut run = FakeRun::new();
        let flag = run.flag();

        flag.begin_completion();
        flag.set(KICK, Ordering::Release);
        flag.end_completion();

        assert_eq!(
            flag.requested(),
            KICK,
            "ending a completion must leave a concurrent kick pending"
        );

        flag.begin_completion();
        flag.acknowledge_kick();

        assert_eq!(
            flag.requested(),
            COMPLETION,
            "acknowledging a kick must leave a concurrent completion pending"
        );
    }

    #[test]
    fn acknowledging_a_kick_clears_the_request() {
        let mut run = FakeRun::new();
        let flag = run.flag();

        flag.set(KICK, Ordering::Release);
        assert_ne!(
            flag.requested(),
            0,
            "KVM leaves the guest on any non-zero byte"
        );

        flag.acknowledge_kick();
        assert_eq!(
            flag.requested(),
            0,
            "an acknowledged kick must not exit twice"
        );
    }

    #[test]
    fn kicking_a_registered_vcpu_requests_an_exit() {
        let mut run = FakeRun::new();
        let flag = run.flag();
        let state = KickState::register(flag).expect("arm kicks on this thread");

        state.kick().expect("kick the calling thread");

        assert_eq!(
            flag.requested(),
            KICK,
            "a kick must request an exit before signalling"
        );
    }

    #[test]
    fn kicking_a_dropped_vcpu_does_nothing() {
        let mut run = FakeRun::new();
        let flag = run.flag();
        let state = KickState::register(flag).expect("arm kicks on this thread");

        state.deregister();
        state
            .kick()
            .expect("kicking a dropped vCPU is not an error");

        assert_eq!(flag.requested(), 0, "a dropped vCPU must not be signalled");
    }

    #[test]
    fn the_kick_signal_is_a_free_real_time_signal() {
        let signal = kick_signal();

        assert!(
            signal > vmm_sys_util::signal::SIGRTMIN(),
            "SIGRTMIN itself belongs to libkrun while both VMMs share a process"
        );
        assert!(
            signal <= vmm_sys_util::signal::SIGRTMAX(),
            "the kick signal must be a real-time signal"
        );
    }
}
