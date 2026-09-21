// Copyright 2026 BoxLite Contributors
// SPDX-License-Identifier: Apache-2.0

//! KVM memory-slot allocation, private to this backend.
//!
//! KVM identifies each guest memory mapping by a small integer slot, and the
//! VMM's [`MemoryRegion`] has no such field: the backend
//! picks the slot, because HVF has no equivalent concept.
//!
//! The ioctl itself is passed in as a closure, so every decision here, which
//! slot a region gets, what happens when one is missing, and what is rolled
//! back when the ioctl fails, is testable on a host without `/dev/kvm`.

use std::{
    collections::HashMap,
    io::{self, ErrorKind},
    sync::Mutex,
};

use kvm_bindings::kvm_userspace_memory_region;

use crate::{Error, MemoryRegion, Result};

/// A mapping KVM currently holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mapped {
    slot: u32,
    size: usize,
    host_addr: u64,
}

/// Slot numbers for a VM's guest memory mappings.
///
/// Slots are reused after an unmap: hot-plugged memory will map and unmap over
/// a VM's lifetime, so a counter that only ever grows, as libkrun uses
/// (`libkrun/src/vmm/src/linux/vstate.rs:491`), would exhaust the host's slots.
#[derive(Debug)]
pub(super) struct SlotAllocator {
    /// What the host allows, from `KVM_CAP_NR_MEMSLOTS`.
    max: u32,
    /// Lowest slot never handed out.
    next: u32,
    /// Slots returned by an unmap, reused before `next` grows.
    free: Vec<u32>,
    /// Live mappings, keyed by the guest address that identifies them.
    mapped: HashMap<u64, Mapped>,
}

impl SlotAllocator {
    pub(super) fn new(max: u32) -> Self {
        Self {
            max,
            next: 0,
            free: Vec::new(),
            mapped: HashMap::new(),
        }
    }

    /// Claims a slot for `region`, before the ioctl that would use it.
    fn reserve(&mut self, region: &MemoryRegion) -> io::Result<u32> {
        if self.mapped.contains_key(&region.guest_addr) {
            return Err(io::Error::new(
                ErrorKind::AlreadyExists,
                format!("guest address {:#x} is already mapped", region.guest_addr),
            ));
        }
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None if self.next < self.max => {
                let slot = self.next;
                self.next += 1;
                slot
            }
            None => {
                return Err(io::Error::new(
                    ErrorKind::QuotaExceeded,
                    format!("the host allows only {} memory slots per VM", self.max),
                ));
            }
        };
        self.mapped.insert(
            region.guest_addr,
            Mapped {
                slot,
                size: region.size,
                host_addr: region.host_addr.as_ptr() as u64,
            },
        );
        Ok(slot)
    }

    /// Gives a reserved slot back after the ioctl that would have used it failed.
    fn release(&mut self, guest_addr: u64) {
        if let Some(mapping) = self.mapped.remove(&guest_addr) {
            self.free.push(mapping.slot);
        }
    }

    /// Finds the slot holding `region`, rejecting a region that was never mapped
    /// or that does not describe the mapping recorded at its guest address.
    fn lookup(&self, region: &MemoryRegion) -> io::Result<u32> {
        let Some(mapping) = self.mapped.get(&region.guest_addr) else {
            return Err(io::Error::new(
                ErrorKind::NotFound,
                format!("no mapping at guest address {:#x}", region.guest_addr),
            ));
        };
        if mapping.size != region.size || mapping.host_addr != region.host_addr.as_ptr() as u64 {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "the mapping at guest address {:#x} covers {} bytes of other host memory",
                    region.guest_addr, mapping.size
                ),
            ));
        }
        Ok(mapping.slot)
    }

    /// Forgets a mapping KVM has dropped, making its slot available again.
    fn forget(&mut self, guest_addr: u64) {
        self.release(guest_addr);
    }
}

/// Describes `region` to KVM under `slot`.
fn kvm_region(slot: u32, region: &MemoryRegion) -> kvm_userspace_memory_region {
    kvm_userspace_memory_region {
        slot,
        guest_phys_addr: region.guest_addr,
        memory_size: region.size as u64,
        userspace_addr: region.host_addr.as_ptr() as u64,
        flags: 0,
    }
}

/// Describes the removal of `slot`: KVM drops a mapping whose size is zero.
fn kvm_region_removal(slot: u32, guest_addr: u64) -> kvm_userspace_memory_region {
    kvm_userspace_memory_region {
        slot,
        guest_phys_addr: guest_addr,
        memory_size: 0,
        userspace_addr: 0,
        flags: 0,
    }
}

/// Registers `region` with KVM through `set`.
///
/// A failed ioctl returns the slot, so a caller that retries after fixing the
/// region does not leak it.
pub(super) fn map_region(
    slots: &Mutex<SlotAllocator>,
    region: &MemoryRegion,
    set: impl FnOnce(kvm_userspace_memory_region) -> io::Result<()>,
) -> Result<()> {
    let mut slots = lock(slots);
    let slot = slots.reserve(region).map_err(|source| Error::MapMemory {
        guest_addr: region.guest_addr,
        size: region.size,
        source,
    })?;

    match set(kvm_region(slot, region)) {
        Ok(()) => Ok(()),
        Err(source) => {
            slots.release(region.guest_addr);
            Err(Error::MapMemory {
                guest_addr: region.guest_addr,
                size: region.size,
                source,
            })
        }
    }
}

/// Removes `region` from KVM through `set`.
///
/// A failed ioctl keeps the record, because the guest mapping may still be
/// live; the contract on [`Vm::unmap_memory`](crate::Vm::unmap_memory) requires
/// the host memory to stay backing it in that case.
pub(super) fn unmap_region(
    slots: &Mutex<SlotAllocator>,
    region: &MemoryRegion,
    set: impl FnOnce(kvm_userspace_memory_region) -> io::Result<()>,
) -> Result<()> {
    let mut slots = lock(slots);
    let slot = slots.lookup(region).map_err(|source| Error::UnmapMemory {
        guest_addr: region.guest_addr,
        size: region.size,
        source,
    })?;

    set(kvm_region_removal(slot, region.guest_addr)).map_err(|source| Error::UnmapMemory {
        guest_addr: region.guest_addr,
        size: region.size,
        source,
    })?;
    slots.forget(region.guest_addr);
    Ok(())
}

/// Takes the allocator lock, recovering from a panic in an earlier caller.
///
/// A poisoned allocator is still consistent: every path that can panic does so
/// before it records or removes a mapping.
fn lock(slots: &Mutex<SlotAllocator>) -> std::sync::MutexGuard<'_, SlotAllocator> {
    slots.lock().unwrap_or_else(|poisoned| {
        slots.clear_poison();
        poisoned.into_inner()
    })
}

#[cfg(test)]
mod tests {
    use std::{
        io::{self, ErrorKind},
        ptr::NonNull,
        sync::Mutex,
    };

    use super::{SlotAllocator, kvm_region, kvm_region_removal, map_region, unmap_region};
    use crate::{Error, MemoryRegion};

    const PAGE: usize = 4096;

    /// A region pointing at an address that is never dereferenced: these tests
    /// exercise bookkeeping, not mapping.
    fn region(guest_addr: u64, host_addr: usize, size: usize) -> MemoryRegion {
        MemoryRegion {
            guest_addr,
            host_addr: NonNull::new(host_addr as *mut u8).expect("non-null host address"),
            size,
        }
    }

    fn allocator(max: u32) -> Mutex<SlotAllocator> {
        Mutex::new(SlotAllocator::new(max))
    }

    fn accept(_: kvm_bindings::kvm_userspace_memory_region) -> io::Result<()> {
        Ok(())
    }

    fn reject(
        kind: ErrorKind,
    ) -> impl FnOnce(kvm_bindings::kvm_userspace_memory_region) -> io::Result<()> {
        move |_| Err(io::Error::new(kind, "the host rejected the region"))
    }

    #[test]
    fn slots_are_handed_out_in_order_and_reused_after_an_unmap() {
        let slots = allocator(8);
        let mut issued = Vec::new();
        for index in 0..3u64 {
            let region = region(index * PAGE as u64, 0x1000 + index as usize * PAGE, PAGE);
            map_region(&slots, &region, |kvm| {
                issued.push(kvm.slot);
                Ok(())
            })
            .expect("map a region");
        }
        assert_eq!(issued, vec![0, 1, 2], "slots start at zero and count up");

        let middle = region(PAGE as u64, 0x1000 + PAGE, PAGE);
        unmap_region(&slots, &middle, accept).expect("unmap the middle region");

        let fresh = region(0x9000, 0x9000, PAGE);
        let mut reused = None;
        map_region(&slots, &fresh, |kvm| {
            reused = Some(kvm.slot);
            Ok(())
        })
        .expect("map after an unmap");

        assert_eq!(
            reused,
            Some(1),
            "a released slot is reused before the counter grows"
        );
    }

    #[test]
    fn slot_exhaustion_is_reported_before_the_ioctl() {
        let slots = allocator(1);
        map_region(&slots, &region(0, 0x1000, PAGE), accept).expect("map the only slot");

        let mut attempted = false;
        let error = map_region(&slots, &region(PAGE as u64, 0x2000, PAGE), |_| {
            attempted = true;
            Ok(())
        })
        .expect_err("a second mapping must not fit");

        assert!(!attempted, "an exhausted allocator must not reach KVM");
        let Error::MapMemory { source, .. } = error else {
            panic!("expected a map failure, got {error:?}");
        };
        assert_eq!(source.kind(), ErrorKind::QuotaExceeded);
        assert!(
            source.to_string().contains('1'),
            "the message names the host's limit: {source}"
        );
    }

    #[test]
    fn mapping_the_same_guest_address_twice_is_rejected() {
        let slots = allocator(8);
        map_region(&slots, &region(0x8000, 0x1000, PAGE), accept).expect("map a region");

        let error = map_region(&slots, &region(0x8000, 0x2000, PAGE), accept)
            .expect_err("the guest address is taken");

        let Error::MapMemory { source, .. } = error else {
            panic!("expected a map failure, got {error:?}");
        };
        assert_eq!(source.kind(), ErrorKind::AlreadyExists);
    }

    #[test]
    fn unmapping_an_unknown_or_mismatched_region_is_rejected() {
        let slots = allocator(8);
        let mapped = region(0x8000, 0x1000, PAGE);
        map_region(&slots, &mapped, accept).expect("map a region");

        let cases = [
            (region(0x9000, 0x1000, PAGE), ErrorKind::NotFound),
            (region(0x8000, 0x1000, 2 * PAGE), ErrorKind::InvalidInput),
            (region(0x8000, 0x2000, PAGE), ErrorKind::InvalidInput),
        ];

        for (candidate, expected) in cases {
            let mut attempted = false;
            let error = unmap_region(&slots, &candidate, |_| {
                attempted = true;
                Ok(())
            })
            .expect_err("the region does not describe a live mapping");

            assert!(!attempted, "an unknown region must not reach KVM");
            let Error::UnmapMemory { source, .. } = error else {
                panic!("expected an unmap failure, got {error:?}");
            };
            assert_eq!(source.kind(), expected, "for {candidate:?}");
        }
    }

    #[test]
    fn a_failed_registration_returns_the_slot() {
        let slots = allocator(1);
        let error = map_region(
            &slots,
            &region(0, 0x1000, PAGE),
            reject(ErrorKind::AlreadyExists),
        )
        .expect_err("the host rejected the region");

        let Error::MapMemory {
            guest_addr,
            size,
            source,
        } = error
        else {
            panic!("expected a map failure, got {error:?}");
        };
        assert_eq!((guest_addr, size), (0, PAGE));
        assert_eq!(source.kind(), ErrorKind::AlreadyExists);

        map_region(&slots, &region(PAGE as u64, 0x2000, PAGE), accept)
            .expect("the only slot is free again");
    }

    #[test]
    fn a_failed_removal_keeps_the_mapping() {
        let slots = allocator(8);
        let mapped = region(0x8000, 0x1000, PAGE);
        map_region(&slots, &mapped, accept).expect("map a region");

        let error = unmap_region(&slots, &mapped, reject(ErrorKind::PermissionDenied))
            .expect_err("the host rejected the removal");
        let Error::UnmapMemory { source, .. } = error else {
            panic!("expected an unmap failure, got {error:?}");
        };
        assert_eq!(source.kind(), ErrorKind::PermissionDenied);

        let mut slot = None;
        unmap_region(&slots, &mapped, |kvm| {
            slot = Some(kvm.slot);
            Ok(())
        })
        .expect("the mapping is still recorded and can be retried");
        assert_eq!(slot, Some(0), "the retry addresses the original slot");
    }

    #[test]
    fn regions_translate_to_kvm_memory_regions() {
        let region = region(0x8000_0000, 0x7f00_0000, 2 * PAGE);

        let mapping = kvm_region(3, &region);
        assert_eq!(mapping.slot, 3);
        assert_eq!(mapping.guest_phys_addr, 0x8000_0000);
        assert_eq!(mapping.userspace_addr, 0x7f00_0000);
        assert_eq!(mapping.memory_size, 2 * PAGE as u64);
        assert_eq!(mapping.flags, 0);

        let removal = kvm_region_removal(3, region.guest_addr);
        assert_eq!(removal.slot, 3);
        assert_eq!(removal.guest_phys_addr, 0x8000_0000);
        assert_eq!(
            removal.memory_size, 0,
            "KVM drops a mapping whose size is zero"
        );
    }
}
