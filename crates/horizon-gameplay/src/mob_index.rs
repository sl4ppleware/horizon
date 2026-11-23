//! Finding an entity by its network id without walking the whole list: a
//! multiplayer client looks one up for every movement, data, motion and
//! event packet. A `Mob` is ~3 KB, so a linear search touches a cache line
//! per mob: 1000 movement packets in a 3000-mob farm cost ~5 ms a tick that
//! way, against ~0.01 ms through the index.
//!
//! The index is a cache backed by [`JoinList::structure_stamp`]: while a
//! list's structure has not changed (including across in-place element edits
//! via `get_mut`, `IndexMut` or `update`), both hits and misses resolve in
//! O(1) without scanning the slice. Any structural change (`push`, `remove`,
//! `retain`, `clear`, `lift`/`restore`, dimension swap) invalidates the stamp
//! and triggers one O(N) rebuild. Ids are unique in the list
//! (`spawn_entity` replaces any mob with the same id before adding one), so
//! the answer is always the linear search's.

use horizon_client::fast_hash::FxHashMap;
use horizon_client::join_seq::{JoinList, Joined, ListStamp};
use horizon_client::mobs::Mob;

fn find_in_slice<T>(
    slots: &mut FxHashMap<u32, u32>,
    stamp: &mut Option<ListStamp>,
    items: &[T],
    id_of: impl Fn(&T) -> u32,
    id: u32,
) -> Option<usize> {
    if let Some(&i) = slots.get(&id)
        && items.get(i as usize).is_some_and(|m| id_of(m) == id)
    {
        return Some(i as usize);
    }
    let i = items.iter().position(|m| id_of(m) == id)?;
    slots.clear();
    for (idx, m) in items.iter().enumerate() {
        slots.entry(id_of(m)).or_insert(idx as u32);
    }
    *stamp = None;
    Some(i)
}

fn rebuild_stamped<T, const DIRECT: bool>(
    slots: &mut FxHashMap<u32, u32>,
    stamp_slot: &mut Option<ListStamp>,
    list: &JoinList<T, DIRECT>,
    id_of: &impl Fn(&T) -> u32,
    stamp: ListStamp,
) {
    slots.clear();
    let mut has_zero = false;
    for (i, item) in list.iter().enumerate() {
        let item_id = id_of(item);
        has_zero |= item_id == 0;
        slots.entry(item_id).or_insert(i as u32);
    }
    *stamp_slot = (!has_zero).then_some(stamp);
}

fn find_in_list<T, const DIRECT: bool>(
    slots: &mut FxHashMap<u32, u32>,
    stamp_slot: &mut Option<ListStamp>,
    list: &JoinList<T, DIRECT>,
    id_of: impl Fn(&T) -> u32,
    id: u32,
) -> Option<usize> {
    if list.is_empty() {
        slots.clear();
        *stamp_slot = Some(list.structure_stamp());
        return None;
    }
    let stamp = list.structure_stamp();
    if *stamp_slot == Some(stamp) {
        if let Some(&i) = slots.get(&id) {
            if list.get(i as usize).is_some_and(|e| id_of(e) == id) {
                return Some(i as usize);
            }
        } else {
            return None;
        }
    }
    rebuild_stamped(slots, stamp_slot, list, &id_of, stamp);
    slots.get(&id).copied().map(|i| i as usize)
}

fn replace_or_push_in_list<T: Joined, const DIRECT: bool>(
    slots: &mut FxHashMap<u32, u32>,
    stamp_slot: &mut Option<ListStamp>,
    list: &mut JoinList<T, DIRECT>,
    id_of: impl Fn(&T) -> u32,
    item: T,
) {
    let id = id_of(&item);
    if list.is_empty() {
        slots.clear();
        list.push(item);
        if id != 0 {
            slots.insert(id, 0);
            *stamp_slot = Some(list.structure_stamp());
        } else {
            *stamp_slot = None;
        }
        return;
    }
    let stamp = list.structure_stamp();
    if *stamp_slot != Some(stamp) {
        rebuild_stamped(slots, stamp_slot, list, &id_of, stamp);
    }
    let duplicate = if *stamp_slot == Some(stamp) {
        slots.contains_key(&id)
    } else {
        list.iter().any(|existing| id_of(existing) == id)
    };
    if duplicate {
        list.retain(|existing| id_of(existing) != id);
        list.push(item);
        *stamp_slot = None;
    } else {
        let idx = list.len();
        list.push(item);
        if *stamp_slot == Some(stamp) && id != 0 {
            slots.insert(id, idx as u32);
            *stamp_slot = Some(list.structure_stamp());
        } else {
            *stamp_slot = None;
        }
    }
}

/// Stamp-validated O(1) lookup cache from entity network id to list slot for
/// any [`JoinList`].
#[derive(Default)]
pub struct EntityIndex {
    slots: FxHashMap<u32, u32>,
    stamp: Option<ListStamp>,
}

impl EntityIndex {
    /// Drop cached slot entries and stamp.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.stamp = None;
    }

    /// The position of the first item with `id` in `items`, as
    /// `items.iter().position(|m| id_of(m) == id)`.
    pub fn find<T>(&mut self, items: &[T], id_of: impl Fn(&T) -> u32, id: u32) -> Option<usize> {
        find_in_slice(&mut self.slots, &mut self.stamp, items, id_of, id)
    }

    /// Stamp-validated position lookup on a [`JoinList`]: O(1) on both hits
    /// and misses while `list`'s structure has not changed. Lists holding an
    /// unassigned `id == 0` entry do not cache negative lookups until live ids
    /// are assigned.
    pub fn find_stamped<T, const DIRECT: bool>(
        &mut self,
        list: &JoinList<T, DIRECT>,
        id_of: impl Fn(&T) -> u32,
        id: u32,
    ) -> Option<usize> {
        find_in_list(&mut self.slots, &mut self.stamp, list, id_of, id)
    }

    /// Stamp-validated immutable lookup on a [`JoinList`].
    pub fn get_stamped<'a, T, const DIRECT: bool>(
        &mut self,
        list: &'a JoinList<T, DIRECT>,
        id_of: impl Fn(&T) -> u32,
        id: u32,
    ) -> Option<&'a T> {
        let i = self.find_stamped(list, id_of, id)?;
        Some(&list[i])
    }

    /// Stamp-validated mutable lookup on a direct [`JoinList`].
    pub fn get_mut_stamped<'a, T: Joined>(
        &mut self,
        list: &'a mut JoinList<T>,
        id_of: impl Fn(&T) -> u32,
        id: u32,
    ) -> Option<&'a mut T> {
        let i = self.find_stamped(list, id_of, id)?;
        Some(&mut list[i])
    }

    /// Remove any existing element sharing `item`'s id, then append `item`
    /// while updating the index in O(1) amortized time across spawn bursts.
    pub fn replace_or_push<T: Joined, const DIRECT: bool>(
        &mut self,
        list: &mut JoinList<T, DIRECT>,
        id_of: impl Fn(&T) -> u32,
        item: T,
    ) {
        replace_or_push_in_list(&mut self.slots, &mut self.stamp, list, id_of, item);
    }
}

#[derive(Default)]
pub struct MobIndex {
    slots: FxHashMap<u32, u32>,
    stamp: Option<ListStamp>,
}

impl MobIndex {
    /// The position of the mob with this id in `mobs`, as
    /// `mobs.iter().position(|m| m.id == id)`.
    pub fn find(&mut self, mobs: &[Mob], id: u32) -> Option<usize> {
        self.position(mobs, |m| m.id, id)
    }

    fn position<T>(&mut self, items: &[T], id_of: impl Fn(&T) -> u32, id: u32) -> Option<usize> {
        find_in_slice(&mut self.slots, &mut self.stamp, items, id_of, id)
    }

    /// Stamp-validated position lookup on `mobs`: O(1) on both hits and misses
    /// while `mobs`'s structure has not changed.
    pub fn find_stamped(&mut self, mobs: &JoinList<Mob>, id: u32) -> Option<usize> {
        find_in_list(&mut self.slots, &mut self.stamp, mobs, |m| m.id, id)
    }

    /// The mob with this id, as `mobs.iter_mut().find(|m| m.id == id)`.
    pub fn get_mut<'a>(&mut self, mobs: &'a mut JoinList<Mob>, id: u32) -> Option<&'a mut Mob> {
        let i = self.find_stamped(mobs, id)?;
        Some(&mut mobs[i])
    }

    /// The mob with this id, as `mobs.iter().find(|m| m.id == id)`.
    pub fn get<'a>(&mut self, mobs: &'a [Mob], id: u32) -> Option<&'a Mob> {
        let i = self.find(mobs, id)?;
        Some(&mobs[i])
    }

    /// Stamp-validated immutable lookup on `mobs`: O(1) on both hits and
    /// misses while `mobs`'s structure has not changed.
    pub fn get_stamped<'a>(&mut self, mobs: &'a JoinList<Mob>, id: u32) -> Option<&'a Mob> {
        let i = self.find_stamped(mobs, id)?;
        Some(&mobs[i])
    }

    /// Drop any existing mob with `mob.id`, then append `mob` while keeping
    /// the index synchronized in O(1) amortized time across spawn bursts.
    pub fn replace_or_push(&mut self, mobs: &mut JoinList<Mob>, mob: Mob) {
        replace_or_push_in_list(&mut self.slots, &mut self.stamp, mobs, |m| m.id, mob);
    }

    /// Drop cached slot entries and stamp.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.stamp = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_client::join_seq::{JoinSeq, RestoreOrder};
    use std::cell::Cell;

    #[derive(Clone, Debug)]
    struct Entry {
        id: u32,
        join: JoinSeq,
        value: u32,
    }

    impl Joined for Entry {
        fn join_seq(&self) -> JoinSeq {
            self.join
        }
        fn set_join_seq(&mut self, seq: JoinSeq) {
            self.join = seq;
        }
    }

    fn entry(id: u32, value: u32) -> Entry {
        Entry {
            id,
            join: JoinSeq::UNJOINED,
            value,
        }
    }

    /// Whatever the list went through since the last lookup, the index
    /// answers as the linear search does.
    #[test]
    fn finds_what_a_linear_search_finds_after_any_change() {
        let mut index = MobIndex::default();
        let check = |index: &mut MobIndex, ids: &[u32]| {
            for id in 0..20 {
                let found = index.position(ids, |&m| m, id);
                assert_eq!(
                    found,
                    ids.iter().position(|&m| m == id),
                    "id {id} in {ids:?}"
                );
            }
        };
        let mut ids = vec![7, 3, 9, 12];
        check(&mut index, &ids);
        ids.remove(1);
        check(&mut index, &ids);
        ids.push(3);
        ids.swap(0, 2);
        check(&mut index, &ids);
        ids.retain(|&m| m != 9);
        ids.insert(0, 15);
        check(&mut index, &ids);
        // Another list altogether: a dimension swapped in.
        check(&mut index, &[12, 15, 1]);
        ids.clear();
        check(&mut index, &ids);
    }

    #[test]
    fn stamped_misses_do_not_scan_list_and_survive_in_place_edits() {
        let mut list = JoinList::<Entry>::default();
        for id in 1..=1_000 {
            list.push(entry(id, id * 10));
        }
        let mut index = EntityIndex::default();
        let reads = Cell::new(0usize);
        let id_of = |e: &Entry| {
            reads.set(reads.get() + 1);
            e.id
        };

        // First lookup populates the cache in one O(N) pass.
        assert_eq!(index.find_stamped(&list, id_of, 999_999), None);
        assert_eq!(reads.get(), 1_000);

        // Subsequent misses do zero element visits.
        reads.set(0);
        for missing in 10_000..10_500 {
            assert_eq!(index.find_stamped(&list, id_of, missing), None);
        }
        assert_eq!(reads.get(), 0, "negative lookups must not walk the list");

        // In-place edits bump `stamp()` but keep `structure_stamp()` intact.
        for id in [1, 500, 1_000] {
            let e = index.get_mut_stamped(&mut list, id_of, id).unwrap();
            e.value += 1;
        }
        list[0].value += 7;
        list.update(42, |e| e.value += 3);
        list.for_each_mut(|e| e.value += 1);

        reads.set(0);
        for missing in 20_000..20_200 {
            assert_eq!(index.find_stamped(&list, id_of, missing), None);
        }
        assert_eq!(
            reads.get(),
            0,
            "in-place element edits must not invalidate structure-stamped misses"
        );

        // Positive hits verify exactly one slot.
        reads.set(0);
        assert_eq!(index.find_stamped(&list, id_of, 500), Some(499));
        assert_eq!(reads.get(), 1);
    }

    #[test]
    fn stamped_index_invalidates_on_structural_mutations_and_dimension_swaps() {
        let mut list = JoinList::<Entry>::default();
        for id in [7, 3, 9, 12] {
            list.push(entry(id, id));
        }
        let mut index = EntityIndex::default();
        let check = |index: &mut EntityIndex, list: &JoinList<Entry>| {
            for id in 0..20 {
                assert_eq!(
                    index.find_stamped(list, |e| e.id, id),
                    list.iter().position(|e| e.id == id),
                    "id {id}"
                );
            }
        };

        check(&mut index, &list);
        list.remove(1);
        check(&mut index, &list);
        list.push(entry(3, 30));
        list.swap(0, 2);
        check(&mut index, &list);
        list.retain(|e| e.id != 9);
        list.insert(0, entry(15, 15));
        check(&mut index, &list);

        let mut lifted = list.lift();
        check(&mut index, &list);
        lifted.push(entry(18, 18));
        list.restore(lifted, RestoreOrder::SurvivorsFirst);
        check(&mut index, &list);

        // Another list (dimension swap) has a distinct ListStamp identity.
        let mut other = JoinList::<Entry>::default();
        for id in [12, 15, 1] {
            other.push(entry(id, id));
        }
        check(&mut index, &other);
        check(&mut index, &list);

        // Unstamped slice lookup on a foreign slice invalidates the stamped cache.
        assert_eq!(index.find(&[entry(19, 19)], |e| e.id, 19), Some(0));
        check(&mut index, &list);

        list.clear();
        check(&mut index, &list);
    }

    #[test]
    fn replace_or_push_stays_linear_on_spawn_bursts_and_replaces_duplicates() {
        let mut list = JoinList::<Entry, false>::default();
        let mut index = EntityIndex::default();
        let reads = Cell::new(0usize);
        let id_of = |e: &Entry| {
            reads.set(reads.get() + 1);
            e.id
        };

        for id in 1..=1_000 {
            index.replace_or_push(&mut list, id_of, entry(id, id * 2));
        }
        assert_eq!(list.len(), 1_000);
        // Each push only inspects the new item's id once (no O(N^2) scans).
        assert_eq!(reads.get(), 1_000);

        // Index is already warm right after the burst: hits and misses are O(1).
        reads.set(0);
        assert_eq!(index.find_stamped(&list, id_of, 999_999), None);
        assert_eq!(reads.get(), 0);
        assert_eq!(index.find_stamped(&list, id_of, 777), Some(776));
        assert_eq!(reads.get(), 1);

        // Duplicate id replaces the previous entry while preserving other order.
        let mut small = JoinList::<Entry>::default();
        let mut small_idx = EntityIndex::default();
        for (id, value) in [(40, 1), (41, 2), (40, 3), (42, 4)] {
            small_idx.replace_or_push(&mut small, |e| e.id, entry(id, value));
        }
        let snapshot: Vec<_> = small.iter().map(|e| (e.id, e.value)).collect();
        assert_eq!(snapshot, [(41, 2), (40, 3), (42, 4)]);
        assert_eq!(small_idx.find_stamped(&small, |e| e.id, 40), Some(1));
        assert_eq!(small_idx.find_stamped(&small, |e| e.id, 41), Some(0));
        assert_eq!(small_idx.find_stamped(&small, |e| e.id, 42), Some(2));
    }

    #[test]
    fn unassigned_zero_ids_never_cause_stale_negative_lookups() {
        let mut list = JoinList::<Entry>::default();
        list.push(entry(0, 1));
        list.push(entry(10, 2));
        let mut index = EntityIndex::default();

        assert_eq!(index.find_stamped(&list, |e| e.id, 77), None);
        // Delayed server id assignment updates `id` in place from 0 -> 77.
        list.update(0, |e| e.id = 77);
        assert_eq!(index.find_stamped(&list, |e| e.id, 77), Some(0));
        assert_eq!(index.find_stamped(&list, |e| e.id, 0), None);
    }
}
