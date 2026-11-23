//! `move_off_page` and incremental per-page compaction cursor (`PageTargets`).
use super::*;

/// Maximum queued keys retained in [`PageTargets`] after a page finishes compacting.
const MAX_RETAINED_KEYS: usize = 4096;

enum MoveOutcome {
    NotOnPage,
    Moved,
    BudgetExhausted,
    NoRoom,
}

#[allow(clippy::too_many_arguments)]
fn move_one(
    arena: &mut arena::Arena,
    encoder: &mut wgpu::CommandEncoder,
    flat: &mut Option<(&mut flat::Flat, &wgpu::Device, &mut upload::Uploads)>,
    records: Option<&mut gpu_culling::Records>,
    meshlets: Option<&mut surface_setup::mesh_shader::MeshSectionStore>,
    s: &mut SectionGpu,
    page: u32,
    part: Part,
    budget: &mut u64,
    moved: bool,
) -> MoveOutcome {
    // Nothing of a mesh here: split, one without solid quads (its (0, 0) is no
    // range of page 0); in the other arenas, one without translucent quads.
    let Some((_, start, vertices)) = part.of(s).filter(|r| r.0 == page) else {
        return MoveOutcome::NotOnPage;
    };
    let bytes = vertices * arena.stride();
    if moved && bytes > *budget {
        return MoveOutcome::BudgetExhausted;
    }
    let Some(to) = arena.alloc_moved(vertices) else {
        return MoveOutcome::NoRoom;
    };
    arena.copy(encoder, (page, start), to, vertices);
    arena.release(page, start, vertices);
    if let (Some((flat, device, uploads)), Some(id)) = (flat, s.flat_id) {
        let key = match part {
            Part::Translucent => to.0 + flat::TRANSLUCENT_KEYS,
            _ => to.0,
        };
        flat.place(device, encoder, uploads, id, key, to.1, vertices);
    }
    let old_solid_base = s.solid_base();
    part.set(s, to);
    // Its solid quads start the range in either layout.
    if let (Part::Section { .. }, Some(records), Some(id)) = (part, records, s.gpu_id) {
        records.relocate(id, to.0, u64::from(s.solid_base()));
    }
    if let (Part::Section { .. }, Some(store)) = (part, meshlets)
        && s.solid > 0
    {
        store.relocate_section(page, old_solid_base, to.0, s.solid_base());
    }
    *budget = budget.saturating_sub(bytes);
    MoveOutcome::Moved
}

/// Cached resident section and far-LOD tile keys holding ranges on one arena's
/// closing page, drained incrementally across compaction frames so each frame
/// touches only the meshes moved instead of re-scanning every resident section
/// from index 0.
#[derive(Default)]
pub(super) struct PageTargets {
    /// `(part, page)` this queue was collected for, if active.
    target: Option<(Part, u32)>,
    /// Remaining resident section keys to move off `page` (popped from the back).
    sections: Vec<SectionKey>,
    /// Remaining far-LOD tile keys to move off `page` (popped from the back).
    lods: Vec<crate::lod::TileKey>,
}

impl PageTargets {
    pub(super) fn clear(&mut self) {
        self.target = None;
        self.sections.clear();
        self.lods.clear();
        if self.sections.capacity() > MAX_RETAINED_KEYS {
            self.sections.shrink_to(MAX_RETAINED_KEYS);
        }
        if self.lods.capacity() > MAX_RETAINED_KEYS {
            self.lods.shrink_to(MAX_RETAINED_KEYS);
        }
    }

    pub(super) fn matches(&self, part: Part, page: u32) -> bool {
        self.target == Some((part, page))
    }

    pub(super) fn is_empty(&self) -> bool {
        self.sections.is_empty() && self.lods.is_empty()
    }

    /// Scans `sections` and `lods` once to collect every key holding a range on
    /// `(part, page)`, returning the total rounded vertex span held there.
    pub(super) fn collect(
        &mut self,
        part: Part,
        page: u32,
        sections: &resident::Resident<SectionGpu>,
        lods: &mut FxHashMap<crate::lod::TileKey, lod::Tile>,
    ) -> u64 {
        self.sections.clear();
        self.lods.clear();
        let mut held = 0u64;
        for (key, s, _) in sections.iter() {
            if let Some((_, _, n)) = part.of(s).filter(|r| r.0 == page) {
                held += arena::Arena::span(n);
                self.sections.push(key);
            }
        }
        for (&key, tile) in lods.iter_mut() {
            let mut any = false;
            for s in tile.meshes_mut() {
                if let Some((_, _, n)) = part.of(s).filter(|r| r.0 == page) {
                    held += arena::Arena::span(n);
                    any = true;
                }
            }
            if any {
                self.lods.push(key);
            }
        }
        // Reverse so `.pop()` drains in forward iteration order.
        self.sections.reverse();
        self.lods.reverse();
        self.target = Some((part, page));
        held
    }

    /// Incremental equivalent of `move_off_page_with_meshlets`: collects the keys
    /// on `(part, page)` once if needed, then pops and moves entries until the
    /// next range would pass `budget` bytes (always moving the first range).
    /// Returns `false` when the other pages have no room left.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn move_step(
        &mut self,
        arena: &mut arena::Arena,
        encoder: &mut wgpu::CommandEncoder,
        mut flat: Option<(&mut flat::Flat, &wgpu::Device, &mut upload::Uploads)>,
        mut records: Option<&mut gpu_culling::Records>,
        mut meshlets: Option<&mut surface_setup::mesh_shader::MeshSectionStore>,
        sections: &mut resident::Resident<SectionGpu>,
        lods: &mut FxHashMap<crate::lod::TileKey, lod::Tile>,
        page: u32,
        part: Part,
        budget: &mut u64,
    ) -> bool {
        debug_assert!(
            records.is_none() || matches!(part, Part::Section { .. }),
            "records follow the section arena's ranges"
        );
        if !self.matches(part, page) {
            self.collect(part, page, sections, lods);
        }
        let mut moved = false;
        while let Some(&key) = self.sections.last() {
            let Some(s) = sections.find_mut(key) else {
                self.sections.pop();
                continue;
            };
            match move_one(
                arena,
                encoder,
                &mut flat,
                records.as_deref_mut(),
                meshlets.as_deref_mut(),
                s,
                page,
                part,
                budget,
                moved,
            ) {
                MoveOutcome::NotOnPage => {
                    self.sections.pop();
                }
                MoveOutcome::Moved => {
                    self.sections.pop();
                    moved = true;
                }
                MoveOutcome::BudgetExhausted => return true,
                MoveOutcome::NoRoom => {
                    self.clear();
                    return false;
                }
            }
        }
        while let Some(&key) = self.lods.last() {
            let Some(tile) = lods.get_mut(&key) else {
                self.lods.pop();
                continue;
            };
            for s in tile.meshes_mut() {
                match move_one(
                    arena,
                    encoder,
                    &mut flat,
                    records.as_deref_mut(),
                    meshlets.as_deref_mut(),
                    s,
                    page,
                    part,
                    budget,
                    moved,
                ) {
                    MoveOutcome::NotOnPage => {}
                    MoveOutcome::Moved => {
                        moved = true;
                    }
                    MoveOutcome::BudgetExhausted => return true,
                    MoveOutcome::NoRoom => {
                        self.clear();
                        return false;
                    }
                }
            }
            self.lods.pop();
        }
        if arena.used(page) > 0 {
            debug_assert_eq!(
                arena.used(page),
                0,
                "compaction queue missed a mesh on closing page"
            );
            self.collect(part, page, sections, lods);
            if self.is_empty() {
                self.clear();
                return false;
            }
        }
        true
    }
}

/// `Renderer::compact_arena`'s and `shrink_translucent`'s moves: what the meshes
/// hold on `page` of `arena` (`part`) goes to its other pages in video memory,
/// until the next range would pass `budget` bytes (it keeps what is left; the
/// first range moves whatever its size, so none keeps the page closed), the
/// copies into `encoder`, the ranges' flat draw ids (`flat`) and, in the
/// section arena, GPU culling's records (`records`) along. False when the other
/// pages have no room left.
#[cfg(test)]
pub(super) fn move_off_page<'a>(
    arena: &mut arena::Arena,
    encoder: &mut wgpu::CommandEncoder,
    flat: Option<(&mut flat::Flat, &wgpu::Device, &mut upload::Uploads)>,
    records: Option<&mut gpu_culling::Records>,
    meshes: impl Iterator<Item = &'a mut SectionGpu>,
    page: u32,
    part: Part,
    budget: &mut u64,
) -> bool {
    move_off_page_with_meshlets(
        arena, encoder, flat, records, None, meshes, page, part, budget,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn move_off_page_with_meshlets<'a>(
    arena: &mut arena::Arena,
    encoder: &mut wgpu::CommandEncoder,
    mut flat: Option<(&mut flat::Flat, &wgpu::Device, &mut upload::Uploads)>,
    mut records: Option<&mut gpu_culling::Records>,
    mut meshlets: Option<&mut surface_setup::mesh_shader::MeshSectionStore>,
    meshes: impl Iterator<Item = &'a mut SectionGpu>,
    page: u32,
    part: Part,
    budget: &mut u64,
) -> bool {
    debug_assert!(
        records.is_none() || matches!(part, Part::Section { .. }),
        "records follow the section arena's ranges"
    );
    let mut moved = false;
    for s in meshes {
        match move_one(
            arena,
            encoder,
            &mut flat,
            records.as_deref_mut(),
            meshlets.as_deref_mut(),
            s,
            page,
            part,
            budget,
            moved,
        ) {
            MoveOutcome::NotOnPage => {}
            MoveOutcome::Moved => moved = true,
            MoveOutcome::BudgetExhausted => break,
            MoveOutcome::NoRoom => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = crate::gfx::test_gpu::instance();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn read_word(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        offset: u64,
    ) -> u32 {
        let (tx, rx) = std::sync::mpsc::channel();
        wgpu::util::DownloadBuffer::read_buffer(
            device,
            queue,
            &buffer.slice(offset..offset + 4),
            move |result| tx.send(result.unwrap().to_vec()).unwrap(),
        );
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        u32::from_le_bytes(rx.recv().unwrap()[..4].try_into().unwrap())
    }

    fn section_at(page: u32, start: u64, solid_quads: u32) -> SectionGpu {
        SectionGpu {
            page,
            start,
            translucent_at: (0, 0),
            solid: solid_quads,
            solid_slots: u64::from(solid_quads) * 4,
            packed: false,
            packed_far: false,
            sides: crate::mesher::Sides::unsorted(solid_quads),
            translucent: 0,
            sort: super::super::translucency::SortState::new(Vec::new()),
            sorted_indices: None,
            flat_id: None,
            gpu_id: None,
            gpu_fading: false,
            fading: None,
            leaving: None,
            animated_layers: Box::new([]),
            ray_looks: Default::default(),
        }
    }

    #[test]
    fn page_targets_drains_incrementally_across_frames_and_skips_removed_or_replaced() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let stride = std::mem::size_of::<crate::mesher::TerrainVertex>() as u64;
        let grain = arena::GRAIN;
        let quads = (grain / 4) as u32;
        let mut arena = arena::Arena::new(&device, stride, 8 * grain, false);
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut uploads = upload::Uploads::new(&device);

        // Allocate 6 sections on page 0 and open a second page with room for all of them.
        let mut resident = resident::Resident::default();
        let mut lods: FxHashMap<crate::lod::TileKey, lod::Tile> = FxHashMap::default();
        let mut page0_keys = Vec::new();
        for i in 0..6u32 {
            let (page, start) = arena.alloc(&device, &mut encoder, grain).unwrap();
            assert_eq!(page, 0);
            uploads.buffer(
                &mut encoder,
                arena.buffer(page),
                start * stride,
                &(1000 + i).to_le_bytes(),
            );
            let key = (i as i32, 0, 0);
            resident.insert(key, section_at(page, start, quads));
            page0_keys.push((key, 1000 + i));
        }
        assert!(arena.open_page(&device, 64 * grain));
        let part = Part::Section { split: true };
        arena.close(0);
        // Add 40 unrelated live sections on page 1 so Resident is mostly non-target sections.
        for i in 0..40i32 {
            let (page, start) = arena.alloc_existing(grain).unwrap();
            assert_eq!(page, 1);
            resident.insert((100 + i, 0, 0), section_at(page, start, quads));
        }
        let mut targets = PageTargets::default();
        let held = targets.collect(part, 0, &resident, &mut lods);
        assert_eq!(held, 6 * grain);
        assert_eq!(targets.sections.len(), 6);
        assert!(targets.matches(part, 0));

        // Frame 1: budget for 2 sections -> pops 2 from queue, leaves 4 queued.
        let mut budget = 2 * grain * stride;
        assert!(targets.move_step(
            &mut arena,
            &mut encoder,
            None,
            None,
            None,
            &mut resident,
            &mut lods,
            0,
            part,
            &mut budget,
        ));
        assert_eq!(budget, 0);
        assert_eq!(targets.sections.len(), 4);
        assert_eq!(arena.used(0), 4 * grain);

        // Between frames: remove key #2 and replace key #3 with a fresh allocation on page 1.
        let removed = resident.remove(page0_keys[2].0).unwrap();
        arena.release(removed.page, removed.start, grain);
        let old3 = resident.remove(page0_keys[3].0).unwrap();
        arena.release(old3.page, old3.start, grain);
        let (rep_page, rep_start) = arena.alloc(&device, &mut encoder, grain).unwrap();
        assert_eq!(rep_page, 1);
        uploads.buffer(
            &mut encoder,
            arena.buffer(rep_page),
            rep_start * stride,
            &9999u32.to_le_bytes(),
        );
        resident.insert(page0_keys[3].0, section_at(rep_page, rep_start, quads));

        // Frame 2: zero budget still skips removed #2 & replaced #3 and moves 1 live section (#4).
        let mut budget = 0;
        assert!(targets.move_step(
            &mut arena,
            &mut encoder,
            None,
            None,
            None,
            &mut resident,
            &mut lods,
            0,
            part,
            &mut budget,
        ));
        assert_eq!(targets.sections.len(), 1);
        assert_eq!(arena.used(0), grain);

        // Frame 3: moves the final section (#5), emptying page 0 completely.
        let mut budget = u64::MAX;
        assert!(targets.move_step(
            &mut arena,
            &mut encoder,
            None,
            None,
            None,
            &mut resident,
            &mut lods,
            0,
            part,
            &mut budget,
        ));
        assert!(targets.is_empty());
        assert_eq!(arena.used(0), 0);

        uploads.finish(&encoder);
        queue.submit([encoder.finish()]);

        // Verify every surviving section moved cleanly to page 1 with its exact bytes intact.
        for &(key, expected) in &page0_keys {
            if key == page0_keys[2].0 {
                assert!(!resident.contains(key));
                continue;
            }
            let s = resident.find(key).unwrap();
            assert_eq!(s.page, 1);
            let want = if key == page0_keys[3].0 {
                9999
            } else {
                expected
            };
            assert_eq!(
                read_word(&device, &queue, arena.buffer(1), s.start * stride),
                want
            );
        }
        drop(arena.trim());
        assert!(!arena.has_page(0));
    }

    #[test]
    fn page_targets_clears_when_other_pages_have_no_room() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some((device, _queue)) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let stride = std::mem::size_of::<crate::mesher::TerrainVertex>() as u64;
        let grain = arena::GRAIN;
        let quads = (grain / 4) as u32;
        let mut arena = arena::Arena::new(&device, stride, grain, false);
        let mut encoder = device.create_command_encoder(&Default::default());
        let (p0, s0) = arena.alloc(&device, &mut encoder, grain).unwrap();
        assert_eq!(p0, 0);
        let mut resident = resident::Resident::default();
        resident.insert((0, 0, 0), section_at(p0, s0, quads));
        let mut lods: FxHashMap<crate::lod::TileKey, lod::Tile> = FxHashMap::default();

        // Only page 0 exists -> closing it leaves no destination page -> returns false & clears.
        let part = Part::Section { split: true };
        arena.close(0);
        let mut targets = PageTargets::default();
        let mut budget = u64::MAX;
        assert!(!targets.move_step(
            &mut arena,
            &mut encoder,
            None,
            None,
            None,
            &mut resident,
            &mut lods,
            0,
            part,
            &mut budget,
        ));
        assert!(targets.is_empty());
        assert!(!targets.matches(part, 0));
    }
}
