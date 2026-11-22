//! Vanilla 26.3 MeshData.SortState / VertexSorting.byDistance.
//! Keep vertex order intact; only the quad indices change as the camera moves.

/// MeshData.decodeQuadCentroids uses the opposite corners, not all four vertices.
pub fn quad_center(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| (a[i] + b[i]) / 2.0)
}

/// Fixed-point encoding for section-local dyadic quad centers in `[-8.0, 23.99951171875]`:
/// `SCALE = 2048.0` represents every binary fraction `k / 2048` (covering standard 1/16-block
/// model geometry such as glass, ice, slabs, panes, portals, and banners) without rounding,
/// cutting center storage from 12 B/quad to 6 B/quad (2x L1 cache density during sorting).
///
/// Exact round-trip invariant: `SortState::new` and `SortState::from_quads` select
/// `Centers::Compact` only when every coordinate of every quad round-trips identically through
/// `decode_compact(encode_compact(v)) == v`, and `Sorter::sort_compact` subtracts decoded
/// coordinates directly from `camera`. Compact sorting therefore preserves 100% bit-exact
/// distance and tie-breaking parity with full-precision `f32` sorting. Non-dyadic coordinates
/// (such as flowing fluid heights `k / 9` or `0.001` epsilon offsets) fall back to
/// `Centers::Full`, while non-empty flat single-plane horizontal surfaces (such as ocean water)
/// hit `Centers::Coplanar` (0 B/quad, no re-sort) before `Compact` is checked.
const COMPACT_OFFSET: f32 = 8.0;
const COMPACT_SCALE: f32 = 2048.0;
const COMPACT_INV_SCALE: f32 = 1.0 / COMPACT_SCALE;
const COMPACT_MIN: f32 = -COMPACT_OFFSET;
const COMPACT_MAX: f32 = 24.0 - COMPACT_INV_SCALE;

/// Below this quad count, in-place unstable sort on packed `(dist_key, quad)` `u64`
/// beats radix sort's histogram pass; at or above it, 4-pass 8-bit LSD radix sort
/// with per-byte skip runs in O(n) with zero allocations.
const RADIX_THRESHOLD: usize = 128;

#[inline(always)]
fn dist_key(d2: f32) -> u32 {
    // `d2` is a sum of squares (non-negative or NaN): clearing the sign bit and
    // inverting turns IEEE-754 order into descending distance order, matching
    // `b.total_cmp(&a)`.
    !(d2.to_bits() & 0x7fff_ffff)
}

#[inline(always)]
fn encode_compact(v: f32) -> u16 {
    ((v + COMPACT_OFFSET) * COMPACT_SCALE).round() as u16
}

#[inline(always)]
fn decode_compact(q: u16) -> f32 {
    f32::from(q) * COMPACT_INV_SCALE - COMPACT_OFFSET
}

#[inline(always)]
fn is_exact_compact(v: f32) -> bool {
    (COMPACT_MIN..=COMPACT_MAX).contains(&v) && decode_compact(encode_compact(v)) == v
}

/// Whether a non-empty center list lies on a single grid-aligned horizontal plane
/// inside the section (e.g., flat ocean water at `y = iy + 0.887889` or flat
/// ice/glass top at integer `y`). Disjoint horizontal tiles on one plane never
/// overlap in screen space, and co-located back faces (water underside) are
/// back-face culled by the translucent terrain pipelines (`cull_mode: Back`/`Front`)
/// and share identical distance anyway. Rejects empty sections (so viewpoint
/// cache updates match vanilla with zero allocation), `y.fract() == 0.5`, and
/// off-center X/Z so vertical glass/pane/portal/waterfall faces in one Y layer
/// are never misclassified.
fn is_coplanar_horizontal(centers: &[[f32; 3]]) -> bool {
    let Some(&[_, y0, _]) = centers.first() else {
        return false;
    };
    if !(0.0..=16.0).contains(&y0) || y0.fract() == 0.5 {
        return false;
    }
    centers.iter().all(|c| {
        c[1] == y0
            && (0.0..=16.0).contains(&c[0])
            && (0.0..=16.0).contains(&c[2])
            && c[0].fract() == 0.5
            && c[2].fract() == 0.5
    })
}

/// Vertex-level coplanar check for `SortState::from_quads`: verifies that all
/// four vertices of every quad lie on one horizontal plane `y = y0` inside the
/// section, closing the case where sloped flowing-water tops share a single
/// quad-center height. Rejects empty sections so viewpoint cache decisions keep
/// vanilla parity with zero allocation.
fn quads_are_coplanar_horizontal<T>(quads: &[[T; 4]], pos: &impl Fn(&T) -> [f32; 3]) -> bool {
    let Some(first) = quads.first() else {
        return false;
    };
    let y0 = pos(&first[0])[1];
    if !(0.0..=16.0).contains(&y0) {
        return false;
    }
    quads.iter().all(|q| {
        let p0 = pos(&q[0]);
        let p1 = pos(&q[1]);
        let p2 = pos(&q[2]);
        let p3 = pos(&q[3]);
        if p0[1] != y0 || p1[1] != y0 || p2[1] != y0 || p3[1] != y0 {
            return false;
        }
        let c = quad_center(p0, p2);
        (0.0..=16.0).contains(&c[0])
            && (0.0..=16.0).contains(&c[2])
            && c[0].fract() == 0.5
            && c[2].fract() == 0.5
    })
}

fn fits_compact(centers: &[[f32; 3]]) -> bool {
    centers
        .iter()
        .all(|c| is_exact_compact(c[0]) && is_exact_compact(c[1]) && is_exact_compact(c[2]))
}

#[derive(Default)]
pub struct Sorter {
    keys: Vec<u64>,
    radix_a: Vec<(u32, u32)>,
    radix_b: Vec<(u32, u32)>,
    indices: Vec<u32>,
}

impl Sorter {
    pub fn sort(&mut self, centers: &[[f32; 3]], camera: [f32; 3]) -> &[u32] {
        self.sort_by_dist(centers, |c| {
            let dx = camera[0] - c[0];
            let dy = camera[1] - c[1];
            let dz = camera[2] - c[2];
            // Match vanilla 26.3 JOML Vector3f.distanceSquared (Math.fma with USE_MATH_FMA=false).
            dx * dx + (dy * dy + dz * dz)
        })
    }

    fn sort_compact(&mut self, centers: &[[u16; 3]], camera: [f32; 3]) -> &[u32] {
        self.sort_by_dist(centers, |c| {
            let dx = camera[0] - decode_compact(c[0]);
            let dy = camera[1] - decode_compact(c[1]);
            let dz = camera[2] - decode_compact(c[2]);
            // Match vanilla 26.3 JOML Vector3f.distanceSquared (Math.fma with USE_MATH_FMA=false).
            dx * dx + (dy * dy + dz * dz)
        })
    }

    fn identity(&mut self, quads: u32) -> &[u32] {
        let len = quads as usize;
        self.indices.clear();
        self.indices.reserve(len * 6);
        for quad in 0..quads {
            let v = quad * 4;
            self.indices
                .extend_from_slice(&[v, v + 1, v + 2, v + 2, v + 3, v]);
        }
        &self.indices
    }

    fn sort_by_dist<T>(&mut self, items: &[T], mut dist_sq: impl FnMut(&T) -> f32) -> &[u32] {
        let len = items.len();
        self.indices.clear();
        if len == 0 {
            return &self.indices;
        }
        if len == 1 {
            self.indices.extend_from_slice(&[0, 1, 2, 2, 3, 0]);
            return &self.indices;
        }
        self.indices.reserve(len * 6);
        if len < RADIX_THRESHOLD {
            // Pack `(dist_key << 32) | quad_index` so an in-place unstable sort
            // preserves exact mesh order on ties with zero heap allocation.
            self.keys.clear();
            self.keys.reserve(len);
            for (i, item) in items.iter().enumerate() {
                let key = (u64::from(dist_key(dist_sq(item))) << 32) | u64::from(i as u32);
                self.keys.push(key);
            }
            self.keys.sort_unstable();
            for &entry in &self.keys {
                let v = (entry as u32) * 4;
                self.indices
                    .extend_from_slice(&[v, v + 1, v + 2, v + 2, v + 3, v]);
            }
            return &self.indices;
        }
        // 4-pass 8-bit LSD radix sort with per-byte skip: O(n), stable (preserves
        // mesh order on exact ties), and zero heap allocation after warm-up.
        let mut hist = [[0u32; 256]; 4];
        self.radix_a.clear();
        self.radix_a.reserve(len);
        for (i, item) in items.iter().enumerate() {
            let key = dist_key(dist_sq(item));
            hist[0][(key & 0xff) as usize] += 1;
            hist[1][((key >> 8) & 0xff) as usize] += 1;
            hist[2][((key >> 16) & 0xff) as usize] += 1;
            hist[3][(key >> 24) as usize] += 1;
            self.radix_a.push((key, i as u32));
        }
        let first_key = self.radix_a[0].0;
        let mut from_a = true;
        for (pass, counts) in hist.iter_mut().enumerate() {
            let shift = pass * 8;
            let first_bucket = ((first_key >> shift) & 0xff) as usize;
            if counts[first_bucket] as usize == len {
                continue;
            }
            let mut sum = 0u32;
            for c in counts.iter_mut() {
                let n = *c;
                *c = sum;
                sum += n;
            }
            if from_a {
                self.radix_b.resize(len, (0, 0));
                for &(key, quad) in &self.radix_a {
                    let b = ((key >> shift) & 0xff) as usize;
                    let dst = counts[b] as usize;
                    counts[b] = (dst + 1) as u32;
                    self.radix_b[dst] = (key, quad);
                }
            } else {
                for &(key, quad) in &self.radix_b {
                    let b = ((key >> shift) & 0xff) as usize;
                    let dst = counts[b] as usize;
                    counts[b] = (dst + 1) as u32;
                    self.radix_a[dst] = (key, quad);
                }
            }
            from_a = !from_a;
        }
        let sorted = if from_a { &self.radix_a } else { &self.radix_b };
        for &(_, quad) in sorted {
            let v = quad * 4;
            self.indices
                .extend_from_slice(&[v, v + 1, v + 2, v + 2, v + 3, v]);
        }
        &self.indices
    }
}

enum Centers {
    /// Flat single-plane horizontal surface (e.g., ocean water): identity order
    /// is emitted once on first visibility and never re-sorted. Zero per-quad RAM.
    Coplanar(u32),
    /// Section-local dyadic coordinates in `[-8.0, 23.99951171875]` where every coordinate
    /// round-trips bit-exact through `decode_compact(encode_compact(v)) == v`, 6 B/quad.
    Compact(Box<[[u16; 3]]>),
    /// Full-precision coordinates for out-of-range geometry (LOD tiles, overhanging models)
    /// or non-dyadic centers (e.g., flowing fluid heights `k / 9` and epsilon offsets).
    Full(Box<[[f32; 3]]>),
}

pub struct SortState {
    centers: Centers,
    last_block: [i32; 3],
    last_pov: [i8; 3],
    sorted: bool,
}

impl SortState {
    pub fn new(centers: Vec<[f32; 3]>) -> Self {
        let centers = if is_coplanar_horizontal(&centers) {
            Centers::Coplanar(centers.len() as u32)
        } else if fits_compact(&centers) {
            let compact: Box<[[u16; 3]]> = centers
                .into_iter()
                .map(|c| {
                    [
                        encode_compact(c[0]),
                        encode_compact(c[1]),
                        encode_compact(c[2]),
                    ]
                })
                .collect();
            Centers::Compact(compact)
        } else {
            Centers::Full(centers.into_boxed_slice())
        };
        Self {
            centers,
            last_block: [0; 3],
            last_pov: [0; 3],
            sorted: false,
        }
    }

    /// Builds sort state directly from quad vertices, checking all four vertex Y
    /// coordinates before taking `Centers::Coplanar` so sloped flowing-water
    /// tops whose centers happen to share one height are never treated as flat.
    /// Avoids allocating an intermediate `Vec<[f32; 3]>` for coplanar or compact
    /// sections.
    pub fn from_quads<T>(quads: &[[T; 4]], pos: impl Fn(&T) -> [f32; 3]) -> Self {
        let centers = if quads_are_coplanar_horizontal(quads, &pos) {
            Centers::Coplanar(quads.len() as u32)
        } else if quads.iter().all(|q| {
            let c = quad_center(pos(&q[0]), pos(&q[2]));
            is_exact_compact(c[0]) && is_exact_compact(c[1]) && is_exact_compact(c[2])
        }) {
            let compact: Box<[[u16; 3]]> = quads
                .iter()
                .map(|q| {
                    let c = quad_center(pos(&q[0]), pos(&q[2]));
                    [
                        encode_compact(c[0]),
                        encode_compact(c[1]),
                        encode_compact(c[2]),
                    ]
                })
                .collect();
            Centers::Compact(compact)
        } else {
            let full: Box<[[f32; 3]]> = quads
                .iter()
                .map(|q| quad_center(pos(&q[0]), pos(&q[2])))
                .collect();
            Centers::Full(full)
        };
        Self {
            centers,
            last_block: [0; 3],
            last_pov: [0; 3],
            sorted: false,
        }
    }

    pub fn is_sorted(&self) -> bool {
        self.sorted
    }

    /// Sorted again when next drawn: a frame given up (#13624) dropped the last order's upload.
    pub fn forget(&mut self) {
        self.sorted = false;
    }

    fn sort_now<'a>(
        &self,
        camera: [f64; 3],
        origin: [i32; 3],
        sorter: &'a mut Sorter,
    ) -> &'a [u32] {
        let local = std::array::from_fn(|i| (camera[i] - f64::from(origin[i])) as f32);
        match &self.centers {
            Centers::Coplanar(quads) => sorter.identity(*quads),
            Centers::Compact(centers) => sorter.sort_compact(centers, local),
            Centers::Full(centers) => sorter.sort(centers, local),
        }
    }

    /// Large tiles cannot use the section viewpoint shortcut: the camera may
    /// cross a quad's plane while remaining on the same side of the first 16³.
    pub fn update_lod<'a>(
        &mut self,
        camera: [f64; 3],
        origin: [i32; 3],
        sorter: &'a mut Sorter,
    ) -> Option<&'a [u32]> {
        if let Centers::Coplanar(quads) = self.centers {
            if self.sorted {
                return None;
            }
            self.sorted = true;
            return Some(sorter.identity(quads));
        }
        let block = camera.map(|v| v.floor() as i32);
        if self.sorted && self.last_block == block {
            return None;
        }
        self.last_block = block;
        self.last_pov = [0; 3];
        self.sorted = true;
        Some(self.sort_now(camera, origin, sorter))
    }

    /// TranslucencyPointOfView and ResortTransparencyTask: diagonal views keep
    /// their ordering until the point of view changes; aligned views also
    /// resort on crossing a block boundary. Sub-block movement needs no upload.
    pub fn update<'a>(
        &mut self,
        camera: [f64; 3],
        origin: [i32; 3],
        sorter: &'a mut Sorter,
    ) -> Option<&'a [u32]> {
        if let Centers::Coplanar(quads) = self.centers {
            if self.sorted {
                return None;
            }
            self.sorted = true;
            return Some(sorter.identity(quads));
        }
        let block = camera.map(|v| v.floor() as i32);
        let pov = std::array::from_fn(|i| ((block[i] >> 4) - (origin[i] >> 4)).clamp(-1, 1) as i8);
        if self.sorted && self.last_pov == pov && (self.last_block == block || !pov.contains(&0)) {
            return None;
        }
        self.last_block = block;
        self.last_pov = pov;
        self.sorted = true;
        Some(self.sort_now(camera, origin, sorter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_distances_keep_mesh_order_and_triangle_winding() {
        let mut sorter = Sorter::default();
        assert_eq!(
            sorter.sort(
                &[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [0.0; 3]
            ),
            &[8, 9, 10, 10, 11, 8, 0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4]
        );
        assert!(sorter.sort(&[], [0.0; 3]).is_empty());
    }

    #[test]
    fn movement_reverses_faces_without_moving_vertices() {
        let centers = [[0.5, 0.5, 0.0], [0.5, 0.5, 1.0]];
        let mut sorter = Sorter::default();
        assert_eq!(sorter.sort(&centers, [0.5, 0.5, -2.0])[0], 4);
        assert_eq!(sorter.sort(&centers, [0.5, 0.5, 3.0])[0], 0);
    }

    #[test]
    fn cached_viewpoint_handles_negative_and_large_coordinates() {
        let origin = [-30_000_000, -64, 30_000_000];
        let mut state = SortState::new(vec![[0.5; 3]]);
        let mut sorter = Sorter::default();
        let camera = [-29_999_999.5, -63.5, 30_000_000.5];
        assert!(state.update(camera, origin, &mut sorter).is_some());
        assert!(state.update(camera, origin, &mut sorter).is_none());
        assert!(
            state
                .update([-29_999_999.1, -63.1, 30_000_000.9], origin, &mut sorter)
                .is_none()
        );
        assert!(
            state
                .update([-29_999_998.9, -63.1, 30_000_000.9], origin, &mut sorter)
                .is_some()
        );
    }

    #[test]
    fn large_tiles_resort_inside_the_same_section_viewpoint() {
        let mut state = SortState::new(vec![[32.0, 0.0, 0.0], [96.0, 0.0, 0.0]]);
        let mut sorter = Sorter::default();
        assert_eq!(
            state
                .update_lod([48.0, 20.0, 20.0], [0; 3], &mut sorter)
                .unwrap()[0],
            4
        );
        assert!(
            state
                .update_lod([48.5, 20.0, 20.0], [0; 3], &mut sorter)
                .is_none()
        );
        // Both views have positive x/y/z signs relative to the first section,
        // so the section-sized shortcut would incorrectly keep the old order.
        assert_eq!(
            state
                .update_lod([80.0, 20.0, 20.0], [0; 3], &mut sorter)
                .unwrap()[0],
            0
        );
    }

    #[test]
    fn coplanar_flat_ocean_emits_once_and_never_resorts() {
        let mut centers = Vec::with_capacity(512);
        let y = 14.0 + 0.888_888_9 - 0.001;
        for z in 0..16 {
            for x in 0..16 {
                let c = [x as f32 + 0.5, y, z as f32 + 0.5];
                // Top and back-facing underside share the same center.
                centers.push(c);
                centers.push(c);
            }
        }
        let mut state = SortState::new(centers);
        assert!(matches!(state.centers, Centers::Coplanar(512)));
        assert!(!state.is_sorted());
        let mut sorter = Sorter::default();
        let first = state
            .update([8.5, 70.0, 8.5], [0, 48, 0], &mut sorter)
            .expect("first frame uploads identity indices");
        assert_eq!(first.len(), 512 * 6);
        assert_eq!(&first[..12], &[0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4]);
        assert!(state.is_sorted());
        // Crossing blocks, sections, and moving above/below never resorts.
        for cam in [
            [9.5, 70.0, 8.5],
            [8.5, 50.0, 8.5],
            [-40.0, 100.0, 120.0],
            [0.1, 62.0, 0.1],
        ] {
            assert!(state.update(cam, [0, 48, 0], &mut sorter).is_none());
        }
        // Dropping a frame (#13624) forces the identity indices to be uploaded again.
        state.forget();
        assert!(!state.is_sorted());
        let reuploaded = state
            .update([0.1, 62.0, 0.1], [0, 48, 0], &mut sorter)
            .expect("forget must trigger a fresh upload");
        assert_eq!(reuploaded.len(), 512 * 6);
        assert!(state.is_sorted());
    }

    #[test]
    fn vertical_faces_at_same_y_still_resort() {
        // Two parallel vertical glass faces in the same Y layer share `y = 0.5`.
        let mut state = SortState::new(vec![[0.5, 0.5, 0.0], [0.5, 0.5, 1.0]]);
        assert!(matches!(state.centers, Centers::Compact(_)));
        let mut sorter = Sorter::default();
        assert_eq!(
            state.update([0.5, 0.5, -2.0], [0; 3], &mut sorter).unwrap()[0],
            4
        );
        assert_eq!(
            state.update([0.5, 0.5, 3.0], [0; 3], &mut sorter).unwrap()[0],
            0
        );
        state.forget();
        assert!(!state.is_sorted());
        assert_eq!(
            state.update([0.5, 0.5, 3.0], [0; 3], &mut sorter).unwrap()[0],
            0
        );
    }

    fn reference_sort(centers: &[[f32; 3]], camera: [f32; 3]) -> Vec<u32> {
        let mut reference: Vec<(u32, f32)> = centers
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let dx = camera[0] - c[0];
                let dy = camera[1] - c[1];
                let dz = camera[2] - c[2];
                (i as u32, dx * dx + (dy * dy + dz * dz))
            })
            .collect();
        reference.sort_by(|a, b| b.1.total_cmp(&a.1));
        reference
            .iter()
            .flat_map(|&(q, _)| {
                let v = q * 4;
                [v, v + 1, v + 2, v + 2, v + 3, v]
            })
            .collect()
    }

    #[test]
    fn radix_and_compact_sort_match_stable_reference() {
        let mut centers = Vec::with_capacity(512);
        for i in 0..512u32 {
            // Mix distinct distances, exact ties, 1/16 model offsets, and 1/2048 steps across Y layers.
            let x = (i % 16) as f32 + 0.5;
            let y = ((i / 16) % 4) as f32
                + match i % 7 {
                    0 => 1.0 / 2048.0,
                    1 => 1.0 / 16.0,
                    _ => 0.5,
                };
            let z = ((i / 64) % 8) as f32 + 0.5;
            centers.push([x, y, z]);
        }
        assert!(centers.len() >= RADIX_THRESHOLD);
        let mut sorter = Sorter::default();
        let mut state = SortState::new(centers.clone());
        assert!(matches!(state.centers, Centers::Compact(_)));

        // Include a near-zero camera where adding `COMPACT_OFFSET` to `camera` would lose precision.
        for camera in [[7.25f32, 2.75, 3.125], [0.1f32, 0.0, 0.0]] {
            let expected = reference_sort(&centers, camera);
            assert_eq!(sorter.sort(&centers, camera), expected.as_slice());

            let got = state
                .update(camera.map(f64::from), [0; 3], &mut sorter)
                .unwrap();
            assert_eq!(got, expected.as_slice());
        }
    }

    #[test]
    fn packed_sort_under_radix_threshold_matches_stable_reference() {
        let count_ties = |pts: &[[f32; 3]], cam: [f32; 3]| -> usize {
            let mut d2: Vec<u32> = pts
                .iter()
                .map(|c| {
                    let [dx, dy, dz] = [cam[0] - c[0], cam[1] - c[1], cam[2] - c[2]];
                    (dx * dx + (dy * dy + dz * dz)).to_bits()
                })
                .collect();
            d2.sort_unstable();
            d2.windows(2).filter(|w| w[0] == w[1]).count()
        };

        // 100 dyadic quads (< RADIX_THRESHOLD): 16 four-way symmetric equidistant groups around
        // `[8.0, 4.0, 8.0]` (64 quads) plus 18 co-located duplicate pairs (36 quads).
        let mut centers = Vec::with_capacity(100);
        for g in 0..16u32 {
            let dx = (g % 4) as f32 + 1.0 / 16.0;
            let dy = (g / 4) as f32 + 1.0 / 2048.0;
            let dz = ((g * 3) % 5) as f32 + 0.5;
            centers.push([8.0 + dx, 4.0 + dy, 8.0 + dz]);
            centers.push([8.0 - dx, 4.0 + dy, 8.0 - dz]);
            centers.push([8.0 + dx, 4.0 - dy, 8.0 - dz]);
            centers.push([8.0 - dx, 4.0 - dy, 8.0 + dz]);
        }
        for p in 0..18u32 {
            let c = [
                (p % 6) as f32 + 3.0 / 16.0,
                (p / 6) as f32 + 5.0 / 2048.0,
                ((p * 5) % 7) as f32 + 0.5,
            ];
            centers.push(c);
            centers.push(c);
        }
        assert_eq!(centers.len(), 100);
        assert!(centers.len() < RADIX_THRESHOLD);
        assert!(count_ties(&centers, [8.0, 4.0, 8.0]) >= 66);
        assert!(count_ties(&centers, [0.1, 0.0, 0.0]) >= 18);

        let mut sorter = Sorter::default();
        let mut state = SortState::new(centers.clone());
        assert!(matches!(state.centers, Centers::Compact(_)));

        for camera in [[8.0f32, 4.0, 8.0], [0.1, 0.0, 0.0], [15.75, 1.25, 2.5]] {
            let expected = reference_sort(&centers, camera);
            assert_eq!(sorter.sort(&centers, camera), expected.as_slice());

            let got = state
                .update(camera.map(f64::from), [0; 3], &mut sorter)
                .unwrap();
            assert_eq!(got, expected.as_slice());
        }

        // Also cover the `< RADIX_THRESHOLD` packed path on `Centers::Full` (non-dyadic `k / 9`)
        // with exact co-located ties.
        let mut flowing = Vec::with_capacity(100);
        for i in 0..50u32 {
            let c = [
                (i % 10) as f32 + 0.5,
                ((i / 10) % 4) as f32 + ((i % 8) as f32 + 1.0) / 9.0 - 0.001,
                ((i * 3) % 10) as f32 + 0.5,
            ];
            flowing.push(c);
            flowing.push(c);
        }
        assert_eq!(flowing.len(), 100);
        assert!(flowing.len() < RADIX_THRESHOLD);
        assert!(count_ties(&flowing, [7.25, 2.75, 1.5]) >= 50);

        let mut flow_state = SortState::new(flowing.clone());
        assert!(matches!(flow_state.centers, Centers::Full(_)));
        for camera in [[7.25f32, 2.75, 1.5], [0.1, 0.0, 0.0]] {
            let expected = reference_sort(&flowing, camera);
            let got = flow_state
                .update(camera.map(f64::from), [0; 3], &mut sorter)
                .unwrap();
            assert_eq!(got, expected.as_slice());
        }
    }

    #[test]
    fn non_dyadic_centers_stay_full_and_preserve_exact_order() {
        // Both 1.0001 and 1.0002 quantize to 1.0 on a 1/2048 grid; requiring exact round-trip
        // keeps them in `Centers::Full` so full-precision distance ordering is preserved.
        let mut state = SortState::new(vec![[1.0001, 0.5, 0.5], [1.0002, 0.5, 0.5]]);
        assert!(matches!(state.centers, Centers::Full(_)));
        let mut sorter = Sorter::default();
        let indices = state
            .update([0.0, 0.5, 0.5], [0; 3], &mut sorter)
            .expect("first frame sorts");
        assert_eq!(indices, &[4, 5, 6, 6, 7, 4, 0, 1, 2, 2, 3, 0]);

        // Flowing fluid heights (`k / 9`) and `0.001` epsilon offsets also stay on `Centers::Full`
        // and match the stable full-precision reference through the radix sort path.
        let mut flowing = Vec::with_capacity(256);
        for i in 0..256u32 {
            let x = (i % 16) as f32 + 0.5;
            let y = ((i / 16) % 4) as f32 + ((i % 8) as f32 + 1.0) / 9.0 - 0.001;
            let z = ((i / 64) % 4) as f32 + 0.5;
            flowing.push([x, y, z]);
        }
        let camera = [7.25f32, 2.75, 1.5];
        let expected = reference_sort(&flowing, camera);

        let mut flow_state = SortState::new(flowing);
        assert!(matches!(flow_state.centers, Centers::Full(_)));
        let got = flow_state
            .update(camera.map(f64::from), [0; 3], &mut sorter)
            .unwrap();
        assert_eq!(got, expected.as_slice());
    }

    #[test]
    fn empty_section_follows_viewpoint_caching_with_no_allocations() {
        let empty_quads: &[[[f32; 3]; 4]] = &[];
        for mut state in [
            SortState::new(Vec::new()),
            SortState::from_quads(empty_quads, |&p| p),
        ] {
            assert!(matches!(&state.centers, Centers::Compact(c) if c.is_empty()));
            assert!(!state.is_sorted());

            let mut sorter = Sorter::default();
            let origin = [0, 0, 0];
            // Matches vanilla TranslucencyPointOfView / tools/test_translucency_parity.py `count = 0`:
            // diagonal views reuse cache until pov changes; aligned views also update on block crossings.
            let cases = [
                ([-32.5, -32.5, -32.5], true),  // first frame, pov = [-1, -1, -1]
                ([-32.1, -32.1, -32.1], false), // same block & diagonal pov
                ([-31.9, -31.9, -31.9], false), // block changes, still diagonal pov = [-1, -1, -1]
                ([-16.0, -16.0, -16.0], false), // still diagonal pov = [-1, -1, -1]
                ([0.0, 0.0, 0.0], true),        // enters section, pov = [0, 0, 0]
                ([0.9, 0.9, 0.9], false),       // same block inside section
                ([1.1, 1.1, 1.1], true),        // block crossing while axis-aligned
                ([16.0, 16.0, 16.0], true),     // exits to diagonal pov = [1, 1, 1]
                ([32.0, 32.0, 32.0], false),    // still diagonal pov = [1, 1, 1]
                ([8.0, 32.0, 32.0], true),      // enters axis-aligned pov = [0, 1, 1]
                ([8.0, 33.0, 32.0], true),      // block crossing while axis-aligned
            ];
            for (camera, expect_update) in cases {
                let result = state.update(camera, origin, &mut sorter);
                assert_eq!(result.is_some(), expect_update, "camera {camera:?}");
                if let Some(indices) = result {
                    assert!(indices.is_empty());
                }
            }
            assert!(state.is_sorted());
            // Empty sorts never allocate scratch or index buffers.
            assert_eq!(sorter.indices.capacity(), 0);
            assert_eq!(sorter.keys.capacity(), 0);
            assert_eq!(sorter.radix_a.capacity(), 0);
            assert_eq!(sorter.radix_b.capacity(), 0);

            let empty_indices: &[u32] = &[];
            state.forget();
            assert!(!state.is_sorted());
            assert_eq!(
                state.update([8.0, 33.0, 32.0], origin, &mut sorter),
                Some(empty_indices)
            );

            // `update_lod` also reports updates on block changes for empty sections.
            state.forget();
            assert_eq!(
                state.update_lod([0.5, 0.5, 0.5], origin, &mut sorter),
                Some(empty_indices)
            );
            assert!(
                state
                    .update_lod([0.9, 0.9, 0.9], origin, &mut sorter)
                    .is_none()
            );
            assert_eq!(
                state.update_lod([1.5, 0.5, 0.5], origin, &mut sorter),
                Some(empty_indices)
            );
        }
    }

    #[test]
    fn from_quads_rejects_sloped_flowing_water_with_shared_center_y() {
        // Sloped flowing-water tops along +X all share center y = 0.7 (not 0.5) and x/z = ix + 0.5,
        // but their vertex Y coordinates differ across corners (0.8 vs 0.6).
        let sloped: Vec<[[f32; 3]; 4]> = (0..16)
            .map(|z| {
                let fz = z as f32;
                [
                    [0.0, 0.8, fz],
                    [0.0, 0.8, fz + 1.0],
                    [1.0, 0.6, fz + 1.0],
                    [1.0, 0.6, fz],
                ]
            })
            .collect();
        let mut state = SortState::from_quads(&sloped, |&p| p);
        assert!(matches!(state.centers, Centers::Full(_)));

        let mut sorter = Sorter::default();
        assert_eq!(
            state.update([0.5, 0.7, -4.0], [0; 3], &mut sorter).unwrap()[0],
            15 * 4
        );
        assert_eq!(
            state.update([0.5, 0.7, 20.0], [0; 3], &mut sorter).unwrap()[0],
            0
        );

        // Dyadic sloped quads (center y = 0.625) stay Compact rather than Coplanar.
        let dyadic_sloped = [
            [
                [0.0, 0.75, 0.0],
                [0.0, 0.75, 1.0],
                [1.0, 0.5, 1.0],
                [1.0, 0.5, 0.0],
            ],
            [
                [0.0, 0.75, 1.0],
                [0.0, 0.75, 2.0],
                [1.0, 0.5, 2.0],
                [1.0, 0.5, 1.0],
            ],
        ];
        let dyadic_state = SortState::from_quads(&dyadic_sloped, |&p| p);
        assert!(matches!(dyadic_state.centers, Centers::Compact(_)));

        // Saddle quad whose opposite corners p0/p2 share Y = 0.75 while p1/p3 differ.
        let saddle = [[
            [0.0, 0.75, 0.0],
            [0.0, 0.875, 1.0],
            [1.0, 0.75, 1.0],
            [1.0, 0.625, 0.0],
        ]];
        let saddle_state = SortState::from_quads(&saddle, |&p| p);
        assert!(matches!(saddle_state.centers, Centers::Compact(_)));
    }

    #[test]
    fn from_quads_handles_flat_planes_and_vertical_faces() {
        let y = 14.0 + 0.888_888_9 - 0.001;
        let mut ocean = Vec::with_capacity(512);
        for z in 0..16 {
            for x in 0..16 {
                let (fx, fz) = (x as f32, z as f32);
                let top = [
                    [fx, y, fz],
                    [fx, y, fz + 1.0],
                    [fx + 1.0, y, fz + 1.0],
                    [fx + 1.0, y, fz],
                ];
                // Shader-pack rotated diagonal and backward underside share the exact same y plane.
                let back = [top[1], top[0], top[3], top[2]];
                ocean.push(top);
                ocean.push(back);
            }
        }
        let mut ocean_state = SortState::from_quads(&ocean, |&p| p);
        assert!(matches!(ocean_state.centers, Centers::Coplanar(512)));
        let mut sorter = Sorter::default();
        assert_eq!(
            ocean_state
                .update([8.5, 70.0, 8.5], [0, 48, 0], &mut sorter)
                .unwrap()
                .len(),
            512 * 6
        );
        assert!(
            ocean_state
                .update([24.5, 70.0, 8.5], [0, 48, 0], &mut sorter)
                .is_none()
        );

        // Flat horizontal quads at y = 0.5 have all four vertices on y = 0.5 -> Coplanar.
        let flat_half = [
            [
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 1.0],
                [1.0, 0.5, 1.0],
                [1.0, 0.5, 0.0],
            ],
            [
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 1.0],
                [2.0, 0.5, 1.0],
                [2.0, 0.5, 0.0],
            ],
        ];
        assert!(matches!(
            SortState::from_quads(&flat_half, |&p| p).centers,
            Centers::Coplanar(2)
        ));

        // Vertical glass faces in y in [0, 1] (center y = 0.5) are rejected by vertex Y check.
        let vertical = [
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
        ];
        let mut vert_state = SortState::from_quads(&vertical, |&p| p);
        assert!(matches!(vert_state.centers, Centers::Compact(_)));
        assert_eq!(
            vert_state
                .update([0.5, 0.5, -2.0], [0; 3], &mut sorter)
                .unwrap()[0],
            4
        );
        assert_eq!(
            vert_state
                .update([0.5, 0.5, 3.0], [0; 3], &mut sorter)
                .unwrap()[0],
            0
        );
    }
}
