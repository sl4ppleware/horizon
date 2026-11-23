//! Sections a single-player server thread sends its own window, shared
//! instead of copied. The server and the window each held every column they
//! both see: the server its level, the window what the chunk packets decoded
//! to, the same blocks, biomes and light. When the server sends a column to a
//! player on its memory connection it offers the column's sections here, as
//! `Weak` references; the window's decoder takes each one whose blocks,
//! biomes and light are the same as what it decoded, and drops its own copy.
//!
//! Nothing changes what either side sees. The packet is still sent and
//! decoded, and a section is taken only when it equals the decoded one. A
//! `Weak` keeps no section alive, and a server edit to an offered section
//! moves it (`Arc::make_mut`), which the window's `Weak` no longer reaches;
//! once taken, a section either side edits is copied for that side alone.
//!
//! The server also hands its window the columns it changed in bulk (`World::edit_runs`) without sending them: ones
//! the window doesn't hold, whose far terrain still draws them as the window last saw them (`changed`,
//! `take_changed`).
use super::chunks::DirectWire;
use crate::chunk::{Chunk, ColumnLayout, Section};
use crate::fast_hash::FxHashMap;
use horizon_protocol::{
    chunk::BlockEntityInfo,
    packets::play::{self, LevelChunkWithLight},
};
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

/// Columns offered and not taken yet before the dead ones are swept.
const SWEEP: usize = 1 << 13;

/// Maximum live in-flight single-player direct-handoff columns kept at once.
/// Fits a full RD-32 initial burst (4,225 columns) with headroom while staying
/// strictly bounded; once full, new offers return `None` so the server falls
/// back cleanly to full wire bytes without evicting live in-flight stubs.
const DIRECT_MAX: usize = 1 << 13;

/// Sentinel heightmap type id carrying a direct single-player handoff token.
/// Vanilla `Heightmap.Types` ids are non-negative (`0..=5`), so `-1` never
/// collides with a real heightmap, and `chunk.buffer` is empty (`0` bytes),
/// which no valid dimension (`height >= 16`) can produce on the wire.
pub const DIRECT_HEIGHTMAP_KIND: i32 = -1;

struct Offer {
    /// The minimum block Y of `sections[0]`.
    min_y: i32,
    sections: Vec<Weak<Section>>,
}

static OFFERED: Mutex<Option<FxHashMap<(i32, i32), Offer>>> = Mutex::new(None);
static RECORDING_COUNT: AtomicUsize = AtomicUsize::new(0);
static NEXT_DIRECT_TOKEN: AtomicU64 = AtomicU64::new(1);
static DIRECT_OFFERED: Mutex<Option<FxHashMap<u64, Arc<DirectColumn>>>> = Mutex::new(None);

/// Record that a local integrated session has started packet recording,
/// disabling direct chunk handoff so the recorder receives full wire bytes.
pub fn begin_recording() {
    RECORDING_COUNT.fetch_add(1, Ordering::AcqRel);
}

/// Release a recording claim acquired via [`begin_recording`].
pub fn end_recording() {
    let _ = RECORDING_COUNT.try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
        Some(n.saturating_sub(1))
    });
}

/// Whether any local integrated session is actively recording play packets.
pub fn is_recording() -> bool {
    RECORDING_COUNT.load(Ordering::Acquire) != 0
}

/// A server-side column handed directly to the local single-player client,
/// bypassing wire encode -> Pipe -> wire decode -> `same_as` memcmp while
/// retaining lazy wire-materialization metadata for recorder or format fallback.
pub struct DirectColumn {
    x: i32,
    z: i32,
    layout: ColumnLayout,
    sections: Vec<Arc<Section>>,
    shared_generation: u64,
    owner: Weak<()>,
    wire: DirectWire,
}

impl DirectColumn {
    pub(super) fn new(
        x: i32,
        z: i32,
        layout: ColumnLayout,
        sections: Vec<Arc<Section>>,
        shared_generation: u64,
        owner: Weak<()>,
        wire: DirectWire,
    ) -> Self {
        Self {
            x,
            z,
            layout,
            sections,
            shared_generation,
            owner,
            wire,
        }
    }

    pub fn layout(&self) -> ColumnLayout {
        self.layout
    }

    pub fn level_min_y(&self) -> i32 {
        self.wire.level_min_y()
    }

    pub fn level_sections(&self) -> usize {
        self.wire.level_sections()
    }

    pub fn has_skylight(&self) -> bool {
        self.wire.has_skylight()
    }

    pub fn shared_generation(&self) -> u64 {
        self.shared_generation
    }

    pub fn biomes_identity(&self, biome_map: &[u16]) -> bool {
        self.wire.biomes_identity(biome_map)
    }

    pub fn sections(&self) -> &[Arc<Section>] {
        &self.sections
    }

    /// Materialize the full `LevelChunkWithLight` packet when the client's
    /// `ChunkFormat` cannot adopt the direct column directly.
    pub fn materialize_packet(&self, block_entities: &[BlockEntityInfo]) -> LevelChunkWithLight {
        self.wire
            .materialize_packet(self.x, self.z, &self.sections, block_entities)
    }
}

/// Extract the direct-handoff token from a minimal stub `LevelChunkWithLight`
/// packet, or `None` for any normal wire-encoded chunk packet.
pub fn direct_token(packet: &LevelChunkWithLight) -> Option<u64> {
    if !packet.chunk.buffer.is_empty() {
        return None;
    }
    match packet.chunk.heightmaps.as_slice() {
        [(kind, longs)] if *kind == DIRECT_HEIGHTMAP_KIND && longs.len() == 1 => {
            Some(longs[0] as u64)
        }
        _ => None,
    }
}

/// Register a [`DirectColumn`] for direct single-player handoff and return its
/// unique token, or `None` if the bounded in-flight table is full.
pub fn offer_direct(column: DirectColumn) -> Option<u64> {
    let Ok(mut guard) = DIRECT_OFFERED.lock() else {
        return None;
    };
    let map = guard.get_or_insert_with(FxHashMap::default);
    if map.len() >= DIRECT_MAX / 2 {
        map.retain(|_, c| c.owner.strong_count() > 0);
    }
    if map.len() >= DIRECT_MAX {
        return None;
    }
    let token = NEXT_DIRECT_TOKEN.fetch_add(1, Ordering::Relaxed);
    map.insert(token, Arc::new(column));
    Some(token)
}

/// Borrow a direct-handoff column without removing it (used by background
/// `ChunkDecoder` workers so a subsequent `ChunkFormat` change or recorder
/// capture still sees the entry until `release_direct` runs on retirement).
pub fn peek_direct(token: u64, x: i32, z: i32) -> Option<Arc<DirectColumn>> {
    let Ok(guard) = DIRECT_OFFERED.lock() else {
        return None;
    };
    let col = guard.as_ref()?.get(&token)?;
    (col.x == x && col.z == z).then(|| Arc::clone(col))
}

/// Remove and return a direct-handoff column when coordinates match.
pub fn take_direct(token: u64, x: i32, z: i32) -> Option<Arc<DirectColumn>> {
    let Ok(mut guard) = DIRECT_OFFERED.lock() else {
        return None;
    };
    let map = guard.as_mut()?;
    let matches = map.get(&token).is_some_and(|c| c.x == x && c.z == z);
    if matches { map.remove(&token) } else { None }
}

/// Release a previously peeked direct-handoff entry once the client has
/// retired, ignored, or dropped the corresponding chunk packet.
pub fn release_direct(packet: &LevelChunkWithLight) {
    let Some(token) = direct_token(packet) else {
        return;
    };
    let Ok(mut guard) = DIRECT_OFFERED.lock() else {
        return;
    };
    let Some(map) = guard.as_mut() else {
        return;
    };
    if map
        .get(&token)
        .is_some_and(|c| c.x == packet.x && c.z == packet.z)
    {
        map.remove(&token);
    }
}

/// Materialize full wire bytes for an in-flight direct-handoff stub if packet
/// recording starts while the stub is already buffered in the local `Pipe`.
pub(crate) fn materialize_for_recorder(
    packet: &LevelChunkWithLight,
) -> Option<(play::Clientbound, Vec<u8>)> {
    let token = direct_token(packet)?;
    let col = peek_direct(token, packet.x, packet.z)?;
    let (full, payload) =
        col.wire
            .materialize_payload(col.x, col.z, &col.sections, &packet.chunk.block_entities);
    Some((play::Clientbound::LevelChunkWithLight(full), payload))
}

/// Drop direct-handoff entries whose owning [`super::chunks::ChunkEncoder`] has
/// been dropped, without disturbing entries belonging to any other live world.
pub(super) fn sweep_dead_direct() {
    let Ok(mut guard) = DIRECT_OFFERED.lock() else {
        return;
    };
    if let Some(map) = guard.as_mut() {
        map.retain(|_, c| c.owner.strong_count() > 0);
    }
}

/// Columns changed in bulk and not sent to the window, the latest of each, until the window takes them.
static CHANGED: Mutex<Option<FxHashMap<(i32, i32), Arc<Chunk>>>> = Mutex::new(None);
/// The most columns waiting for the window: past them the later changes are left out (a window that stopped
/// taking them, as a server with no window of its own never does, holds no more than this).
const CHANGED_MAX: usize = 1 << 13;

/// The server changed `chunk` in bulk and didn't send it to the player on its memory connection: the window's far
/// terrain is to summarize it again.
pub fn changed(chunk: &Arc<Chunk>) {
    let Ok(mut changed) = CHANGED.lock() else {
        return;
    };
    let changed = changed.get_or_insert_with(Default::default);
    let key = (chunk.x, chunk.z);
    if changed.len() < CHANGED_MAX || changed.contains_key(&key) {
        changed.insert(key, chunk.clone());
    }
}

/// The columns changed in bulk since the window last asked (`changed`), each once, as it was last changed.
pub fn take_changed() -> Vec<Arc<Chunk>> {
    match CHANGED.lock() {
        Ok(mut changed) => changed
            .take()
            .map(|c| c.into_values().collect())
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// The server sends `chunk` to a player on its memory connection.
pub fn offer(chunk: &Chunk) {
    let Ok(mut offered) = OFFERED.lock() else {
        return;
    };
    let offered = offered.get_or_insert_with(Default::default);
    if offered.len() >= SWEEP {
        offered.retain(|_, o| o.sections.iter().any(|w| w.strong_count() > 0));
        if offered.len() >= SWEEP / 2 {
            offered.clear();
        }
    }
    offered.insert(
        (chunk.x, chunk.z),
        Offer {
            min_y: chunk.min_y(),
            sections: chunk.sections.iter().map(Arc::downgrade).collect(),
        },
    );
}

/// The window decoded `chunk` from a packet: each of its sections the same
/// as the server's offered one at that height becomes the server's. Returns
/// how many did.
pub fn take(chunk: &mut Chunk) -> usize {
    let offer = match OFFERED.lock() {
        Ok(mut offered) => offered.as_mut().and_then(|o| o.remove(&(chunk.x, chunk.z))),
        Err(_) => None,
    };
    let Some(offer) = offer else {
        return 0;
    };
    let offset = (chunk.min_y() - offer.min_y) >> 4;
    let mut taken = 0;
    for (i, section) in chunk.sections.iter_mut().enumerate() {
        let Some(theirs) = usize::try_from(i as i32 + offset)
            .ok()
            .and_then(|j| offer.sections.get(j))
            .and_then(Weak::upgrade)
        else {
            continue;
        };
        if Arc::ptr_eq(section, &theirs) {
            continue;
        }
        if theirs.same_as(section) {
            *section = theirs;
            taken += 1;
        }
    }
    taken
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Light, MIN_Y};

    fn column(x: i32) -> Chunk {
        let mut chunk = Chunk::empty(x, 7);
        chunk.set(1, MIN_Y + 3, 2, 5);
        chunk.set(4, 70, 4, 9);
        Arc::make_mut(&mut chunk.sections[2]).light =
            Light::packed(Box::new(std::array::from_fn(|i| {
                if i % 3 == 0 { 0xF2 } else { 0x70 }
            })));
        chunk
    }

    #[test]
    fn equal_sections_are_shared_and_edits_stay_apart() {
        let server = column(901);
        offer(&server);
        let mut window = column(901);
        let taken = take(&mut window);
        // The two edited sections and the lit one; the uniform ones were
        // already the same shared allocation.
        assert_eq!(taken, 3);
        for (a, b) in server.sections.iter().zip(&window.sections) {
            assert!(Arc::ptr_eq(a, b));
        }
        let mut edited = window.clone();
        edited.set(1, MIN_Y + 3, 2, 6);
        assert_eq!(server.get(1, MIN_Y + 3, 2), 5);
        assert_eq!(edited.get(1, MIN_Y + 3, 2), 6);
        // Taken once: a second decode of the column has nothing to take.
        assert_eq!(take(&mut column(901)), 0);
    }

    #[test]
    fn differing_or_edited_sections_are_kept() {
        let mut server = column(902);
        offer(&server);
        // Edited after the offer: the server's section moved.
        server.set(1, MIN_Y + 3, 2, 8);
        let mut window = column(902);
        window.set(4, 70, 4, 10);
        let taken = take(&mut window);
        // Only the lit section is still the same on both sides.
        assert_eq!(taken, 1);
        assert_eq!(window.get(1, MIN_Y + 3, 2), 5);
        assert_eq!(window.get(4, 70, 4), 10);
    }

    #[test]
    fn columns_changed_unsent_are_taken_once_as_last_changed() {
        let first = Arc::new(column(903));
        let mut again = column(903);
        again.set(1, MIN_Y + 3, 2, 9);
        let other = Arc::new(column(904));
        changed(&first);
        changed(&other);
        changed(&Arc::new(again));
        let mut taken = take_changed();
        taken.sort_by_key(|c| (c.x, c.z));
        assert_eq!(taken.len(), 2);
        assert_eq!(
            taken[0].get(1, MIN_Y + 3, 2),
            9,
            "the column as it was last changed"
        );
        assert!(Arc::ptr_eq(&taken[1], &other));
        assert!(take_changed().is_empty(), "taken once");
    }

    #[test]
    fn direct_handoff_peek_take_release_and_owner_sweep() {
        use super::super::chunks::ChunkEncoder;
        use horizon_protocol::registry::Builtin;

        let states = [
            (Arc::<str>::from("minecraft:air"), false),
            (Arc::<str>::from("minecraft:stone"), false),
        ];
        let biomes = ["minecraft:plains".to_string()];
        let encoder_a = ChunkEncoder::new(states.clone(), &Builtin::default(), &biomes, &biomes);
        let encoder_b = ChunkEncoder::new(states, &Builtin::default(), &biomes, &biomes);

        let mut world = crate::world::World::new(horizon_world::registry::BlockStateId(0), 0);
        let mut c1 = Chunk::empty(701, -702);
        c1.set(1, MIN_Y + 4, 2, 1);
        let mut c2 = Chunk::empty(703, -704);
        c2.set(3, MIN_Y + 8, 4, 1);
        world.chunks.insert((701, -702), Arc::new(c1));
        world.chunks.insert((703, -704), Arc::new(c2));

        let d1 = encoder_a
            .direct_column(&world.chunks[&(701, -702)], &world, &|_, _| None)
            .unwrap();
        let t1 = offer_direct(d1).unwrap();
        // Coordinate mismatch never steals or reads the entry.
        assert!(peek_direct(t1, 701, 0).is_none());
        assert!(take_direct(t1, 0, -702).is_none());
        // Peek leaves the entry intact; take removes it once.
        let peeked = peek_direct(t1, 701, -702).unwrap();
        assert_eq!(
            peeked.sections().len(),
            world.chunks[&(701, -702)].sections.len()
        );
        let taken = take_direct(t1, 701, -702).unwrap();
        assert!(Arc::ptr_eq(&peeked, &taken));
        assert!(peek_direct(t1, 701, -702).is_none());
        assert!(take_direct(t1, 701, -702).is_none());

        // `release_direct` removes a peeked stub entry without affecting other tokens.
        let d1_again = encoder_a
            .direct_column(&world.chunks[&(701, -702)], &world, &|_, _| None)
            .unwrap();
        let t1_again = offer_direct(d1_again).unwrap();
        let stub_body = encoder_a.direct_stub_body(
            &world.chunks[&(701, -702)],
            &world.block_entities,
            t1_again,
        );
        let mut r = horizon_protocol::buf::Reader::new(&stub_body);
        let id = r.varint().unwrap();
        let play::Clientbound::LevelChunkWithLight(stub_pkt) =
            play::Clientbound::decode(id, r.rest()).unwrap()
        else {
            panic!("expected LevelChunkWithLight stub");
        };
        assert_eq!(direct_token(&stub_pkt), Some(t1_again));
        assert!(peek_direct(t1_again, 701, -702).is_some());
        release_direct(&stub_pkt);
        assert!(peek_direct(t1_again, 701, -702).is_none());

        // Dropping `encoder_a` sweeps only `encoder_a`'s dead entries while leaving
        // `encoder_b`'s live entries intact.
        let t_a = offer_direct(
            encoder_a
                .direct_column(&world.chunks[&(701, -702)], &world, &|_, _| None)
                .unwrap(),
        )
        .unwrap();
        let t_b = offer_direct(
            encoder_b
                .direct_column(&world.chunks[&(703, -704)], &world, &|_, _| None)
                .unwrap(),
        )
        .unwrap();
        drop(encoder_a);
        assert!(peek_direct(t_a, 701, -702).is_none());
        assert!(take_direct(t_b, 703, -704).is_some());
        drop(encoder_b);

        // Recording flag refcount turns on while claimed and releases cleanly.
        begin_recording();
        assert!(is_recording());
        end_recording();
    }
}
