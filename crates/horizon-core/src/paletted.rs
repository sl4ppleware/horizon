//! Resident `PalettedContainer` of Java Edition 26.3: single-value, linear and
//! hash-map palettes with vanilla's bit-width thresholds, and the global
//! palette beyond them, over `SimpleBitStorage` words (values never span
//! longs). One container type serves blocks (16^3 entries) and biomes (4^3).
//!
//! Growth follows `PalettedContainer.onResize`: a full palette asks for one
//! more bit, the new configuration is built by re-adding the distinct values
//! in index order, and the value that did not fit is added last. Palettes
//! never shrink while resident; `pack` writes the canonical saved form.

use crate::nbt::Tag;
use crate::storage::{locate, mask};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `Strategy.createForBlockStates`: 4096 entries, linear palettes up to 4
    /// bits and hash-map palettes from 5 to 8.
    Blocks,
    /// `Strategy.createForBiomes`: 64 entries, linear palettes up to 3 bits.
    Biomes,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strategy {
    pub kind: Kind,
    /// `ceillog2` of the registry size: the width of the global palette.
    pub global_bits: u8,
}
/// `getConfigurationForBitCount`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Config {
    Single,
    /// Palette of up to `1 << bits` entries, stored in `bits` bits.
    List(u8),
    Global,
}
impl Strategy {
    pub const fn blocks(global_bits: u8) -> Self {
        Self {
            kind: Kind::Blocks,
            global_bits,
        }
    }
    pub const fn biomes(global_bits: u8) -> Self {
        Self {
            kind: Kind::Biomes,
            global_bits,
        }
    }
    pub const fn entries(self) -> usize {
        match self.kind {
            Kind::Blocks => 4096,
            Kind::Biomes => 64,
        }
    }
    fn config(self, bits: u8) -> Config {
        match (self.kind, bits) {
            (_, 0) => Config::Single,
            (Kind::Blocks, 1..=4) => Config::List(4),
            (Kind::Blocks, 5..=8) => Config::List(bits),
            (Kind::Biomes, 1..=3) => Config::List(bits),
            _ => Config::Global,
        }
    }
    /// Whether a container written with leading byte `bits` lists its palette
    /// (all but the global one).
    pub fn wire_has_palette(self, bits: u8) -> bool {
        self.config(bits) != Config::Global
    }
    /// Bits of the storage a configuration keeps in memory.
    fn storage_bits(self, config: Config) -> u8 {
        match config {
            Config::Single => 0,
            Config::List(bits) => bits,
            Config::Global => self.global_bits,
        }
    }
}
/// `Mth.ceillog2`.
pub fn ceil_log2(n: usize) -> u8 {
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as u8
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PaletteError {
    /// A palette bigger than its configuration allows, or empty.
    BadPalette { len: usize },
    /// Storage with the wrong number of longs.
    BadStorage { expected: usize, actual: usize },
    /// A stored index past the palette.
    BadIndex { index: u32, palette: usize },
    /// The declared bit count does not match the palette.
    BadBits { declared: u8, expected: u8 },
}
impl std::fmt::Display for PaletteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PaletteError {}

/// Small palettes (all 2/3-bit biomes and most underground 4-bit block
/// sections) stay inline inside [`ListMeta`] so they avoid a second heap
/// allocation and a second pointer dereference on `get`. Nine entries fill
/// the exact 40-byte footprint needed alongside `Heap(Vec<u32>)`.
const INLINE_PALETTE_CAP: usize = 9;

#[derive(Clone, Debug)]
enum ListPalette {
    Inline {
        len: u8,
        entries: [u32; INLINE_PALETTE_CAP],
    },
    Heap(Vec<u32>),
}

impl ListPalette {
    fn from_single(a: u32) -> Self {
        let mut entries = [0u32; INLINE_PALETTE_CAP];
        entries[0] = a;
        Self::Inline { len: 1, entries }
    }

    fn from_pair(a: u32, b: u32) -> Self {
        let mut entries = [0u32; INLINE_PALETTE_CAP];
        entries[0] = a;
        entries[1] = b;
        Self::Inline { len: 2, entries }
    }

    fn from_triple(a: u32, b: u32, c: u32) -> Self {
        let mut entries = [0u32; INLINE_PALETTE_CAP];
        entries[0] = a;
        entries[1] = b;
        entries[2] = c;
        Self::Inline { len: 3, entries }
    }

    fn from_vec(mut vec: Vec<u32>, shrink: bool) -> Self {
        if vec.len() <= INLINE_PALETTE_CAP {
            let mut entries = [0u32; INLINE_PALETTE_CAP];
            entries[..vec.len()].copy_from_slice(&vec);
            Self::Inline {
                len: vec.len() as u8,
                entries,
            }
        } else {
            if shrink {
                vec.shrink_to_fit();
            }
            Self::Heap(vec)
        }
    }

    #[inline]
    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Inline { len, entries } => &entries[..usize::from(*len)],
            Self::Heap(vec) => vec,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Inline { len, .. } => usize::from(*len),
            Self::Heap(vec) => vec.len(),
        }
    }

    fn push(&mut self, value: u32) -> usize {
        match self {
            Self::Inline { len, entries } => {
                let idx = usize::from(*len);
                if idx < entries.len() {
                    entries[idx] = value;
                    *len += 1;
                    idx
                } else {
                    let mut vec = Vec::with_capacity(entries.len() * 2);
                    vec.extend_from_slice(entries);
                    vec.push(value);
                    *self = Self::Heap(vec);
                    idx
                }
            }
            Self::Heap(vec) => {
                let idx = vec.len();
                vec.push(value);
                idx
            }
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Inline { .. } => 0,
            Self::Heap(vec) => vec.capacity() * std::mem::size_of::<u32>(),
        }
    }
}
/// Words of `bits`-wide values: `entries.div_ceil(64 / bits)`.
pub fn storage_len(bits: u8, entries: usize) -> usize {
    if bits == 0 {
        0
    } else {
        entries.div_ceil(64 / usize::from(bits))
    }
}
fn read(words: &[u64], bits: u8, i: usize) -> u32 {
    let (word, shift) = locate(bits, i);
    ((words[word] >> shift) & mask(bits)) as u32
}
fn write(words: &mut [u64], bits: u8, i: usize, value: u32) {
    let (word, shift) = locate(bits, i);
    words[word] = (words[word] & !(mask(bits) << shift)) | (u64::from(value) << shift);
}
/// `word + add`, bit by bit: which bits carried out of their position. The
/// carry out of bit `i` is `a & b | (a | b) & !sum` for `sum = a + b`.
#[inline]
fn carry_out(word: u64, add: u64, sum: u64) -> u64 {
    (word & add) | ((word | add) & !sum)
}
/// The first of `count` `bits`-wide fields of `word` that is `len` or more.
fn first_bad(word: u64, count: usize, bits: usize, len: usize) -> PaletteError {
    for slot in 0..count {
        let index = ((word >> (slot * bits)) & ((1 << bits) - 1)) as u32;
        if index as usize >= len {
            return PaletteError::BadIndex {
                index,
                palette: len,
            };
        }
    }
    unreachable!("check_indices found no field past the palette")
}
/// Values below this are looked up in a table indexed by value once a
/// palette outgrows the scan: every block state and biome id of 26.3.
const DENSE_VALUES: u32 = 1 << 16;
thread_local! {
    /// `palette_ids`' table: 1 + the id of each value seen, 0 for the others.
    /// Only the entries of the palette being built are set, and they are
    /// cleared again before it returns.
    static DENSE_IDS: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}
/// The distinct `values` in first-occurrence order and each value's index in
/// them. A run of one value (most of a section) is matched once, a scan finds
/// up to 16 distinct values faster than any lookup, and past that a table
/// indexed by value finds them without hashing.
fn palette_ids(values: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut palette: Vec<u32> = Vec::new();
    let mut ids = Vec::with_capacity(values.len());
    let mut last: Option<(u32, u32)> = None;
    let mut rest = values;
    while let Some((&value, tail)) = rest.split_first() {
        let id = match last {
            Some((v, id)) if v == value => id,
            _ if palette.len() > 16 => break,
            _ => match palette.iter().position(|&v| v == value) {
                Some(id) => id as u32,
                None => {
                    palette.push(value);
                    palette.len() as u32 - 1
                }
            },
        };
        last = Some((value, id));
        ids.push(id);
        rest = tail;
    }
    if rest.is_empty() {
        return (palette, ids);
    }
    if palette.iter().chain(rest).all(|&v| v < DENSE_VALUES) {
        DENSE_IDS.with(|table| {
            let mut table = table.borrow_mut();
            for (id, &v) in palette.iter().enumerate() {
                let v = v as usize;
                if table.len() <= v {
                    table.resize(v + 1, 0);
                }
                table[v] = id as u32 + 1;
            }
            for &value in rest {
                let v = value as usize;
                if table.len() <= v {
                    table.resize(v + 1, 0);
                }
                let id = match table[v] {
                    0 => {
                        palette.push(value);
                        table[v] = palette.len() as u32;
                        palette.len() as u32 - 1
                    }
                    slot => slot - 1,
                };
                ids.push(id);
            }
            for &v in &palette {
                table[v as usize] = 0;
            }
        });
    } else {
        let mut lookup: std::collections::HashMap<u32, u32> = palette
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();
        for &value in rest {
            let id = *lookup.entry(value).or_insert_with(|| {
                palette.push(value);
                palette.len() as u32 - 1
            });
            ids.push(id);
        }
    }
    (palette, ids)
}
/// Words holding `ids` in `bits`-wide slots, accumulated word-by-word in
/// registers (one store per 64-bit word, zero per-entry `locate` calls).
fn pack_words(bits: u8, entries: usize, mut ids: impl Iterator<Item = u32>) -> Vec<u64> {
    if bits == 0 || entries == 0 {
        return Vec::new();
    }
    let bits_u = usize::from(bits);
    let per = 64 / bits_u;
    let word_count = entries.div_ceil(per);
    let mut words = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        let mut word = 0u64;
        let mut shift = 0usize;
        for _ in 0..per {
            let Some(id) = ids.next() else {
                break;
            };
            word |= u64::from(id) << shift;
            shift += bits_u;
        }
        words.push(word);
    }
    words
}

/// Slice-specialized word-wise packer with a 16-nibble unrolled 4-bit path.
fn pack_words_slice(bits: u8, ids: &[u32]) -> Vec<u64> {
    if bits == 0 || ids.is_empty() {
        return Vec::new();
    }
    if bits == 4 {
        let (chunks, rem) = ids.as_chunks::<16>();
        let mut words = Vec::with_capacity(chunks.len() + usize::from(!rem.is_empty()));
        for c in chunks {
            let word = u64::from(c[0])
                | (u64::from(c[1]) << 4)
                | (u64::from(c[2]) << 8)
                | (u64::from(c[3]) << 12)
                | (u64::from(c[4]) << 16)
                | (u64::from(c[5]) << 20)
                | (u64::from(c[6]) << 24)
                | (u64::from(c[7]) << 28)
                | (u64::from(c[8]) << 32)
                | (u64::from(c[9]) << 36)
                | (u64::from(c[10]) << 40)
                | (u64::from(c[11]) << 44)
                | (u64::from(c[12]) << 48)
                | (u64::from(c[13]) << 52)
                | (u64::from(c[14]) << 56)
                | (u64::from(c[15]) << 60);
            words.push(word);
        }
        if !rem.is_empty() {
            let mut word = 0u64;
            for (slot, &id) in rem.iter().enumerate() {
                word |= u64::from(id) << (slot * 4);
            }
            words.push(word);
        }
        return words;
    }
    let bits_u = usize::from(bits);
    let per = 64 / bits_u;
    let mut words = Vec::with_capacity(ids.len().div_ceil(per));
    for chunk in ids.chunks(per) {
        let mut word = 0u64;
        let mut shift = 0usize;
        for &id in chunk {
            word |= u64::from(id) << shift;
            shift += bits_u;
        }
        words.push(word);
    }
    words
}

/// Packs up to 64 1-bit palette indices directly into a single `u64`.
fn pack_word_1bit(ids: &[u32]) -> u64 {
    debug_assert!(ids.len() <= 64);
    let mut word = 0u64;
    for (i, &id) in ids.iter().enumerate() {
        word |= u64::from(id & 1) << i;
    }
    word
}

/// Spreads 32 1-bit entries into the even bit positions `0, 2, ..., 62` of a
/// 2-bit-per-entry `u64` word.
#[inline]
fn spread_u32_to_2bit(x: u32) -> u64 {
    let mut v = u64::from(x);
    v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
    v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
    v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    v = (v | (v << 2)) & 0x3333_3333_3333_3333;
    (v | (v << 1)) & 0x5555_5555_5555_5555
}

/// Words filled with `entries` copies of `value` at `bits` width (padding zero).
fn fill_words(bits: u8, entries: usize, value: u32) -> Vec<u64> {
    if bits == 0 || entries == 0 {
        return Vec::new();
    }
    let bits_u = usize::from(bits);
    let per = 64 / bits_u;
    let pattern = (0..per).fold(0u64, |w, slot| w | (u64::from(value) << (slot * bits_u)));
    let full = entries / per;
    let rem = entries - full * per;
    let mut words = Vec::with_capacity(full + usize::from(rem > 0));
    words.resize(full, pattern);
    if rem > 0 {
        let used = (1u64 << (rem * bits_u)) - 1;
        words.push(pattern & used);
    }
    words
}

/// Whether every live entry `0..entries` in `words` holds `id`.
fn all_words_equal_id(words: &[u64], bits: u8, entries: usize, id: u64) -> bool {
    let bits_u = usize::from(bits);
    let per = 64 / bits_u;
    let pattern = (0..per).fold(0u64, |word, slot| word | (id << (slot * bits_u)));
    let full = entries / per;
    let used_full = if per * bits_u == 64 {
        u64::MAX
    } else {
        (1u64 << (per * bits_u)) - 1
    };
    if words[..full].iter().any(|&w| (w & used_full) != pattern) {
        return false;
    }
    let tail = entries - full * per;
    tail == 0 || {
        let used = (1u64 << (tail * bits_u)) - 1;
        words[full] & used == pattern & used
    }
}

/// Visits every 4-bit nibble of `word` in slot order `0..16`.
#[inline(always)]
fn visit_4bit_word(word: u64, visit: &mut impl FnMut(usize)) {
    visit((word & 0xF) as usize);
    visit(((word >> 4) & 0xF) as usize);
    visit(((word >> 8) & 0xF) as usize);
    visit(((word >> 12) & 0xF) as usize);
    visit(((word >> 16) & 0xF) as usize);
    visit(((word >> 20) & 0xF) as usize);
    visit(((word >> 24) & 0xF) as usize);
    visit(((word >> 28) & 0xF) as usize);
    visit(((word >> 32) & 0xF) as usize);
    visit(((word >> 36) & 0xF) as usize);
    visit(((word >> 40) & 0xF) as usize);
    visit(((word >> 44) & 0xF) as usize);
    visit(((word >> 48) & 0xF) as usize);
    visit(((word >> 52) & 0xF) as usize);
    visit(((word >> 56) & 0xF) as usize);
    visit(((word >> 60) & 0xF) as usize);
}

/// Zero-temp-buffer canonical packer for list-paletted containers (`<= 256`
/// entries): discovers first-occurrence order in a single word-wise pass with
/// a stack `[u16; 256]` remap table, skipping bit-repacking entirely when the
/// stored words are already canonical at minimal width.
fn pack_list(strategy: Strategy, bits: u8, words: &[u64], list: &[u32]) -> Packed {
    let entries = strategy.entries();
    let mut remap = [u16::MAX; 256];
    let mut new_palette: Vec<u32> = Vec::with_capacity(list.len().min(16));
    let mut is_identity = true;
    let mut pad_dirty = 0u64;

    let mut visit = |old_id: usize| {
        if remap[old_id] == u16::MAX {
            let val = list[old_id];
            let new_id = match new_palette.iter().position(|&v| v == val) {
                Some(existing) => existing,
                None => {
                    let id = new_palette.len();
                    new_palette.push(val);
                    id
                }
            };
            remap[old_id] = new_id as u16;
            if new_id != old_id {
                is_identity = false;
            }
        }
    };

    if bits == 4 {
        if let Some((&first, rest)) = words.split_first() {
            visit_4bit_word(first, &mut visit);
            let mut prev = first;
            for &word in rest {
                if word == prev {
                    continue;
                }
                prev = word;
                visit_4bit_word(word, &mut visit);
            }
        }
    } else {
        let bits_u = usize::from(bits);
        let per = 64 / bits_u;
        let m = mask(bits);
        let full = entries / per;
        let rem = entries - full * per;
        let full_pad_mask = if per * bits_u < 64 {
            u64::MAX << (per * bits_u)
        } else {
            0
        };
        if let Some((&first, rest)) = words[..full].split_first() {
            pad_dirty |= first & full_pad_mask;
            let mut w = first;
            for _ in 0..per {
                visit((w & m) as usize);
                w >>= bits_u;
            }
            let mut prev = first;
            for &word in rest {
                if word == prev {
                    continue;
                }
                prev = word;
                pad_dirty |= word & full_pad_mask;
                let mut w = word;
                for _ in 0..per {
                    visit((w & m) as usize);
                    w >>= bits_u;
                }
            }
        }
        if rem > 0 {
            let tail = words[full];
            pad_dirty |= tail & (u64::MAX << (rem * bits_u));
            let mut w = tail;
            for _ in 0..rem {
                visit((w & m) as usize);
                w >>= bits_u;
            }
        }
    }

    let new_bits = match strategy.config(ceil_log2(new_palette.len())) {
        Config::Single => {
            return Packed {
                palette: new_palette,
                bits: 0,
                data: None,
            };
        }
        Config::List(b) => b,
        Config::Global => ceil_log2(new_palette.len()),
    };

    if is_identity && new_bits == bits {
        let mut data = words.to_vec();
        if pad_dirty != 0 {
            let bits_u = usize::from(bits);
            let per = 64 / bits_u;
            let full = entries / per;
            let rem = entries - full * per;
            if per * bits_u < 64 {
                let keep = (1u64 << (per * bits_u)) - 1;
                for w in &mut data[..full] {
                    *w &= keep;
                }
            }
            if rem > 0 {
                let keep = (1u64 << (rem * bits_u)) - 1;
                data[full] &= keep;
            }
        }
        return Packed {
            palette: new_palette,
            bits: new_bits,
            data: Some(data),
        };
    }

    if bits == 4 && new_bits == 4 {
        let mut data = Vec::with_capacity(words.len());
        for &word in words {
            let out = u64::from(remap[(word & 0xF) as usize])
                | (u64::from(remap[((word >> 4) & 0xF) as usize]) << 4)
                | (u64::from(remap[((word >> 8) & 0xF) as usize]) << 8)
                | (u64::from(remap[((word >> 12) & 0xF) as usize]) << 12)
                | (u64::from(remap[((word >> 16) & 0xF) as usize]) << 16)
                | (u64::from(remap[((word >> 20) & 0xF) as usize]) << 20)
                | (u64::from(remap[((word >> 24) & 0xF) as usize]) << 24)
                | (u64::from(remap[((word >> 28) & 0xF) as usize]) << 28)
                | (u64::from(remap[((word >> 32) & 0xF) as usize]) << 32)
                | (u64::from(remap[((word >> 36) & 0xF) as usize]) << 36)
                | (u64::from(remap[((word >> 40) & 0xF) as usize]) << 40)
                | (u64::from(remap[((word >> 44) & 0xF) as usize]) << 44)
                | (u64::from(remap[((word >> 48) & 0xF) as usize]) << 48)
                | (u64::from(remap[((word >> 52) & 0xF) as usize]) << 52)
                | (u64::from(remap[((word >> 56) & 0xF) as usize]) << 56)
                | (u64::from(remap[((word >> 60) & 0xF) as usize]) << 60);
            data.push(out);
        }
        return Packed {
            palette: new_palette,
            bits: new_bits,
            data: Some(data),
        };
    }

    let src_bits = usize::from(bits);
    let src_per = 64 / src_bits;
    let src_mask = mask(bits);
    let dst_bits = usize::from(new_bits);
    let dst_per = 64 / dst_bits;
    let mut data = Vec::with_capacity(entries.div_ceil(dst_per));
    let mut out_word = 0u64;
    let mut out_slot = 0usize;
    let mut out_shift = 0usize;
    let mut remaining = entries;
    for &word in words {
        let count = src_per.min(remaining);
        remaining -= count;
        let mut w = word;
        for _ in 0..count {
            let new_id = u64::from(remap[(w & src_mask) as usize]);
            w >>= src_bits;
            out_word |= new_id << out_shift;
            out_slot += 1;
            out_shift += dst_bits;
            if out_slot == dst_per {
                data.push(out_word);
                out_word = 0;
                out_slot = 0;
                out_shift = 0;
            }
        }
    }
    if out_slot > 0 {
        data.push(out_word);
    }
    Packed {
        palette: new_palette,
        bits: new_bits,
        data: Some(data),
    }
}

/// Table-driven word-wise list-to-list resize preserving vanilla
/// first-occurrence palette order and appending `last` at the end.
fn resize_list_to_list(
    old_bits: u8,
    old_words: &[u64],
    old_list: &[u32],
    new_bits: u8,
    entries: usize,
    last: u32,
) -> (Vec<u32>, Vec<u64>, usize) {
    let mut old_to_new = [u16::MAX; 256];
    let mut new_palette = Vec::with_capacity((old_list.len() + 1).min(1usize << new_bits));
    let src_bits = usize::from(old_bits);
    let src_per = 64 / src_bits;
    let src_mask = mask(old_bits);
    let dst_bits = usize::from(new_bits);
    let dst_per = 64 / dst_bits;
    let mut new_words = Vec::with_capacity(entries.div_ceil(dst_per));
    let mut out_word = 0u64;
    let mut out_slot = 0usize;
    let mut out_shift = 0usize;
    let mut remaining = entries;

    for &word in old_words {
        let count = src_per.min(remaining);
        remaining -= count;
        let mut w = word;
        for _ in 0..count {
            let old_id = (w & src_mask) as usize;
            w >>= src_bits;
            let mapped = old_to_new[old_id];
            let new_id = if mapped != u16::MAX {
                u64::from(mapped)
            } else {
                let val = old_list[old_id];
                let id = match new_palette.iter().position(|&v| v == val) {
                    Some(existing) => existing,
                    None => {
                        let id = new_palette.len();
                        new_palette.push(val);
                        id
                    }
                };
                old_to_new[old_id] = id as u16;
                id as u64
            };
            out_word |= new_id << out_shift;
            out_slot += 1;
            out_shift += dst_bits;
            if out_slot == dst_per {
                new_words.push(out_word);
                out_word = 0;
                out_slot = 0;
                out_shift = 0;
            }
        }
    }
    if out_slot > 0 {
        new_words.push(out_word);
    }

    let last_id = match new_palette.iter().position(|&v| v == last) {
        Some(existing) => existing,
        None => {
            let id = new_palette.len();
            new_palette.push(last);
            id
        }
    };
    (new_palette, new_words, last_id)
}

/// Word-wise list-to-global resize streaming resolved values directly into
/// `global_bits`-wide words.
fn resize_list_to_global(
    old_bits: u8,
    old_words: &[u64],
    old_list: &[u32],
    global_bits: u8,
    entries: usize,
) -> Vec<u64> {
    let src_bits = usize::from(old_bits);
    let src_per = 64 / src_bits;
    let src_mask = mask(old_bits);
    let dst_bits = usize::from(global_bits);
    let dst_per = 64 / dst_bits;
    let mut new_words = Vec::with_capacity(entries.div_ceil(dst_per));
    let mut out_word = 0u64;
    let mut out_slot = 0usize;
    let mut out_shift = 0usize;
    let mut remaining = entries;

    for &word in old_words {
        let count = src_per.min(remaining);
        remaining -= count;
        let mut w = word;
        for _ in 0..count {
            let val = u64::from(old_list[(w & src_mask) as usize]);
            w >>= src_bits;
            out_word |= val << out_shift;
            out_slot += 1;
            out_shift += dst_bits;
            if out_slot == dst_per {
                new_words.push(out_word);
                out_word = 0;
                out_slot = 0;
                out_shift = 0;
            }
        }
    }
    if out_slot > 0 {
        new_words.push(out_word);
    }
    new_words
}

/// Unpacks global-palette words into `out` in index order.
fn unpack_global(bits: u8, words: &[u64], out: &mut [u32]) {
    let bits_u = usize::from(bits);
    let per = 64 / bits_u;
    let m = mask(bits);
    for (chunk, &word) in out.chunks_mut(per).zip(words) {
        let mut w = word;
        for cell in chunk {
            *cell = (w & m) as u32;
            w >>= bits_u;
        }
    }
}

/// The saved form: distinct values in index order, and the packed indices
/// when there is more than one value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packed {
    pub palette: Vec<u32>,
    /// Bits per index in `data` (0 when `data` is absent).
    pub bits: u8,
    pub data: Option<Vec<u64>>,
}

/// Unique container identifier: every fresh container and every clone (which is
/// copy-on-write's way of editing a shared one) gets a new `uid`, paired with
/// the per-container `edits` counter stored inside [`Inner`].
#[derive(Debug)]
struct Uid(u64);
impl Default for Uid {
    fn default() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// Byte budget for cached canonical section palettes held by resident
/// [`PalettedContainer`]s, shared by every save folder that holds the `Arc`.
#[derive(Debug)]
pub struct SavedPaletteBudget {
    limit: usize,
    used: AtomicUsize,
}

impl SavedPaletteBudget {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            used: AtomicUsize::new(0),
        }
    }

    /// Bytes currently reserved by live cached section palettes.
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// Maximum bytes this budget admits.
    pub fn limit(&self) -> usize {
        self.limit
    }

    fn reserve(&self, charge: usize) -> bool {
        let mut used = self.used.load(Ordering::Relaxed);
        loop {
            let Some(next) = used.checked_add(charge).filter(|&next| next <= self.limit) else {
                return false;
            };
            match self
                .used
                .compare_exchange_weak(used, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return true,
                Err(actual) => used = actual,
            }
        }
    }
}

#[derive(Debug)]
struct SavedNbt {
    budget: Arc<SavedPaletteBudget>,
    charge: usize,
    ids: Box<[u32]>,
    words: Box<[i64]>,
}

impl Drop for SavedNbt {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.charge, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug)]
struct ListMeta {
    palette: ListPalette,
    saved_nbt: OnceLock<Arc<SavedNbt>>,
}

impl ListMeta {
    fn new(palette: ListPalette) -> Self {
        Self {
            palette,
            saved_nbt: OnceLock::new(),
        }
    }
}

/// Compact 32-byte container storage: small metadata (`edits`, `strategy`,
/// `bits`) packs immediately after the 1-byte enum tag, `words: Box<[u64]>`
/// stays 1-deref hot for `get`/`unpack`/`get_run`, `Single` and 2-biome
/// `InlineBiomes1` allocate zero heap memory and zero `OnceLock` storage.
#[derive(Clone, Debug)]
enum Inner {
    Single {
        edits: u32,
        strategy: Strategy,
        value: u32,
    },
    /// 64-entry 1-bit biome section (exactly 2 palette entries, 1 `u64` word):
    /// 100% inline with zero heap allocations.
    InlineBiomes1 {
        edits: u32,
        strategy: Strategy,
        palette: [u32; 2],
        word: u64,
    },
    List {
        edits: u32,
        strategy: Strategy,
        bits: u8,
        words: Box<[u64]>,
        meta: Box<ListMeta>,
    },
    Global {
        edits: u32,
        strategy: Strategy,
        bits: u8,
        words: Box<[u64]>,
        saved_nbt: Box<OnceLock<Arc<SavedNbt>>>,
    },
}

impl Inner {
    #[inline]
    fn edits(&self) -> u32 {
        match self {
            Self::Single { edits, .. }
            | Self::InlineBiomes1 { edits, .. }
            | Self::List { edits, .. }
            | Self::Global { edits, .. } => *edits,
        }
    }

    #[inline]
    fn reset_edits(&mut self) {
        match self {
            Self::Single { edits, .. }
            | Self::InlineBiomes1 { edits, .. }
            | Self::List { edits, .. }
            | Self::Global { edits, .. } => *edits = 0,
        }
    }

    #[inline]
    fn strategy(&self) -> Strategy {
        match self {
            Self::Single { strategy, .. }
            | Self::InlineBiomes1 { strategy, .. }
            | Self::List { strategy, .. }
            | Self::Global { strategy, .. } => *strategy,
        }
    }
}

#[derive(Debug)]
pub struct PalettedContainer {
    inner: Inner,
    /// Unique container id (see [`Uid`]); paired with `inner.edits()` for [`Self::stamp`].
    uid: Uid,
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::size_of::<Inner>() == 32);
    assert!(std::mem::size_of::<PalettedContainer>() == 40);
    assert!(std::mem::size_of::<ListMeta>() <= 64);
};

impl Clone for PalettedContainer {
    fn clone(&self) -> Self {
        let mut inner = self.inner.clone();
        inner.reset_edits();
        Self {
            inner,
            uid: Uid::default(),
        }
    }
}

impl PalettedContainer {
    /// `(uid, edits)`: equal to an earlier answer only if this is the same container and none of
    /// its entries was written since (`edits` wraps after 2^32 writes; a reader that keeps a
    /// stamp for that long is not one this serves).
    #[inline]
    pub fn stamp(&self) -> (u64, u32) {
        (self.uid.0, self.inner.edits())
    }
    /// `new PalettedContainer(initialValue, strategy)`.
    pub fn new(strategy: Strategy, value: u32) -> Self {
        Self {
            inner: Inner::Single {
                edits: 0,
                strategy,
                value,
            },
            uid: Uid::default(),
        }
    }
    #[inline]
    pub fn strategy(&self) -> Strategy {
        self.inner.strategy()
    }
    /// Bits per stored index (`BitStorage.getBits`).
    #[inline]
    pub fn bits(&self) -> u8 {
        match &self.inner {
            Inner::Single { .. } => 0,
            Inner::InlineBiomes1 { .. } => 1,
            Inner::List { bits, .. } | Inner::Global { bits, .. } => *bits,
        }
    }
    /// What the stored ids stand for — the values of the list palette, `None`
    /// for a single value (`uniform()`) and for the global palette, whose
    /// stored ids are the values.
    #[inline]
    pub fn palette(&self) -> Option<&[u32]> {
        match &self.inner {
            Inner::InlineBiomes1 { palette, .. } => Some(palette.as_slice()),
            Inner::List { meta, .. } => Some(meta.palette.as_slice()),
            Inner::Single { .. } | Inner::Global { .. } => None,
        }
    }
    /// Entries in the palette (the global palette reports its bit capacity).
    #[inline]
    pub fn palette_len(&self) -> usize {
        match &self.inner {
            Inner::Single { .. } => 1,
            Inner::InlineBiomes1 { .. } => 2,
            Inner::List { meta, .. } => meta.palette.len(),
            Inner::Global { bits, .. } => 1 << (*bits).min(31),
        }
    }
    #[inline]
    pub fn is_global(&self) -> bool {
        matches!(self.inner, Inner::Global { .. })
    }
    #[inline]
    pub fn get(&self, i: usize) -> u32 {
        match &self.inner {
            Inner::Single { value, .. } => *value,
            Inner::InlineBiomes1 { palette, word, .. } => palette[((*word >> i) & 1) as usize],
            Inner::List {
                bits, words, meta, ..
            } => meta.palette.as_slice()[read(words, *bits, i) as usize],
            Inner::Global { bits, words, .. } => read(words, *bits, i),
        }
    }
    /// `get` of the entries `start..start + out.len()`, the palette matched once.
    #[inline]
    pub fn get_run(&self, start: usize, out: &mut [u32]) {
        if out.is_empty() {
            return;
        }
        match &self.inner {
            Inner::Single { value, .. } => out.fill(*value),
            Inner::InlineBiomes1 { palette, word, .. } => {
                let [p0, p1] = *palette;
                let mut w = *word >> start;
                for cell in out.iter_mut() {
                    *cell = if (w & 1) == 0 { p0 } else { p1 };
                    w >>= 1;
                }
            }
            Inner::List {
                bits: 4,
                words,
                meta,
                ..
            } if out.len() == 16 && (start & 15) == 0 => {
                let list = meta.palette.as_slice();
                let word = words[start >> 4];
                for (k, cell) in out.iter_mut().enumerate() {
                    *cell = list[((word >> (k * 4)) & 15) as usize];
                }
            }
            Inner::List {
                bits, words, meta, ..
            } => {
                let list = meta.palette.as_slice();
                let bits_val = *bits;
                let bits_u = usize::from(bits_val);
                let per = 64 / bits_u;
                let m = mask(bits_val);
                let (mut word_idx, shift0) = locate(bits_val, start);
                let first_avail = per - (start - word_idx * per);
                let (first, mut rest) = out.split_at_mut(out.len().min(first_avail));
                let mut w = words[word_idx] >> shift0;
                for cell in first {
                    *cell = list[(w & m) as usize];
                    w >>= bits_u;
                }
                while !rest.is_empty() {
                    word_idx += 1;
                    let (chunk, next) = rest.split_at_mut(rest.len().min(per));
                    let mut w = words[word_idx];
                    for cell in chunk {
                        *cell = list[(w & m) as usize];
                        w >>= bits_u;
                    }
                    rest = next;
                }
            }
            Inner::Global { bits, words, .. } => {
                let bits_val = *bits;
                let bits_u = usize::from(bits_val);
                let per = 64 / bits_u;
                let m = mask(bits_val);
                let (mut word_idx, shift0) = locate(bits_val, start);
                let first_avail = per - (start - word_idx * per);
                let (first, mut rest) = out.split_at_mut(out.len().min(first_avail));
                let mut w = words[word_idx] >> shift0;
                for cell in first {
                    *cell = (w & m) as u32;
                    w >>= bits_u;
                }
                while !rest.is_empty() {
                    word_idx += 1;
                    let (chunk, next) = rest.split_at_mut(rest.len().min(per));
                    let mut w = words[word_idx];
                    for cell in chunk {
                        *cell = (w & m) as u32;
                        w >>= bits_u;
                    }
                    rest = next;
                }
            }
        }
    }
    /// The value when every entry is equal, without the palette having to be
    /// minimal.
    pub fn uniform(&self) -> Option<u32> {
        match &self.inner {
            Inner::Single { value, .. } => Some(*value),
            Inner::InlineBiomes1 { palette, word, .. } => {
                let [p0, p1] = *palette;
                if *word == 0 || p0 == p1 {
                    Some(p0)
                } else if *word == u64::MAX {
                    Some(p1)
                } else {
                    None
                }
            }
            Inner::List {
                strategy,
                bits,
                words,
                meta,
                ..
            } => {
                let list = meta.palette.as_slice();
                if list.len() == 1 {
                    return Some(list[0]);
                }
                let first_id = read(words, *bits, 0);
                let first = list[first_id as usize];
                all_words_equal_id(words, *bits, strategy.entries(), u64::from(first_id))
                    .then_some(first)
            }
            Inner::Global {
                strategy,
                bits,
                words,
                ..
            } => {
                let first = read(words, *bits, 0);
                all_words_equal_id(words, *bits, strategy.entries(), u64::from(first))
                    .then_some(first)
            }
        }
    }
    /// Whether `other` is stored the same way (bits, palette and words): then
    /// it holds the same values, found without unpacking either. (Containers
    /// stored differently can still hold the same values.)
    pub fn same_storage(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (Inner::Single { value: a, .. }, Inner::Single { value: b, .. }) => a == b,
            (
                Inner::InlineBiomes1 {
                    palette: pa,
                    word: wa,
                    ..
                },
                Inner::InlineBiomes1 {
                    palette: pb,
                    word: wb,
                    ..
                },
            ) => pa == pb && wa == wb,
            (
                Inner::List {
                    bits: ba,
                    words: wa,
                    meta: ma,
                    ..
                },
                Inner::List {
                    bits: bb,
                    words: wb,
                    meta: mb,
                    ..
                },
            ) => ba == bb && ma.palette.as_slice() == mb.palette.as_slice() && wa == wb,
            (
                Inner::Global {
                    bits: ba,
                    words: wa,
                    ..
                },
                Inner::Global {
                    bits: bb,
                    words: wb,
                    ..
                },
            ) => ba == bb && wa == wb,
            _ => false,
        }
    }
    /// Whether all entries equal `value`.
    pub fn is_uniform(&self, value: u32) -> bool {
        match &self.inner {
            Inner::Single { value: v, .. } => *v == value,
            Inner::InlineBiomes1 {
                palette: [p0, p1],
                word,
                ..
            } => (*word == u64::MAX || *p0 == value) && (*word == 0 || *p1 == value),
            Inner::List {
                strategy,
                bits,
                words,
                meta,
                ..
            } => {
                let list = meta.palette.as_slice();
                match list.iter().position(|&v| v == value) {
                    None => false,
                    Some(_) if list.len() == 1 => true,
                    Some(id) => all_words_equal_id(words, *bits, strategy.entries(), id as u64),
                }
            }
            Inner::Global {
                strategy,
                bits,
                words,
                ..
            } => all_words_equal_id(words, *bits, strategy.entries(), u64::from(value)),
        }
    }
    /// `getAndSet`: stores `value` and returns the previous one.
    pub fn set(&mut self, i: usize, value: u32) -> u32 {
        match &mut self.inner {
            Inner::Single {
                edits,
                value: current,
                ..
            } => {
                *edits = edits.wrapping_add(1);
                let old = *current;
                if old == value {
                    return old;
                }
                let id = self.resize(1, value);
                self.write_raw(i, id as u32);
                old
            }
            Inner::InlineBiomes1 {
                edits,
                palette,
                word,
                ..
            } => {
                *edits = edits.wrapping_add(1);
                let old_id = ((*word >> i) & 1) as usize;
                let old = palette[old_id];
                if value == palette[0] {
                    *word &= !(1u64 << i);
                    return old;
                }
                if value == palette[1] {
                    *word |= 1u64 << i;
                    return old;
                }
                let id = self.resize(2, value);
                self.write_raw(i, id as u32);
                old
            }
            Inner::List {
                edits,
                bits,
                words,
                meta,
                ..
            } => {
                *edits = edits.wrapping_add(1);
                let bits_val = *bits;
                let (word_idx, shift) = locate(bits_val, i);
                let m = mask(bits_val);
                let old_id = ((words[word_idx] >> shift) & m) as usize;
                let old = meta.palette.as_slice()[old_id];
                if old == value {
                    // Preserve `saved_nbt` on no-op writes while still matching
                    // `id_for` if a wire palette had duplicate entries.
                    if let Some(first_id) = meta.palette.as_slice().iter().position(|&v| v == value)
                        && first_id != old_id
                    {
                        words[word_idx] = (words[word_idx] & !(m << shift))
                            | (u64::from(first_id as u32) << shift);
                    }
                    return old;
                }
                meta.saved_nbt.take();
                if let Some(id) = meta.palette.as_slice().iter().position(|&v| v == value) {
                    words[word_idx] =
                        (words[word_idx] & !(m << shift)) | (u64::from(id as u32) << shift);
                    return old;
                }
                if meta.palette.len() < (1usize << bits_val) {
                    let id = meta.palette.push(value);
                    words[word_idx] =
                        (words[word_idx] & !(m << shift)) | (u64::from(id as u32) << shift);
                    return old;
                }
                let next_bits = bits_val + 1;
                let id = self.resize(next_bits, value);
                self.write_raw(i, id as u32);
                old
            }
            Inner::Global {
                edits,
                bits,
                words,
                saved_nbt,
                ..
            } => {
                *edits = edits.wrapping_add(1);
                let (word_idx, shift) = locate(*bits, i);
                let m = mask(*bits);
                let old = ((words[word_idx] >> shift) & m) as u32;
                if old != value {
                    saved_nbt.take();
                    words[word_idx] =
                        (words[word_idx] & !(m << shift)) | (u64::from(value) << shift);
                }
                old
            }
        }
    }
    fn write_raw(&mut self, i: usize, id: u32) {
        match &mut self.inner {
            Inner::Single { .. } => {}
            Inner::InlineBiomes1 { word, .. } => {
                let bit = 1u64 << i;
                if (id & 1) != 0 {
                    *word |= bit;
                } else {
                    *word &= !bit;
                }
            }
            Inner::List { bits, words, .. } | Inner::Global { bits, words, .. } => {
                write(words, *bits, i, id);
            }
        }
    }
    fn make_list_inner(
        edits: u32,
        strategy: Strategy,
        bits: u8,
        palette: Vec<u32>,
        words: Vec<u64>,
        shrink: bool,
    ) -> Inner {
        if strategy.kind == Kind::Biomes && bits == 1 && palette.len() == 2 {
            return Inner::InlineBiomes1 {
                edits,
                strategy,
                palette: [palette[0], palette[1]],
                word: words[0],
            };
        }
        Inner::List {
            edits,
            strategy,
            bits,
            words: words.into_boxed_slice(),
            meta: Box::new(ListMeta::new(ListPalette::from_vec(palette, shrink))),
        }
    }
    /// `PalettedContainer.onResize`: rebuild with the configuration for
    /// `bits`, then add `last` and return its id.
    fn resize(&mut self, bits: u8, last: u32) -> usize {
        let strategy = self.strategy();
        let edits = self.inner.edits();
        let config = strategy.config(bits);
        let storage_bits = strategy.storage_bits(config);
        let entries = strategy.entries();
        match (&self.inner, config) {
            (_, Config::Single) => {
                self.inner = Inner::Single {
                    edits,
                    strategy,
                    value: last,
                };
                0
            }
            (Inner::Single { value: only, .. }, Config::List(1))
                if strategy.kind == Kind::Biomes =>
            {
                let only = *only;
                if only == last {
                    self.inner = Inner::List {
                        edits,
                        strategy,
                        bits: 1,
                        words: Box::<[u64]>::from([0u64]),
                        meta: Box::new(ListMeta::new(ListPalette::from_single(only))),
                    };
                    0
                } else {
                    self.inner = Inner::InlineBiomes1 {
                        edits,
                        strategy,
                        palette: [only, last],
                        word: 0,
                    };
                    1
                }
            }
            (Inner::Single { value: only, .. }, Config::List(new_bits)) => {
                let only = *only;
                let (palette, id) = if only == last {
                    (ListPalette::from_single(only), 0)
                } else {
                    (ListPalette::from_pair(only, last), 1)
                };
                let words = vec![0u64; storage_len(new_bits, entries)].into_boxed_slice();
                self.inner = Inner::List {
                    edits,
                    strategy,
                    bits: new_bits,
                    words,
                    meta: Box::new(ListMeta::new(palette)),
                };
                id
            }
            (Inner::Single { value: only, .. }, Config::Global) => {
                let only = *only;
                let words = fill_words(storage_bits, entries, only).into_boxed_slice();
                self.inner = Inner::Global {
                    edits,
                    strategy,
                    bits: storage_bits,
                    words,
                    saved_nbt: Box::new(OnceLock::new()),
                };
                last as usize
            }
            (
                Inner::InlineBiomes1 {
                    palette: [p0, p1],
                    word,
                    ..
                },
                Config::List(2),
            ) => {
                let (p0, p1, w) = (*p0, *p1, *word);
                let (new_palette, words, id) = if w == 0 || p0 == p1 {
                    let words = Box::<[u64]>::from([0u64, 0u64]);
                    if p0 == last {
                        (ListPalette::from_single(p0), words, 0)
                    } else {
                        (ListPalette::from_pair(p0, last), words, 1)
                    }
                } else if w == u64::MAX {
                    let words = Box::<[u64]>::from([0u64, 0u64]);
                    if p1 == last {
                        (ListPalette::from_single(p1), words, 0)
                    } else {
                        (ListPalette::from_pair(p1, last), words, 1)
                    }
                } else {
                    let (first, second, id_bits) = if (w & 1) == 0 {
                        (p0, p1, w)
                    } else {
                        (p1, p0, !w)
                    };
                    let lo = spread_u32_to_2bit(id_bits as u32);
                    let hi = spread_u32_to_2bit((id_bits >> 32) as u32);
                    let words = Box::<[u64]>::from([lo, hi]);
                    if last == first {
                        (ListPalette::from_pair(first, second), words, 0)
                    } else if last == second {
                        (ListPalette::from_pair(first, second), words, 1)
                    } else {
                        (ListPalette::from_triple(first, second, last), words, 2)
                    }
                };
                self.inner = Inner::List {
                    edits,
                    strategy,
                    bits: 2,
                    words,
                    meta: Box::new(ListMeta::new(new_palette)),
                };
                id
            }
            (
                Inner::InlineBiomes1 {
                    palette: [p0, p1],
                    word,
                    ..
                },
                Config::List(new_bits),
            ) => {
                let old_list = [*p0, *p1];
                let old_words = [*word];
                let (new_palette, new_words, id) =
                    resize_list_to_list(1, &old_words, &old_list, new_bits, entries, last);
                self.inner =
                    Self::make_list_inner(edits, strategy, new_bits, new_palette, new_words, false);
                id
            }
            (
                Inner::InlineBiomes1 {
                    palette: [p0, p1],
                    word,
                    ..
                },
                Config::Global,
            ) => {
                let old_list = [*p0, *p1];
                let old_words = [*word];
                let new_words =
                    resize_list_to_global(1, &old_words, &old_list, storage_bits, entries);
                self.inner = Inner::Global {
                    edits,
                    strategy,
                    bits: storage_bits,
                    words: new_words.into_boxed_slice(),
                    saved_nbt: Box::new(OnceLock::new()),
                };
                last as usize
            }
            (
                Inner::List {
                    bits: old_bits,
                    words: old_words,
                    meta: old_meta,
                    ..
                },
                Config::List(new_bits),
            ) => {
                let (new_palette, new_words, id) = resize_list_to_list(
                    *old_bits,
                    old_words,
                    old_meta.palette.as_slice(),
                    new_bits,
                    entries,
                    last,
                );
                self.inner =
                    Self::make_list_inner(edits, strategy, new_bits, new_palette, new_words, false);
                id
            }
            (
                Inner::List {
                    bits: old_bits,
                    words: old_words,
                    meta: old_meta,
                    ..
                },
                Config::Global,
            ) => {
                let new_words = resize_list_to_global(
                    *old_bits,
                    old_words,
                    old_meta.palette.as_slice(),
                    storage_bits,
                    entries,
                );
                self.inner = Inner::Global {
                    edits,
                    strategy,
                    bits: storage_bits,
                    words: new_words.into_boxed_slice(),
                    saved_nbt: Box::new(OnceLock::new()),
                };
                last as usize
            }
            (
                Inner::Global {
                    bits: old_bits,
                    words: old_words,
                    ..
                },
                Config::List(new_bits),
            ) => {
                let mut vals = vec![0u32; entries];
                unpack_global(*old_bits, old_words, &mut vals);
                let (mut new_palette, ids) = palette_ids(&vals);
                let id = match new_palette.iter().position(|&v| v == last) {
                    Some(existing) => existing,
                    None => {
                        let id = new_palette.len();
                        new_palette.push(last);
                        id
                    }
                };
                let new_words = pack_words_slice(new_bits, &ids);
                self.inner =
                    Self::make_list_inner(edits, strategy, new_bits, new_palette, new_words, false);
                id
            }
            (
                Inner::Global {
                    bits: old_bits,
                    words: old_words,
                    ..
                },
                Config::Global,
            ) => {
                let mut vals = vec![0u32; entries];
                unpack_global(*old_bits, old_words, &mut vals);
                let new_words = pack_words_slice(storage_bits, &vals).into_boxed_slice();
                self.inner = Inner::Global {
                    edits,
                    strategy,
                    bits: storage_bits,
                    words: new_words,
                    saved_nbt: Box::new(OnceLock::new()),
                };
                last as usize
            }
        }
    }
    /// All values in index order.
    pub fn unpack(&self, out: &mut [u32]) {
        assert_eq!(out.len(), self.strategy().entries());
        match &self.inner {
            Inner::Single { value, .. } => out.fill(*value),
            Inner::InlineBiomes1 { palette, word, .. } => {
                let [p0, p1] = *palette;
                let mut w = *word;
                for cell in out.iter_mut() {
                    *cell = if (w & 1) == 0 { p0 } else { p1 };
                    w >>= 1;
                }
            }
            Inner::List {
                bits: 4,
                words,
                meta,
                ..
            } => {
                let list = meta.palette.as_slice();
                for (row, &word) in out.as_chunks_mut::<16>().0.iter_mut().zip(words.iter()) {
                    for (x, cell) in row.iter_mut().enumerate() {
                        *cell = list[((word >> (x * 4)) & 15) as usize];
                    }
                }
            }
            Inner::List {
                bits, words, meta, ..
            } => {
                let list = meta.palette.as_slice();
                let bits_val = *bits;
                let bits_u = usize::from(bits_val);
                let per = 64 / bits_u;
                let m = mask(bits_val);
                for (chunk, &word) in out.chunks_mut(per).zip(words.iter()) {
                    let mut w = word;
                    for cell in chunk {
                        *cell = list[(w & m) as usize];
                        w >>= bits_u;
                    }
                }
            }
            Inner::Global { bits, words, .. } => unpack_global(*bits, words, out),
        }
    }
    /// The stored ids through `map`: `out[i] = map[id of entry i]`, in index
    /// order, the id being a palette id for a list palette and the value
    /// itself for the global one (see `palette`). A single-value palette
    /// stores no ids: `out` is left as it is.
    pub fn unpack_ids_mapped<T: Copy>(&self, out: &mut [T], map: &[T]) {
        assert_eq!(out.len(), self.strategy().entries());
        match &self.inner {
            Inner::Single { .. } => {}
            Inner::InlineBiomes1 { word, .. } => {
                let mut w = *word;
                for cell in out.iter_mut() {
                    *cell = map[(w & 1) as usize];
                    w >>= 1;
                }
            }
            Inner::List { bits: 4, words, .. } | Inner::Global { bits: 4, words, .. } => {
                // Four bits a linear palette's usual width: a whole word of them is a
                // row of 16, and immediate shifts for the compiler to unroll.
                for (row, &word) in out.as_chunks_mut::<16>().0.iter_mut().zip(words.iter()) {
                    for (x, cell) in row.iter_mut().enumerate() {
                        *cell = map[((word >> (x * 4)) & 0xF) as usize];
                    }
                }
            }
            Inner::List { bits, words, .. } | Inner::Global { bits, words, .. } => {
                let bits_val = *bits;
                let bits_u = usize::from(bits_val);
                let per = 64 / bits_u;
                let m = mask(bits_val);
                for (chunk, &word) in out.chunks_mut(per).zip(words.iter()) {
                    let mut w = word;
                    for cell in chunk {
                        *cell = map[(w & m) as usize];
                        w >>= bits_u;
                    }
                }
            }
        }
    }
    /// The palette id of every entry in index order, and the list palette
    /// they index (`get(i)` is `list[ids[i]]`), without looking the values
    /// up; `None` (and `ids` untouched) for a single-value or global palette.
    pub fn unpack_ids(&self, ids: &mut [u16]) -> Option<&[u32]> {
        match &self.inner {
            Inner::Single { .. } | Inner::Global { .. } => None,
            Inner::InlineBiomes1 { palette, word, .. } => {
                assert_eq!(ids.len(), 64);
                let mut w = *word;
                for cell in ids.iter_mut() {
                    *cell = (w & 1) as u16;
                    w >>= 1;
                }
                Some(palette.as_slice())
            }
            Inner::List {
                strategy,
                bits,
                words,
                meta,
                ..
            } => {
                assert_eq!(ids.len(), strategy.entries());
                let bits_val = *bits;
                if bits_val == 4 {
                    for (row, &word) in ids.as_chunks_mut::<16>().0.iter_mut().zip(words.iter()) {
                        for (x, cell) in row.iter_mut().enumerate() {
                            *cell = ((word >> (x * 4)) & 0xF) as u16;
                        }
                    }
                } else {
                    let bits_u = usize::from(bits_val);
                    let per = 64 / bits_u;
                    let m = mask(bits_val);
                    for (chunk, &word) in ids.chunks_mut(per).zip(words.iter()) {
                        let mut w = word;
                        for cell in chunk {
                            // List palettes stay at a few bits (eight at most for blocks).
                            *cell = (w & m) as u16;
                            w >>= bits_u;
                        }
                    }
                }
                Some(meta.palette.as_slice())
            }
        }
    }
    /// A container holding `values` in the smallest configuration for their
    /// distinct count, with palette ids in first-occurrence order, like a
    /// container unpacked from disk.
    pub fn from_values(strategy: Strategy, values: &[u32]) -> Self {
        assert_eq!(values.len(), strategy.entries());
        let Some((&first, rest)) = values.split_first() else {
            unreachable!("strategy entries are always non-empty");
        };
        let Some(diff_offset) = rest.iter().position(|&v| v != first) else {
            return Self::new(strategy, first);
        };
        if strategy.kind == Kind::Biomes {
            let first_diff = diff_offset + 1;
            let second = values[first_diff];
            let mut word = 1u64 << first_diff;
            let mut only_two = true;
            for (i, &v) in values.iter().enumerate().skip(first_diff + 1) {
                if v == second {
                    word |= 1u64 << i;
                } else if v != first {
                    only_two = false;
                    break;
                }
            }
            if only_two {
                return Self {
                    inner: Inner::InlineBiomes1 {
                        edits: 0,
                        strategy,
                        palette: [first, second],
                        word,
                    },
                    uid: Uid::default(),
                };
            }
        }
        let (palette, ids) = palette_ids(values);
        Self::build(strategy, palette, &ids)
    }
    /// A container over a distinct-value `palette` and an id per entry.
    fn build(strategy: Strategy, palette: Vec<u32>, ids: &[u32]) -> Self {
        let config = strategy.config(ceil_log2(palette.len()));
        match config {
            Config::Single => Self::new(strategy, palette[0]),
            Config::List(1) if strategy.kind == Kind::Biomes && palette.len() == 2 => Self {
                inner: Inner::InlineBiomes1 {
                    edits: 0,
                    strategy,
                    palette: [palette[0], palette[1]],
                    word: pack_word_1bit(ids),
                },
                uid: Uid::default(),
            },
            Config::List(bits) => Self {
                inner: Inner::List {
                    edits: 0,
                    strategy,
                    bits,
                    words: pack_words_slice(bits, ids).into_boxed_slice(),
                    meta: Box::new(ListMeta::new(ListPalette::from_vec(palette, true))),
                },
                uid: Uid::default(),
            },
            Config::Global => {
                let bits = strategy.global_bits;
                Self {
                    inner: Inner::Global {
                        edits: 0,
                        strategy,
                        bits,
                        words: pack_words(
                            bits,
                            ids.len(),
                            ids.iter().map(|&id| palette[id as usize]),
                        )
                        .into_boxed_slice(),
                        saved_nbt: Box::new(OnceLock::new()),
                    },
                    uid: Uid::default(),
                }
            }
        }
    }
    /// `PalettedContainer.read`: the container as sent on the wire. `bits` is
    /// the leading byte, `palette` the VarInt entries (a single one for
    /// `bits == 0`, none for the global palette) and `words` the longs.
    pub fn from_wire(
        strategy: Strategy,
        bits: u8,
        palette: Vec<u32>,
        words: Vec<u64>,
    ) -> Result<Self, PaletteError> {
        let config = strategy.config(bits);
        let storage_bits = strategy.storage_bits(config);
        let expected = storage_len(storage_bits, strategy.entries());
        if words.len() != expected {
            return Err(PaletteError::BadStorage {
                expected,
                actual: words.len(),
            });
        }
        match config {
            Config::Single => {
                if palette.len() != 1 {
                    return Err(PaletteError::BadPalette { len: palette.len() });
                }
                Ok(Self::new(strategy, palette[0]))
            }
            Config::List(bits) => {
                if palette.is_empty() || palette.len() > 1 << bits {
                    return Err(PaletteError::BadPalette { len: palette.len() });
                }
                if strategy.kind == Kind::Biomes && bits == 1 && palette.len() == 2 {
                    return Ok(Self {
                        inner: Inner::InlineBiomes1 {
                            edits: 0,
                            strategy,
                            palette: [palette[0], palette[1]],
                            word: words[0],
                        },
                        uid: Uid::default(),
                    });
                }
                let container = Self {
                    inner: Inner::List {
                        edits: 0,
                        strategy,
                        bits,
                        words: words.into_boxed_slice(),
                        meta: Box::new(ListMeta::new(ListPalette::from_vec(palette, true))),
                    },
                    uid: Uid::default(),
                };
                container.check_indices()?;
                Ok(container)
            }
            Config::Global => Ok(Self {
                inner: Inner::Global {
                    edits: 0,
                    strategy,
                    bits: storage_bits,
                    words: words.into_boxed_slice(),
                    saved_nbt: Box::new(OnceLock::new()),
                },
                uid: Uid::default(),
            }),
        }
    }
    /// Rejects a stored index at or past the palette: the same inputs are
    /// accepted as an entry-by-entry scan accepts, and the same first offending
    /// entry is named. One whole word is checked at once: adding `2^bits - len`
    /// to a word lifts every field by that much, so a field carries out of its
    /// own bits exactly when its index is past the palette. A carry out of a
    /// field that a previous field's carry reached adds one more, so such a
    /// field can only look bad where a genuinely bad field precedes it in the
    /// same word; the word is then rescanned entry by entry for the error.
    fn check_indices(&self) -> Result<(), PaletteError> {
        let Inner::List {
            strategy,
            bits: stored_bits,
            words,
            meta,
            ..
        } = &self.inner
        else {
            return Ok(());
        };
        let len = meta.palette.len();
        debug_assert!(len >= 1);
        debug_assert_eq!(words.len(), storage_len(*stored_bits, strategy.entries()));
        let bits = usize::from(*stored_bits);
        if len == 1 << bits {
            // Every field is a palette entry.
            return Ok(());
        }
        let per = 64 / bits;
        let entries = strategy.entries();
        // One in the low bit of each field of a word, the field's top bit, and
        // the bias that overflows a field whose index is `len` or more.
        let ones = (0..per).fold(0u64, |a, field| a | 1 << (field * bits));
        let bias = ones * ((1u64 << bits) - len as u64);
        let top = ones << (bits - 1);
        let full = entries / per;
        let (whole, tail) = words.split_at(full);
        for &word in whole {
            let sum = word.wrapping_add(bias);
            if carry_out(word, bias, sum) & top != 0 {
                return Err(first_bad(word, per, bits, len));
            }
        }
        // The last word's fields past `entries` are padding: whatever they
        // hold, the scan never reads them.
        let rest = entries - full * per;
        if rest > 0
            && let Some(&word) = tail.first()
        {
            let used = (1u64 << (rest * bits)) - 1;
            let sum = word.wrapping_add(bias & used);
            if carry_out(word, bias & used, sum) & top & used != 0 {
                return Err(first_bad(word, rest, bits, len));
            }
        }
        Ok(())
    }
    /// `PalettedContainer.write`: the leading bits byte, the palette entries
    /// (none for the global palette) and the raw longs, exactly as resident.
    pub fn to_wire(&self) -> (u8, Vec<u32>, &[u64]) {
        match &self.inner {
            Inner::Single { value, .. } => (0, vec![*value], &[]),
            Inner::InlineBiomes1 { palette, word, .. } => {
                (1, palette.to_vec(), std::slice::from_ref(word))
            }
            Inner::List {
                bits, words, meta, ..
            } => (*bits, meta.palette.as_slice().to_vec(), words),
            Inner::Global { bits, words, .. } => (*bits, Vec::new(), words),
        }
    }
    /// `PalettedContainer.pack`: distinct values in index order, indices
    /// re-encoded for the saved bit width.
    pub fn pack(&self) -> Packed {
        match &self.inner {
            Inner::Single { value, .. } => Packed {
                palette: vec![*value],
                bits: 0,
                data: None,
            },
            Inner::InlineBiomes1 { palette, word, .. } => {
                let [p0, p1] = *palette;
                let w = *word;
                if w == 0 || p0 == p1 {
                    Packed {
                        palette: vec![p0],
                        bits: 0,
                        data: None,
                    }
                } else if w == u64::MAX {
                    Packed {
                        palette: vec![p1],
                        bits: 0,
                        data: None,
                    }
                } else if (w & 1) == 0 {
                    Packed {
                        palette: vec![p0, p1],
                        bits: 1,
                        data: Some(vec![w]),
                    }
                } else {
                    Packed {
                        palette: vec![p1, p0],
                        bits: 1,
                        data: Some(vec![!w]),
                    }
                }
            }
            Inner::List {
                strategy,
                bits,
                words,
                meta,
                ..
            } => pack_list(*strategy, *bits, words, meta.palette.as_slice()),
            Inner::Global { strategy, .. } => {
                if let Some(value) = self.uniform() {
                    return Packed {
                        palette: vec![value],
                        bits: 0,
                        data: None,
                    };
                }
                let entries = strategy.entries();
                let mut values = vec![0u32; entries];
                self.unpack(&mut values);
                let (palette, ids) = palette_ids(&values);
                let bits = match strategy.config(ceil_log2(palette.len())) {
                    Config::Single => 0,
                    Config::List(bits) => bits,
                    Config::Global => ceil_log2(palette.len()),
                };
                let data = (bits != 0).then(|| pack_words_slice(bits, &ids));
                Packed {
                    palette,
                    bits,
                    data,
                }
            }
        }
    }

    fn saved_nbt_lock(&self) -> Option<&OnceLock<Arc<SavedNbt>>> {
        match &self.inner {
            Inner::Single { .. } | Inner::InlineBiomes1 { .. } => None,
            Inner::List { meta, .. } => Some(&meta.saved_nbt),
            Inner::Global { saved_nbt, .. } => Some(saved_nbt),
        }
    }

    fn pack_nbt_fields(
        palette: &[u32],
        data: Option<Vec<i64>>,
        name: impl FnMut(u32) -> Tag,
    ) -> Tag {
        let mut fields = vec![(
            "palette",
            Tag::List(palette.iter().copied().map(name).collect()),
        )];
        if let Some(words) = data {
            fields.push(("data", Tag::LongArray(words)));
        }
        Tag::compound(fields)
    }

    /// Canonical `palette` and `data` NBT compound for a saved section.
    ///
    /// When `budget` is `Some`, a non-uniform container caches its canonical
    /// palette ids and packed `data` words lazily on the first save within the
    /// budget's byte cap. Repeated saves of an unchanged container (or clones
    /// sharing the entry) reuse those words and re-resolve `name` against the
    /// caller's current registry. Mutating a cell via [`Self::set`] drops the
    /// entry and releases its budget reservation when the last owner drops.
    pub fn pack_nbt(
        &self,
        budget: Option<&Arc<SavedPaletteBudget>>,
        name: impl FnMut(u32) -> Tag,
    ) -> Tag {
        let cache = self.saved_nbt_lock();
        if let Some(saved) = cache.and_then(OnceLock::get)
            && budget.is_some_and(|budget| Arc::ptr_eq(budget, &saved.budget))
        {
            return Self::pack_nbt_fields(&saved.ids, Some(saved.words.to_vec()), name);
        }
        let Packed { palette, data, .. } = self.pack();
        let data = data.map(|words| {
            words
                .into_iter()
                .map(|word| word as i64)
                .collect::<Vec<_>>()
        });
        if let (Some(cache), Some(budget), Some(words)) = (cache, budget, data.as_deref())
            && cache.get().is_none()
        {
            let charge = std::mem::size_of::<SavedNbt>()
                + 2 * std::mem::size_of::<usize>()
                + std::mem::size_of_val(palette.as_slice())
                + std::mem::size_of_val(words);
            if budget.reserve(charge) {
                let _ = cache.set(Arc::new(SavedNbt {
                    budget: Arc::clone(budget),
                    charge,
                    ids: palette.as_slice().into(),
                    words: words.into(),
                }));
            }
        }
        Self::pack_nbt_fields(&palette, data, name)
    }

    /// `PalettedContainer.unpack`: a container from the saved form.
    pub fn from_packed(
        strategy: Strategy,
        palette: Vec<u32>,
        data: Option<&[u64]>,
    ) -> Result<Self, PaletteError> {
        if palette.is_empty() {
            return Err(PaletteError::BadPalette { len: 0 });
        }
        let config = strategy.config(ceil_log2(palette.len()));
        let entries = strategy.entries();
        let disc_bits = match config {
            Config::Single => return Ok(Self::new(strategy, palette[0])),
            Config::List(bits) => bits,
            Config::Global => ceil_log2(palette.len()),
        };
        let data = data.ok_or(PaletteError::BadStorage {
            expected: storage_len(disc_bits, entries),
            actual: 0,
        })?;
        let expected = storage_len(disc_bits, entries);
        if data.len() != expected {
            return Err(PaletteError::BadStorage {
                expected,
                actual: data.len(),
            });
        }
        if let Config::List(bits) = config {
            if strategy.kind == Kind::Biomes && bits == 1 && palette.len() == 2 {
                return Ok(Self {
                    inner: Inner::InlineBiomes1 {
                        edits: 0,
                        strategy,
                        palette: [palette[0], palette[1]],
                        word: data[0],
                    },
                    uid: Uid::default(),
                });
            }
            // List palettes already use the saved indices as resident storage.
            // Validate them without allocating an unpacked copy that is unused.
            let container = Self {
                inner: Inner::List {
                    edits: 0,
                    strategy,
                    bits,
                    words: data.into(),
                    meta: Box::new(ListMeta::new(ListPalette::from_vec(palette, true))),
                },
                uid: Uid::default(),
            };
            container.check_indices()?;
            return Ok(container);
        }
        // Global palette: validate every saved index in order and stream
        // resolved values directly into global-width words (zero temp `ids`).
        let src_bits = usize::from(disc_bits);
        let src_per = 64 / src_bits;
        let src_mask = mask(disc_bits);
        let dst_bits = usize::from(strategy.global_bits);
        let dst_per = 64 / dst_bits;
        let mut out_words = Vec::with_capacity(entries.div_ceil(dst_per));
        let mut out_word = 0u64;
        let mut out_slot = 0usize;
        let mut out_shift = 0usize;
        let mut remaining = entries;
        for &word in data {
            let count = src_per.min(remaining);
            remaining -= count;
            let mut w = word;
            for _ in 0..count {
                let index = (w & src_mask) as u32;
                w >>= src_bits;
                let Some(&value) = palette.get(index as usize) else {
                    return Err(PaletteError::BadIndex {
                        index,
                        palette: palette.len(),
                    });
                };
                out_word |= u64::from(value) << out_shift;
                out_slot += 1;
                out_shift += dst_bits;
                if out_slot == dst_per {
                    out_words.push(out_word);
                    out_word = 0;
                    out_slot = 0;
                    out_shift = 0;
                }
            }
        }
        if out_slot > 0 {
            out_words.push(out_word);
        }
        Ok(Self {
            inner: Inner::Global {
                edits: 0,
                strategy,
                bits: strategy.global_bits,
                words: out_words.into_boxed_slice(),
                saved_nbt: Box::new(OnceLock::new()),
            },
            uid: Uid::default(),
        })
    }
    /// Whether any palette entry satisfies `predicate` (`maybeHas`): entries
    /// no longer used by any position still count.
    pub fn maybe_has(&self, mut predicate: impl FnMut(u32) -> bool) -> bool {
        match &self.inner {
            Inner::Single { value, .. } => predicate(*value),
            Inner::InlineBiomes1 { palette, .. } => predicate(palette[0]) || predicate(palette[1]),
            Inner::List { meta, .. } => meta.palette.as_slice().iter().any(|&v| predicate(v)),
            Inner::Global { bits, .. } => (0..1u32 << (*bits).min(31)).any(predicate),
        }
    }
    /// Whether any entry satisfies `predicate`. Unlike [`Self::maybe_has`],
    /// palette entries no entry uses any more don't count.
    pub fn has(&self, mut predicate: impl FnMut(u32) -> bool) -> bool {
        let (strategy, bits_val, words, list) = match &self.inner {
            Inner::Single { value, .. } => return predicate(*value),
            Inner::InlineBiomes1 { palette, word, .. } => {
                let m0 = predicate(palette[0]);
                let m1 = predicate(palette[1]);
                return (m0 && *word != u64::MAX) || (m1 && *word != 0);
            }
            Inner::List {
                strategy,
                bits,
                words,
                meta,
                ..
            } => (*strategy, *bits, words.as_ref(), meta.palette.as_slice()),
            Inner::Global {
                strategy,
                bits,
                words,
                ..
            } => {
                let bits_val = *bits;
                let bits_u = usize::from(bits_val);
                let per = 64 / bits_u;
                let m = mask(bits_val);
                let mut remaining = strategy.entries();
                for &word in words.iter() {
                    let chunk = per.min(remaining);
                    remaining -= chunk;
                    let mut w = word;
                    for _ in 0..chunk {
                        if predicate((w & m) as u32) {
                            return true;
                        }
                        w >>= bits_u;
                    }
                }
                return false;
            }
        };
        let mut matching = [false; 256];
        let mut count = 0usize;
        for (slot, &v) in matching[..list.len()].iter_mut().zip(list) {
            if predicate(v) {
                *slot = true;
                count += 1;
            }
        }
        if count == 0 || count == list.len() {
            return count != 0;
        }
        let (bits_u, entries) = (usize::from(bits_val), strategy.entries());
        let per = 64 / bits_u;
        // The words whose fields all hold entries; the last may be partial.
        let full = entries / per;
        if count <= 4 {
            // A field-wise zero test of each word XORed with the matching
            // index in every field: `(v - lo) & !v & hi` is non-zero exactly
            // when some field of `v` is zero.
            let lo = (0..per).fold(0u64, |a, f| a | 1 << (f * bits_u));
            let hi = lo << (bits_u - 1);
            let full_words = &words[..full];
            let found = matching[..list.len()]
                .iter()
                .enumerate()
                .filter(|(_, m)| **m)
                .any(|(i, _)| {
                    let pattern = lo * i as u64;
                    full_words.iter().any(|&w| {
                        let v = w ^ pattern;
                        v.wrapping_sub(lo) & !v & hi != 0
                    })
                });
            if found {
                return true;
            }
            let rem = entries - full * per;
            if rem > 0 {
                let m = mask(bits_val);
                let mut w = words[full];
                for _ in 0..rem {
                    if matching[(w & m) as usize] {
                        return true;
                    }
                    w >>= bits_u;
                }
            }
            return false;
        }
        let m = mask(bits_val);
        let mut remaining = entries;
        for &word in words {
            let chunk = per.min(remaining);
            remaining -= chunk;
            let mut w = word;
            for _ in 0..chunk {
                if matching[(w & m) as usize] {
                    return true;
                }
                w >>= bits_u;
            }
        }
        false
    }
    /// Heap bytes held by the palette, the storage, and this container's share
    /// of any cached canonical saved palette: clones share one entry, and each
    /// counts its charge divided by the number of containers holding it, so
    /// summing them counts the entry once.
    pub fn heap_bytes(&self) -> usize {
        self.storage_heap_bytes()
            + self
                .saved_nbt_lock()
                .and_then(OnceLock::get)
                .map_or(0, |saved| saved.charge / Arc::strong_count(saved))
    }
    /// Heap bytes held by the palette and the storage.
    fn storage_heap_bytes(&self) -> usize {
        match &self.inner {
            Inner::Single { .. } | Inner::InlineBiomes1 { .. } => 0,
            Inner::List { words, meta, .. } => {
                words.len() * std::mem::size_of::<u64>()
                    + std::mem::size_of::<ListMeta>()
                    + meta.palette.heap_bytes()
            }
            Inner::Global { words, .. } => {
                words.len() * std::mem::size_of::<u64>()
                    + std::mem::size_of::<OnceLock<Arc<SavedNbt>>>()
            }
        }
    }
}
impl PartialEq for PalettedContainer {
    /// Equal when every entry is. The same width, palette and words are the
    /// same entries, whatever they are: compared without reading each.
    fn eq(&self, other: &Self) -> bool {
        let entries = self.strategy().entries();
        if entries != other.strategy().entries() {
            return false;
        }
        if self.same_storage(other) {
            return true;
        }
        (0..entries).all(|i| self.get(i) == other.get(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const BLOCKS: Strategy = Strategy::blocks(16);

    fn cached_saved_nbt(c: &PalettedContainer) -> Option<&Arc<SavedNbt>> {
        c.saved_nbt_lock().and_then(OnceLock::get)
    }

    #[test]
    fn pack_nbt_caches_lazily_and_follows_unchanged_clones() {
        let mut palette = PalettedContainer::new(BLOCKS, 0);
        palette.set(1, 1);
        let budget = Arc::new(SavedPaletteBudget::new(1 << 20));
        let name = |id| Tag::Int(id as i32);

        // Lazy: nothing cached until the first save.
        assert_eq!(budget.used(), 0);
        assert!(cached_saved_nbt(&palette).is_none());
        let base_heap = palette.heap_bytes();

        let first = palette.pack_nbt(Some(&budget), name);
        let cached_charge = budget.used();
        assert!(cached_charge > 0);
        assert_eq!(palette.heap_bytes(), base_heap + cached_charge);
        let old = Arc::downgrade(cached_saved_nbt(&palette).unwrap());

        // An unchanged clone shares the entry without charging the budget again.
        let unchanged = palette.clone();
        assert_eq!(unchanged.pack_nbt(Some(&budget), name), first);
        assert!(old.ptr_eq(&Arc::downgrade(cached_saved_nbt(&unchanged).unwrap())));
        assert_eq!(budget.used(), cached_charge);

        // A no-op write keeps the cache; a real mutation invalidates it.
        palette.set(1, 1);
        assert!(cached_saved_nbt(&palette).is_some());
        palette.set(1, 2);
        assert!(cached_saved_nbt(&palette).is_none());
        assert_eq!(palette.heap_bytes(), base_heap);

        let changed = palette.pack_nbt(Some(&budget), name);
        assert_ne!(first, changed);
        assert_eq!(changed, palette.pack_nbt(None, name));
        // Both old (held by `unchanged`) and new entries are charged until `unchanged` drops.
        assert!(budget.used() > cached_charge);
        drop(unchanged);
        assert!(old.upgrade().is_none());
        let after_drop_unchanged = budget.used();
        assert!(after_drop_unchanged > 0);

        // Cache hit re-resolves registry names at save time.
        assert_eq!(
            palette.pack_nbt(Some(&budget), |id| Tag::Long(i64::from(id) + 100)),
            palette.pack_nbt(None, |id| Tag::Long(i64::from(id) + 100))
        );

        drop(palette);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn heap_bytes_counts_a_shared_cached_palette_once_across_clones() {
        let mut palette = PalettedContainer::new(BLOCKS, 0);
        palette.set(1, 1);
        let base_heap = palette.heap_bytes();
        let budget = Arc::new(SavedPaletteBudget::new(1 << 20));
        palette.pack_nbt(Some(&budget), |id| Tag::Int(id as i32));
        let charge = budget.used();
        let clones = [palette.clone(), palette.clone(), palette.clone()];
        let cached: usize = std::iter::once(&palette)
            .chain(&clones)
            .map(|holder| holder.heap_bytes() - holder.storage_heap_bytes())
            .sum();
        // Four holders: the entry counts once, not four times (less the rounding).
        assert!(
            cached <= charge && cached + 4 > charge,
            "{cached} of {charge}"
        );
        drop(clones);
        assert_eq!(palette.heap_bytes(), base_heap + charge);
    }

    #[test]
    fn pack_nbt_uniform_and_collapsed_sections_bypass_cache() {
        let budget = Arc::new(SavedPaletteBudget::new(1 << 20));
        let name = |id| Tag::Int(id as i32);

        let single = PalettedContainer::new(BLOCKS, 5);
        let mut collapsed = PalettedContainer::new(BLOCKS, 5);
        collapsed.set(100, 9);
        collapsed.set(100, 5);
        let global = global_uniform(3000);

        for container in [&single, &collapsed, &global] {
            assert_eq!(
                container.pack_nbt(Some(&budget), name),
                container.pack_nbt(None, name)
            );
            assert!(cached_saved_nbt(container).is_none());
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn pack_nbt_budget_cap_and_retry_after_drop() {
        let name = |id| Tag::Int(id as i32);
        let mut words = vec![0; storage_len(4, BLOCKS.entries())];
        words[0] = 0x10;
        let load = || PalettedContainer::from_packed(BLOCKS, vec![3, 4], Some(&words)).unwrap();
        let first = load();
        let second = load();
        let budget = Arc::new(SavedPaletteBudget::new(3000));
        assert_eq!(budget.used(), 0);
        assert!(cached_saved_nbt(&first).is_none());

        let expected = first.pack_nbt(Some(&budget), name);
        assert!(cached_saved_nbt(&first).is_some());
        assert!(budget.used() <= budget.limit());

        // Over cap: packs identically without admitting into the cache.
        assert_eq!(second.pack_nbt(Some(&budget), name), expected);
        assert!(cached_saved_nbt(&second).is_none());

        // Dropping the first owner frees its budget so the second can be admitted.
        drop(first);
        assert_eq!(budget.used(), 0);
        assert_eq!(second.pack_nbt(Some(&budget), name), expected);
        assert!(cached_saved_nbt(&second).is_some());
        drop(second);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn pack_nbt_matches_pack_across_cases_and_isolates_budgets() {
        let budget_a = Arc::new(SavedPaletteBudget::new(1 << 20));
        let budget_b = Arc::new(SavedPaletteBudget::new(1 << 20));
        let name = |id| Tag::Int(id as i32);
        for (label, container) in pack_cases() {
            let uncached = container.pack_nbt(None, name);
            let first_a = container.pack_nbt(Some(&budget_a), name);
            let hit_a = container.pack_nbt(Some(&budget_a), name);
            let other_b = container.pack_nbt(Some(&budget_b), name);
            assert_eq!(first_a, uncached, "{label}: first cached save");
            assert_eq!(hit_a, uncached, "{label}: repeated cached save");
            assert_eq!(other_b, uncached, "{label}: isolated budget save");
            assert_eq!(budget_b.used(), 0, "{label}: second budget unaffected");
        }
        assert_eq!(budget_a.used(), 0);
    }
    const BIOMES: Strategy = Strategy::biomes(7);

    #[test]
    fn locate_matches_division_for_every_width() {
        for bits in 1..=32u8 {
            let per = 64 / usize::from(bits);
            for i in 0..4096 {
                let expected = (i / per, ((i % per) * usize::from(bits)) as u32);
                assert_eq!(locate(bits, i), expected, "{bits} bits, entry {i}");
            }
        }
    }
    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }
    #[test]
    fn runs_read_the_stored_values() {
        let mut state = 0x9E37_79B9_7F4A_7C15;
        // One value, a list palette, and the global palette.
        for distinct in [1u64, 5, 3000] {
            let values: Vec<u32> = (0..4096)
                .map(|_| (rng(&mut state) % distinct) as u32)
                .collect();
            let container = PalettedContainer::from_values(BLOCKS, &values);
            for start in 0..4095 {
                let mut pair = [u32::MAX; 2];
                container.get_run(start, &mut pair);
                assert_eq!(pair, values[start..start + 2]);
            }
            for _ in 0..200 {
                let start = (rng(&mut state) % 4096) as usize;
                let len = (rng(&mut state) % 20) as usize;
                let len = len.min(4096 - start);
                let mut run = vec![u32::MAX; len];
                container.get_run(start, &mut run);
                assert_eq!(
                    run,
                    values[start..start + len],
                    "{distinct} values, {start}+{len}"
                );
            }
        }
    }
    #[test]
    fn four_bit_fast_paths_match_get() {
        let mut state = 0x517C_C1B7_2722_0A95;
        // Both minimal (2 values) and full (16 values) 4-bit palettes, plus 5-bit and single.
        for (distinct, expected_bits) in [(1usize, 0u8), (2, 4), (16, 4), (20, 5)] {
            let values: Vec<u32> = (0..4096)
                .map(|i| {
                    let slot = if i < distinct {
                        i
                    } else {
                        (rng(&mut state) as usize) % distinct
                    };
                    100 + slot as u32 * 7
                })
                .collect();
            let container = PalettedContainer::from_values(BLOCKS, &values);
            assert_eq!(
                container.bits(),
                expected_bits,
                "{distinct} distinct values"
            );

            let mut unpacked = vec![u32::MAX; 4096];
            container.unpack(&mut unpacked);
            assert_eq!(unpacked, values, "unpack ({distinct} values)");

            // Every 16-aligned 16-entry row hits the 4-bit `get_run` fast path.
            for row in 0..256 {
                let start = row * 16;
                let mut buf = [u32::MAX; 16];
                container.get_run(start, &mut buf);
                assert_eq!(
                    buf,
                    values[start..start + 16],
                    "aligned 16-run at {start} ({distinct} values)"
                );
            }

            // Unaligned 16-entry runs and non-16 lengths at aligned starts stay exact.
            for (start, len) in [
                (1, 16),
                (15, 16),
                (17, 16),
                (4079, 16),
                (0, 0),
                (0, 1),
                (0, 15),
                (16, 17),
                (32, 32),
            ] {
                let mut buf = vec![u32::MAX; len];
                container.get_run(start, &mut buf);
                assert_eq!(
                    buf,
                    values[start..start + len],
                    "run {start}+{len} ({distinct} values)"
                );
            }
        }
    }
    #[test]
    fn containers_stored_alike_are_the_same() {
        let values: Vec<u32> = (0..4096).map(|i| (i % 7) as u32).collect();
        let a = PalettedContainer::from_values(BLOCKS, &values);
        assert!(a.same_storage(&a.clone()));
        let mut b = a.clone();
        b.set(100, 3);
        assert!(!a.same_storage(&b));
        // the same values stored another way are not found alike (the caller unpacks those)
        let mut c = PalettedContainer::new(BLOCKS, 0);
        for (i, &v) in values.iter().enumerate() {
            c.set(i, v);
        }
        assert!(PalettedContainer::new(BLOCKS, 5).same_storage(&PalettedContainer::new(BLOCKS, 5)));
        assert!(
            !PalettedContainer::new(BLOCKS, 5).same_storage(&PalettedContainer::new(BLOCKS, 6))
        );
        let mut out = vec![0; 4096];
        c.unpack(&mut out);
        assert_eq!(out, values);
    }
    #[test]
    fn grows_through_vanilla_widths() {
        let mut c = PalettedContainer::new(BLOCKS, 0);
        assert_eq!((c.bits(), c.palette_len()), (0, 1));
        let mut expected = Vec::new();
        for n in 1..=300u32 {
            c.set(n as usize, n);
            expected.push(match n + 1 {
                2..=16 => 4,
                17..=32 => 5,
                33..=64 => 6,
                65..=128 => 7,
                129..=256 => 8,
                _ => 16,
            });
            assert_eq!(c.bits(), *expected.last().unwrap(), "after {n} values");
        }
        assert!(c.is_global());
        for n in 1..=300u32 {
            assert_eq!(c.get(n as usize), n);
        }
        let mut b = PalettedContainer::new(BIOMES, 3);
        for n in 1..=9u32 {
            b.set(n as usize, n + 3);
            let want = match n + 1 {
                2 => 1,
                3..=4 => 2,
                5..=8 => 3,
                _ => 7,
            };
            assert_eq!(b.bits(), want, "after {n} biomes");
        }
    }
    #[test]
    fn matches_a_plain_array_under_random_edits() {
        for (strategy, distinct) in [(BLOCKS, 700u64), (BLOCKS, 12), (BIOMES, 9), (BIOMES, 3)] {
            let entries = strategy.entries();
            let mut state = 0x1234_5678_9ABC_DEF1u64;
            let mut c = PalettedContainer::new(strategy, 5);
            let mut plain = vec![5u32; entries];
            for _ in 0..20_000 {
                let r = rng(&mut state);
                let i = (r % entries as u64) as usize;
                let v = ((r >> 20) % distinct) as u32 * 7;
                assert_eq!(c.set(i, v), plain[i]);
                plain[i] = v;
            }
            let mut out = vec![0; entries];
            c.unpack(&mut out);
            assert_eq!(out, plain);
            for (i, &v) in plain.iter().enumerate() {
                assert_eq!(c.get(i), v);
            }
            let packed = c.pack();
            let loaded = PalettedContainer::from_packed(
                strategy,
                packed.palette.clone(),
                packed.data.as_deref(),
            )
            .unwrap();
            assert_eq!(loaded, c);
            assert_eq!(loaded.pack(), packed);
            let again = PalettedContainer::from_values(strategy, &plain);
            assert_eq!(again, c);
            assert_eq!(again.pack(), packed);
            let (bits, palette, words) = c.to_wire();
            let wire =
                PalettedContainer::from_wire(strategy, bits, palette, words.to_vec()).unwrap();
            assert_eq!(wire, c);
            for wanted in [0, 7, 14, 35, 7 * 699, 5] {
                assert_eq!(
                    c.has(|v| v == wanted || v == wanted + 7),
                    plain.iter().any(|&v| v == wanted || v == wanted + 7),
                    "has {wanted} in {distinct} values"
                );
            }
        }
    }
    #[test]
    fn has_matches_a_scan_at_every_width() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for (strategy, distinct) in [
            (BLOCKS, 2u32),
            (BLOCKS, 3),
            (BLOCKS, 17),
            (BLOCKS, 33),
            (BLOCKS, 65),
            (BLOCKS, 129),
            (BIOMES, 2),
            (BIOMES, 3),
            (BIOMES, 5),
        ] {
            let entries = strategy.entries();
            let mut c = PalettedContainer::new(strategy, 0);
            let mut plain = vec![0u32; entries];
            for i in 0..entries {
                let v = (rng(&mut state) % u64::from(distinct)) as u32;
                c.set(i, v);
                plain[i] = v;
            }
            // Take some values away again (stale entries), the last index too.
            for gone in [0, 1, distinct - 1] {
                for i in 0..entries {
                    if plain[i] == gone {
                        c.set(i, distinct);
                        plain[i] = distinct;
                    }
                }
                for m in [3, 5, 7] {
                    let hit = |v: u32| v % m == gone % m;
                    assert_eq!(c.has(hit), plain.iter().any(|&v| hit(v)));
                }
                for wanted in 0..=distinct {
                    for other in [wanted, wanted + 1, wanted + 2, wanted + 3, wanted + 4] {
                        let hit = |v: u32| v == wanted || v == other;
                        assert_eq!(
                            c.has(hit),
                            plain.iter().any(|&v| hit(v)),
                            "{distinct} values, {wanted}/{other}, without {gone}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn has_ignores_stale_palette_entries() {
        let mut c = PalettedContainer::new(BLOCKS, 0);
        assert!(c.has(|v| v == 0) && !c.has(|v| v == 3));
        c.set(4095, 3);
        assert!(c.has(|v| v == 3));
        c.set(4095, 0);
        assert!(c.maybe_has(|v| v == 3) && !c.has(|v| v == 3));
        for v in 1..=300 {
            c.set(v as usize, v);
        }
        assert!(c.is_global() && c.has(|v| v == 300));
        c.set(300, 0);
        assert!(c.maybe_has(|v| v == 300) && !c.has(|v| v == 300));
    }
    #[test]
    fn uniform_detection_ignores_stale_palette_entries() {
        let mut c = PalettedContainer::new(BLOCKS, 0);
        c.set(10, 3);
        assert!(!c.is_uniform(0));
        c.set(10, 0);
        assert!(c.palette_len() == 2 && c.is_uniform(0) && c.uniform() == Some(0));
        assert!(!c.is_uniform(3));
        for i in 0..4096 {
            c.set(i, 3);
        }
        assert_eq!(c.uniform(), Some(3));
    }
    #[test]
    fn rejects_malformed_wire_data() {
        assert!(PalettedContainer::from_wire(BLOCKS, 4, vec![1, 2], vec![0; 255]).is_err());
        assert!(PalettedContainer::from_wire(BLOCKS, 4, vec![1, 2], vec![0x3333; 256]).is_err());
        assert!(PalettedContainer::from_wire(BLOCKS, 4, vec![], vec![0; 256]).is_err());
        assert!(PalettedContainer::from_wire(BLOCKS, 0, vec![1, 2], vec![]).is_err());
        assert!(PalettedContainer::from_wire(BIOMES, 2, vec![1, 2, 3, 4, 5], vec![0; 4]).is_err());
        assert!(PalettedContainer::from_wire(BLOCKS, 0, vec![9], vec![]).is_ok());
    }
    /// Every word of every width against the entry-at-a-time scan, over
    /// valid data, a bad entry anywhere in a word, garbage only in a word's
    /// padding fields, and the last word's padding set to all ones: the same
    /// verdict, and the first offending entry named as before.
    #[test]
    #[ignore = "daily"]
    fn index_check_matches_an_entry_by_entry_scan() {
        fn scan(strategy: Strategy, bits: u8, len: usize, words: &[u64]) -> Result<(), u32> {
            let per = 64 / usize::from(bits);
            for i in 0..strategy.entries() {
                if i / per >= words.len() {
                    break;
                }
                let index = read(words, bits, i);
                if index as usize >= len {
                    return Err(index);
                }
            }
            Ok(())
        }
        fn compare(strategy: Strategy, bits: u8, len: usize, words: &[u64], what: &str) {
            let container = PalettedContainer {
                inner: Inner::List {
                    edits: 0,
                    strategy,
                    bits,
                    words: words.into(),
                    meta: Box::new(ListMeta::new(ListPalette::from_vec(
                        (0..len as u32).collect(),
                        false,
                    ))),
                },
                uid: Uid::default(),
            };
            match (container.check_indices(), scan(strategy, bits, len, words)) {
                (Ok(()), Ok(())) => {}
                (Err(PaletteError::BadIndex { index, palette }), Err(first)) => {
                    assert_eq!((index, palette), (first, len), "{what}");
                }
                (a, b) => panic!("{what}: {a:?} vs {b:?}"),
            }
        }
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for strategy in [BLOCKS, BIOMES] {
            let entries = strategy.entries();
            let max_len = match strategy.kind {
                Kind::Blocks => 256,
                Kind::Biomes => 8,
            };
            for len in 2..=max_len {
                let bits = match strategy.config(ceil_log2(len)) {
                    Config::List(bits) => bits,
                    _ => unreachable!("{len} entries in a single-value palette"),
                };
                let per = 64 / usize::from(bits);
                let words_len = storage_len(bits, entries);
                let valid = pack_words(bits, entries, (0..entries).map(|i| (i % len) as u32));
                let name = format!("{len} entries, {bits} bits");
                compare(strategy, bits, len, &valid, &name);
                if len == 1 << bits {
                    // No field can hold an index past the palette.
                    continue;
                }
                let top = mask(bits) as usize;
                // A bad entry in the first, a middle and the last field of a
                // word, and in the container's last entry.
                for at in [0, per / 2, per - 1, entries / 2, entries - 1] {
                    let mut bad = valid.clone();
                    let index = (len + (rng(&mut state) % 7) as usize).min(top);
                    write(&mut bad, bits, at, index as u32);
                    compare(strategy, bits, len, &bad, &format!("{name}, bad at {at}"));
                }
                // Garbage in the last word's padding fields only.
                let rest = entries - (words_len - 1) * per;
                if rest < per {
                    let mut padded = valid.clone();
                    padded[words_len - 1] |= u64::MAX << (rest * usize::from(bits));
                    compare(strategy, bits, len, &padded, &format!("{name}, padding"));
                }
                // A whole random word, every field drawn from `0..=len`.
                for _ in 0..64 {
                    let mut words = valid.clone();
                    let word = (rng(&mut state) % words_len as u64) as usize;
                    words[word] = 0;
                    for field in 0..per {
                        let index = (rng(&mut state) % (len as u64 + 1)).min(top as u64);
                        words[word] |= index << (field * usize::from(bits));
                    }
                    compare(
                        strategy,
                        bits,
                        len,
                        &words,
                        &format!("{name}, random word {word}"),
                    );
                }
            }
        }
    }
    #[test]
    fn saved_validation_matches_entrywise_reference() {
        // Keep the previous loader as an independent entry-at-a-time reference:
        // errors, resident palette order and unused storage bits must all agree.
        fn reference(
            strategy: Strategy,
            palette: Vec<u32>,
            data: Option<&[u64]>,
        ) -> Result<PalettedContainer, PaletteError> {
            if palette.is_empty() {
                return Err(PaletteError::BadPalette { len: 0 });
            }
            let config = strategy.config(ceil_log2(palette.len()));
            let entries = strategy.entries();
            let bits = match config {
                Config::Single => return Ok(PalettedContainer::new(strategy, palette[0])),
                Config::List(bits) => bits,
                Config::Global => ceil_log2(palette.len()),
            };
            let expected = storage_len(bits, entries);
            let data = data.ok_or(PaletteError::BadStorage {
                expected,
                actual: 0,
            })?;
            if data.len() != expected {
                return Err(PaletteError::BadStorage {
                    expected,
                    actual: data.len(),
                });
            }
            let mut ids = Vec::with_capacity(entries);
            for i in 0..entries {
                let index = read(data, bits, i);
                if index as usize >= palette.len() {
                    return Err(PaletteError::BadIndex {
                        index,
                        palette: palette.len(),
                    });
                }
                ids.push(index);
            }
            Ok(match config {
                Config::List(bits) => PalettedContainer {
                    inner: PalettedContainer::make_list_inner(
                        0,
                        strategy,
                        bits,
                        palette,
                        data.to_vec(),
                        true,
                    ),
                    uid: Uid::default(),
                },
                _ => PalettedContainer::build(strategy, palette, &ids),
            })
        }
        fn compare(strategy: Strategy, palette: &[u32], data: Option<&[u64]>) {
            let expected = reference(strategy, palette.to_vec(), data);
            let actual = PalettedContainer::from_packed(strategy, palette.to_vec(), data);
            match (actual, expected) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.strategy(), b.strategy());
                    assert_eq!(a.bits(), b.bits());
                    assert_eq!(a.is_global(), b.is_global());
                    assert_eq!(a.to_wire(), b.to_wire());
                }
                (Err(a), Err(b)) => assert_eq!(a, b),
                (a, b) => panic!("loader disagreement: {a:?} vs {b:?}"),
            }
        }
        for strategy in [BLOCKS, BIOMES] {
            for len in [
                0, 1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
                256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4096, 4097,
            ] {
                let palette: Vec<u32> = (0..len).map(|i| (i as u32 * 7) % 127).collect();
                compare(strategy, &palette, None);
                compare(strategy, &palette, Some(&[]));
                if len <= 1 {
                    compare(strategy, &palette, Some(&[u64::MAX]));
                    continue;
                }
                let bits = match strategy.config(ceil_log2(len)) {
                    Config::List(bits) => bits,
                    _ => ceil_log2(len),
                };
                let entries = strategy.entries();
                let mut words = pack_words(bits, entries, (0..entries).map(|i| (i % len) as u32));
                compare(strategy, &palette, Some(&words));
                compare(strategy, &palette, Some(&words[..words.len() - 1]));
                // Nonzero padding in every word and in the final partial word
                // is legal and must be preserved by resident list palettes.
                let per = 64 / usize::from(bits);
                for (word, positions) in words.iter_mut().zip((0..entries).step_by(per)) {
                    let used = (entries - positions).min(per) * usize::from(bits);
                    if used < 64 {
                        *word |= u64::MAX << used;
                    }
                }
                compare(strategy, &palette, Some(&words));
                if len < 1usize << bits {
                    for i in [0, per - 1, per, entries / 2, entries - 1] {
                        let mut invalid = words.clone();
                        write(&mut invalid, bits, i, len as u32);
                        // A second invalid entry verifies first-error ordering.
                        write(&mut invalid, bits, entries - 1, mask(bits) as u32);
                        compare(strategy, &palette, Some(&invalid));
                    }
                }
            }
        }
    }
    #[test]
    fn division_magic_is_exact() {
        for bits in 1..=32u8 {
            let per = 64 / usize::from(bits);
            for i in 0..4096usize {
                assert_eq!(locate(bits, i).0, i / per, "bits {bits} index {i}");
            }
        }
    }

    /// The pre-optimization `unpack`, kept to compare against in the benchmark.
    fn baseline_unpack(c: &PalettedContainer, out: &mut [u32]) {
        match &c.inner {
            Inner::Single { value, .. } => out.fill(*value),
            Inner::InlineBiomes1 { palette, word, .. } => {
                for (slot, cell) in out.iter_mut().enumerate() {
                    *cell = palette[((*word >> slot) & 1) as usize];
                }
            }
            Inner::List {
                bits: stored_bits,
                words,
                meta,
                ..
            } => {
                let list = meta.palette.as_slice();
                let bits = usize::from(*stored_bits);
                let per = 64 / bits;
                let m = mask(*stored_bits);
                for (chunk, &word) in out.chunks_mut(per).zip(words.iter()) {
                    for (slot, cell) in chunk.iter_mut().enumerate() {
                        let id = ((word >> (slot * bits)) & m) as usize;
                        *cell = list[id];
                    }
                }
            }
            Inner::Global {
                bits: stored_bits,
                words,
                ..
            } => {
                let bits = usize::from(*stored_bits);
                let per = 64 / bits;
                let m = mask(*stored_bits);
                for (chunk, &word) in out.chunks_mut(per).zip(words.iter()) {
                    for (slot, cell) in chunk.iter_mut().enumerate() {
                        *cell = ((word >> (slot * bits)) & m) as u32;
                    }
                }
            }
        }
    }

    /// `unpack` against the pre-optimization implementation, for a list palette
    /// and the global palette. The result is checked before timing; 25 measured
    /// rounds alternate order and report the median and best microsecond-per-call.
    #[test]
    #[ignore = "microbenchmark; run with --ignored --nocapture"]
    fn benchmark_unpack() {
        use std::{hint::black_box, time::Instant};
        let values: Vec<u32> = (0..4096).map(|i| (i % 12) as u32).collect();
        let list = PalettedContainer::from_values(BLOCKS, &values);
        let global = PalettedContainer::from_values(
            BLOCKS,
            &(0..4096).map(|i| (i * 3) as u32).collect::<Vec<_>>(),
        );
        for (name, container) in [("list", &list), ("global", &global)] {
            let mut expected = vec![0u32; 4096];
            baseline_unpack(container, &mut expected);
            let mut out = vec![0u32; 4096];
            container.unpack(&mut out);
            assert_eq!(expected, out, "unpack {name} differs");
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..25 {
                for optimized in [round % 2 == 0, round % 2 != 0] {
                    let started = Instant::now();
                    for _ in 0..2000 {
                        if optimized {
                            container.unpack(black_box(&mut out));
                        } else {
                            baseline_unpack(black_box(container), black_box(&mut out));
                        }
                    }
                    if round > 0 {
                        times[usize::from(optimized)]
                            .push(started.elapsed().as_secs_f64() * 1e6 / 2000.0);
                    }
                }
            }
            for values in &mut times {
                values.sort_by(f64::total_cmp);
            }
            let median = |v: &[f64]| v[v.len() / 2];
            println!(
                "unpack {name} median: baseline {:.3} us optimized {:.3} us speedup {:.3}x",
                median(&times[0]),
                median(&times[1]),
                median(&times[0]) / median(&times[1])
            );
            println!(
                "unpack {name} best:   baseline {:.3} us optimized {:.3} us speedup {:.3}x",
                times[0][0],
                times[1][0],
                times[0][0] / times[1][0]
            );
        }
    }

    /// The pre-optimization `palette_ids` (scan, then a SipHash map), kept to
    /// compare against.
    fn baseline_palette_ids(values: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut palette: Vec<u32> = Vec::new();
        let mut small = true;
        let mut lookup: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut ids = Vec::with_capacity(values.len());
        for &value in values {
            let id = if small {
                match palette.iter().position(|&v| v == value) {
                    Some(id) => id as u32,
                    None => {
                        palette.push(value);
                        if palette.len() > 16 {
                            small = false;
                            lookup.extend(palette.iter().enumerate().map(|(i, &v)| (v, i as u32)));
                        }
                        palette.len() as u32 - 1
                    }
                }
            } else {
                *lookup.entry(value).or_insert_with(|| {
                    palette.push(value);
                    palette.len() as u32 - 1
                })
            };
            ids.push(id);
        }
        (palette, ids)
    }
    /// Sections like the ones saved: stone, ores and caves under dirt, grass
    /// and air; a mixed surface (plants, leaves, logs) in short runs; a
    /// global palette; ids past the table; and biomes.
    fn palette_cases() -> Vec<(&'static str, Vec<u32>)> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut terrain = Vec::with_capacity(4096);
        for i in 0..4096u32 {
            let y = i >> 8;
            let r = rng(&mut state);
            terrain.push(match y {
                0..=8 if r.is_multiple_of(40) => 2000 + (r >> 8) as u32 % 6,
                0..=8 if r.is_multiple_of(13) => 0,
                0..=8 => 1,
                9..=10 => 10,
                11 => 9,
                _ => 0,
            });
        }
        let mut surface = Vec::with_capacity(4096);
        while surface.len() < 4096 {
            let r = rng(&mut state);
            let run = 1 + (r % 8) as usize;
            let v = 100 + ((r >> 8) % 24) as u32 * 37;
            surface.extend(std::iter::repeat_n(v, run.min(4096 - surface.len())));
        }
        let global: Vec<u32> = (0..4096)
            .map(|_| (rng(&mut state) % 600) as u32 * 41)
            .collect();
        let large: Vec<u32> = (0..4096)
            .map(|i| {
                if i % 3 == 0 {
                    70_000 + (rng(&mut state) % 40) as u32
                } else {
                    i as u32 % 20
                }
            })
            .collect();
        let biomes: Vec<u32> = (0..64).map(|i| [3, 3, 3, 7, 7, 12][i % 6]).collect();
        vec![
            ("terrain", terrain),
            ("surface", surface),
            ("global", global),
            ("large ids", large),
            ("biomes", biomes),
            ("single", vec![5; 4096]),
            ("seventeen", (0..4096).map(|i| (i % 17) as u32).collect()),
        ]
    }
    #[test]
    fn palette_ids_match_first_occurrence_order() {
        for (name, values) in palette_cases() {
            assert_eq!(
                palette_ids(&values),
                baseline_palette_ids(&values),
                "{name}"
            );
            // The table is left clean for the next section.
            assert_eq!(
                palette_ids(&values),
                baseline_palette_ids(&values),
                "{name} again"
            );
        }
        DENSE_IDS.with(|t| assert!(t.borrow().iter().all(|&slot| slot == 0)));
    }
    /// `palette_ids` against the pre-optimization implementation over the
    /// cases above. 25 measured rounds alternate order and report the median
    /// microseconds per call.
    #[test]
    #[ignore = "microbenchmark; run with --ignored --nocapture"]
    fn benchmark_palette_ids() {
        use std::{hint::black_box, time::Instant};
        for (name, values) in palette_cases() {
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..25 {
                for optimized in [round % 2 == 0, round % 2 != 0] {
                    let started = Instant::now();
                    for _ in 0..500 {
                        if optimized {
                            black_box(palette_ids(black_box(&values)));
                        } else {
                            black_box(baseline_palette_ids(black_box(&values)));
                        }
                    }
                    if round > 0 {
                        times[usize::from(optimized)]
                            .push(started.elapsed().as_secs_f64() * 1e6 / 500.0);
                    }
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            let (base, new) = (times[0][times[0].len() / 2], times[1][times[1].len() / 2]);
            println!(
                "palette_ids {name}: baseline {base:.2} us, new {new:.2} us, {:.2}x",
                base / new
            );
        }
    }
    /// Entry-by-entry reference `pack_words` (pre-optimization) for differential tests.
    fn baseline_pack_words(bits: u8, entries: usize, ids: impl Iterator<Item = u32>) -> Vec<u64> {
        let mut words = vec![0u64; storage_len(bits, entries)];
        for (i, id) in ids.enumerate() {
            write(&mut words, bits, i, id);
        }
        words
    }
    /// `pack` as it was before a section of one value was answered from its
    /// words: every cell unpacked, then the distinct values found again.
    fn baseline_pack(c: &PalettedContainer) -> Packed {
        let strategy = c.strategy();
        let entries = strategy.entries();
        let mut values = vec![0u32; entries];
        c.unpack(&mut values);
        let (palette, ids) = palette_ids(&values);
        let bits = match strategy.config(ceil_log2(palette.len())) {
            Config::Single => 0,
            Config::List(bits) => bits,
            Config::Global => ceil_log2(palette.len()),
        };
        let data = (bits != 0).then(|| baseline_pack_words(bits, entries, ids.iter().copied()));
        Packed {
            palette,
            bits,
            data,
        }
    }
    /// A container holding `value` in every cell while stored in the global
    /// palette's width, which `set` keeps once a section has used it.
    fn global_uniform(value: u32) -> PalettedContainer {
        let bits = BLOCKS.global_bits;
        let width = usize::from(bits);
        let per = 64 / width;
        let pattern = (0..per).fold(0u64, |word, slot| word | u64::from(value) << (slot * width));
        PalettedContainer {
            inner: Inner::Global {
                edits: 0,
                strategy: BLOCKS,
                bits,
                words: vec![pattern; BLOCKS.entries().div_ceil(per)].into_boxed_slice(),
                saved_nbt: Box::new(OnceLock::new()),
            },
            uid: Uid::default(),
        }
    }
    /// Sections saved from a column: one value (a single palette, and a list
    /// palette whose other entry died), mixed terrain, the global palette
    /// holding one value, and biomes.
    fn pack_cases() -> Vec<(&'static str, PalettedContainer)> {
        let mixed: Vec<u32> = (0..4096).map(|i| (i % 12) as u32).collect();
        let mut dead = PalettedContainer::new(BLOCKS, 5);
        dead.set(100, 9);
        dead.set(100, 5);
        let biomes: Vec<u32> = (0..64).map(|i| (i % 6) as u32 * 2).collect();
        vec![
            ("single", PalettedContainer::new(BLOCKS, 5)),
            ("dead slot", dead),
            ("mixed", PalettedContainer::from_values(BLOCKS, &mixed)),
            (
                "terrain",
                PalettedContainer::from_values(BLOCKS, &palette_cases()[0].1),
            ),
            ("global uniform", global_uniform(3000)),
            ("biomes", PalettedContainer::from_values(BIOMES, &biomes)),
        ]
    }
    #[test]
    fn pack_matches_unpacking_every_section() {
        for (name, container) in pack_cases() {
            assert_eq!(container.pack(), baseline_pack(&container), "{name}");
        }
    }
    /// A section of one value saves as that value and no indices, whatever its
    /// stored palette looks like: a single palette, a list whose other entry was
    /// overwritten, or the global palette.
    #[test]
    fn uniform_sections_pack_as_one_entry() {
        let mut dead = PalettedContainer::new(BLOCKS, 5);
        dead.set(100, 9);
        dead.set(100, 5);
        for (name, container, value) in [
            ("single", PalettedContainer::new(BLOCKS, 5), 5),
            ("dead slot", dead, 5),
            ("global", global_uniform(3000), 3000),
        ] {
            assert_eq!(
                container.pack(),
                Packed {
                    palette: vec![value],
                    bits: 0,
                    data: None,
                },
                "{name}"
            );
        }
    }
    /// `pack` against the unpacking implementation, over the cases above. 25
    /// measured rounds alternate order and report the median microseconds per
    /// call.
    #[test]
    #[ignore = "microbenchmark; run with --ignored --nocapture"]
    fn benchmark_pack() {
        use std::{hint::black_box, time::Instant};
        for (name, container) in pack_cases() {
            let reps = if container.uniform().is_some() {
                2000
            } else {
                200
            };
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..25 {
                for optimized in [round % 2 == 0, round % 2 != 0] {
                    let started = Instant::now();
                    for _ in 0..reps {
                        if optimized {
                            black_box(container.pack());
                        } else {
                            black_box(baseline_pack(black_box(&container)));
                        }
                    }
                    if round > 0 {
                        times[usize::from(optimized)]
                            .push(started.elapsed().as_secs_f64() * 1e6 / f64::from(reps));
                    }
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            let (base, new) = (times[0][times[0].len() / 2], times[1][times[1].len() / 2]);
            println!(
                "pack {name}: baseline {base:.2} us, new {new:.2} us, {:.2}x",
                base / new
            );
        }
    }

    #[test]
    fn compact_layout_sizes() {
        assert!(std::mem::size_of::<PalettedContainer>() <= 40);
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(std::mem::size_of::<Inner>(), 32);
            assert_eq!(std::mem::size_of::<PalettedContainer>(), 40);
            assert!(std::mem::size_of::<ListMeta>() <= 64);
        }
    }

    #[test]
    fn inline_biomes1_zero_heap_and_full_api() {
        let budget = Arc::new(SavedPaletteBudget::new(1 << 20));
        let name = |id| Tag::Int(id as i32);

        // Via new + set (including large biome ids > u16::MAX).
        let (b0, b1, b2) = (70_001u32, 90_002u32, 110_003u32);
        let mut c = PalettedContainer::new(BIOMES, b0);
        assert_eq!(c.heap_bytes(), 0);
        assert_eq!(c.set(0, b0), b0);
        assert_eq!(c.bits(), 0);
        assert_eq!(c.set(17, b1), b0);
        assert!(matches!(c.inner, Inner::InlineBiomes1 { .. }));
        assert_eq!(c.heap_bytes(), 0);
        assert_eq!(c.bits(), 1);
        assert_eq!(c.palette(), Some([b0, b1].as_slice()));
        assert_eq!(c.palette_len(), 2);
        assert!(!c.is_global());
        assert_eq!(c.get(0), b0);
        assert_eq!(c.get(17), b1);
        assert_eq!(c.uniform(), None);
        assert!(!c.is_uniform(b0));
        assert!(!c.is_uniform(b1));
        assert!(c.maybe_has(|v| v == b1));
        assert!(c.has(|v| v == b1));
        assert!(!c.has(|v| v == b2));

        // Unpack, unpack_ids, unpack_ids_mapped, get_run (including empty tail).
        let mut plain = [b0; 64];
        plain[17] = b1;
        let mut unpacked = [0u32; 64];
        c.unpack(&mut unpacked);
        assert_eq!(unpacked, plain);
        let mut ids = [u16::MAX; 64];
        assert_eq!(c.unpack_ids(&mut ids), Some([b0, b1].as_slice()));
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, u16::from(i == 17));
        }
        let mut mapped = [0u8; 64];
        c.unpack_ids_mapped(&mut mapped, &[11u8, 22u8]);
        for (i, &val) in mapped.iter().enumerate() {
            assert_eq!(val, if i == 17 { 22 } else { 11 });
        }
        let mut run = [0u32; 8];
        c.get_run(15, &mut run);
        assert_eq!(run, plain[15..23]);
        c.get_run(64, &mut []);

        // Wire & packed round-trips stay inline with 0 heap bytes.
        let (bits, pal, words) = c.to_wire();
        assert_eq!(
            (bits, pal.as_slice(), words),
            (1, [b0, b1].as_slice(), [1u64 << 17].as_slice())
        );
        let from_w = PalettedContainer::from_wire(BIOMES, bits, pal, words.to_vec()).unwrap();
        assert!(matches!(from_w.inner, Inner::InlineBiomes1 { .. }));
        assert_eq!(from_w.heap_bytes(), 0);
        assert!(c.same_storage(&from_w));

        let packed = c.pack();
        assert_eq!(packed, baseline_pack(&c));
        let from_p =
            PalettedContainer::from_packed(BIOMES, packed.palette.clone(), packed.data.as_deref())
                .unwrap();
        assert!(matches!(from_p.inner, Inner::InlineBiomes1 { .. }));
        assert_eq!(from_p.heap_bytes(), 0);
        assert_eq!(from_p, c);

        let from_v = PalettedContainer::from_values(BIOMES, &plain);
        assert!(matches!(from_v.inner, Inner::InlineBiomes1 { .. }));
        assert_eq!(from_v.heap_bytes(), 0);
        assert!(from_v.same_storage(&c));

        // pack_nbt matches uncached and never charges budget for inline 2-biome sections.
        assert_eq!(c.pack_nbt(Some(&budget), name), c.pack_nbt(None, name));
        assert_eq!(budget.used(), 0);

        // Bit-0 == 1 flips palette order and word on pack().
        let mut flipped = c.clone();
        flipped.set(0, b1);
        assert_eq!(flipped.pack(), baseline_pack(&flipped));

        // Collapsed back to uniform (word == 0 and word == u64::MAX).
        let mut all_zero = c.clone();
        all_zero.set(17, b0);
        assert_eq!(all_zero.uniform(), Some(b0));
        assert!(all_zero.is_uniform(b0));
        assert!(!all_zero.is_uniform(b1));
        assert!(all_zero.maybe_has(|v| v == b1));
        assert!(!all_zero.has(|v| v == b1));
        assert_eq!(all_zero.pack(), baseline_pack(&all_zero));

        let all_one =
            PalettedContainer::from_wire(BIOMES, 1, vec![b0, b1], vec![u64::MAX]).unwrap();
        assert!(matches!(all_one.inner, Inner::InlineBiomes1 { .. }));
        assert_eq!(all_one.uniform(), Some(b1));
        assert!(all_one.is_uniform(b1));
        assert!(!all_one.is_uniform(b0));
        assert!(!all_one.has(|v| v == b0));
        assert!(all_one.has(|v| v == b1));
        assert_eq!(all_one.pack(), baseline_pack(&all_one));

        // Duplicate wire palette [7, 7] still reports uniform and packs cleanly.
        let dup = PalettedContainer::from_wire(BIOMES, 1, vec![7, 7], vec![0x5555_AAAA_0F0F_1234])
            .unwrap();
        assert!(matches!(dup.inner, Inner::InlineBiomes1 { .. }));
        assert_eq!(dup.uniform(), Some(7));
        assert!(dup.is_uniform(7));
        assert!(!dup.is_uniform(8));
        assert_eq!(dup.pack(), baseline_pack(&dup));

        // Single-entry 1-bit wire biome falls back to List and validates indices.
        let one_pal = PalettedContainer::from_wire(BIOMES, 1, vec![42], vec![0]).unwrap();
        assert_eq!(one_pal.bits(), 1);
        assert_eq!(one_pal.palette(), Some([42].as_slice()));
        assert_eq!(one_pal.uniform(), Some(42));
        assert!(PalettedContainer::from_wire(BIOMES, 1, vec![42], vec![1]).is_err());

        // Growth from InlineBiomes1 -> 2-bit -> 3-bit -> Global.
        let mut grow = c;
        let stamp_before = grow.stamp();
        assert_eq!(grow.set(3, b2), b0);
        assert_eq!(grow.bits(), 2);
        assert_eq!(grow.stamp(), (stamp_before.0, stamp_before.1 + 1));
        assert_eq!(grow.get(3), b2);
        assert_eq!(grow.get(17), b1);
        assert_eq!(grow.get(0), b0);
    }

    #[test]
    fn stamp_preserves_edits_across_resize_and_resets_on_clone() {
        let mut c = PalettedContainer::new(BLOCKS, 0);
        let (uid, edits) = c.stamp();
        assert_eq!(edits, 0);
        // No-op write still increments edits.
        c.set(0, 0);
        assert_eq!(c.stamp(), (uid, 1));
        // Grow Single -> List(4) -> List(5) -> Global.
        for n in 1..=260u32 {
            c.set(n as usize, n);
            assert_eq!(c.stamp(), (uid, 1 + n));
        }
        assert!(c.is_global());
        let cloned = c.clone();
        assert_ne!(cloned.stamp().0, uid);
        assert_eq!(cloned.stamp().1, 0);

        // Same across Biomes Single -> InlineBiomes1 -> List(2).
        let mut b = PalettedContainer::new(BIOMES, 0);
        let buid = b.stamp().0;
        b.set(0, 0);
        b.set(1, 1);
        b.set(2, 2);
        assert_eq!(b.stamp(), (buid, 3));
        let bclone = b.clone();
        assert_ne!(bclone.stamp().0, buid);
        assert_eq!(bclone.stamp().1, 0);
    }

    #[test]
    fn list_palette_inline_spill_and_trimmed_capacity() {
        // <= 9 distinct states stay inline in ListMeta with zero Vec heap bytes.
        let vals_9: Vec<u32> = (0..4096).map(|i| (i % 9) as u32).collect();
        let mut c = PalettedContainer::from_values(BLOCKS, &vals_9);
        assert_eq!(
            c.storage_heap_bytes(),
            256 * std::mem::size_of::<u64>() + std::mem::size_of::<ListMeta>()
        );
        // Adding a 10th distinct state spills cleanly to Heap(Vec<u32>) while
        // preserving every entry.
        assert_eq!(c.set(0, 99), 0);
        assert_eq!(c.palette_len(), 10);
        assert_eq!(c.get(0), 99);
        for i in 1..4096 {
            assert_eq!(c.get(i), (i % 9) as u32);
        }

        // Immutable creation paths trim Vec<u32> capacity to exact length (no doubling slack).
        for distinct in [17usize, 33, 65] {
            let vals: Vec<u32> = (0..4096).map(|i| (i % distinct) as u32).collect();
            let built = PalettedContainer::from_values(BLOCKS, &vals);
            let expected_words = storage_len(built.bits(), 4096);
            assert_eq!(
                built.storage_heap_bytes(),
                expected_words * std::mem::size_of::<u64>()
                    + std::mem::size_of::<ListMeta>()
                    + distinct * std::mem::size_of::<u32>()
            );
            let packed = built.pack();
            let loaded =
                PalettedContainer::from_packed(BLOCKS, packed.palette, packed.data.as_deref())
                    .unwrap();
            assert_eq!(loaded.storage_heap_bytes(), built.storage_heap_bytes());
        }
    }

    #[test]
    fn word_wise_pack_words_matches_entrywise_all_widths() {
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        for bits in [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20] {
            let m = mask(bits) as u32;
            for entries in [1usize, 15, 16, 17, 64, 4096] {
                let ids: Vec<u32> = (0..entries).map(|_| (rng(&mut state) as u32) & m).collect();
                let expected = baseline_pack_words(bits, entries, ids.iter().copied());
                assert_eq!(
                    pack_words_slice(bits, &ids),
                    expected,
                    "pack_words_slice bits={bits} entries={entries}"
                );
                assert_eq!(
                    pack_words(bits, entries, ids.iter().copied()),
                    expected,
                    "pack_words bits={bits} entries={entries}"
                );
            }
        }
    }

    #[test]
    fn direct_word_pack_matches_baseline_across_all_widths_and_edge_cases() {
        let mut state = 0xA076_1D64_78BD_642Fu64;
        // 0-bit Single, 1/2/3-bit Biomes, 4/5/6/7/8-bit Blocks, and 15/16-bit Global.
        let single_b = PalettedContainer::new(BLOCKS, 42);
        assert_eq!(single_b.pack(), baseline_pack(&single_b));
        let single_bio = PalettedContainer::new(BIOMES, 19);
        assert_eq!(single_bio.pack(), baseline_pack(&single_bio));

        for (strategy, bits, max_p) in [
            (BIOMES, 1u8, 2usize),
            (BIOMES, 2, 4),
            (BIOMES, 3, 8),
            (BLOCKS, 4, 16),
            (BLOCKS, 5, 32),
            (BLOCKS, 6, 64),
            (BLOCKS, 7, 128),
            (BLOCKS, 8, 256),
        ] {
            let entries = strategy.entries();
            for mode in 0..8 {
                let (pal, ids): (Vec<u32>, Vec<u32>) = match mode {
                    // Canonical first-occurrence order at full width.
                    0 => {
                        let pal = (0..max_p).map(|i| 100 + i as u32 * 3).collect();
                        let ids = (0..entries)
                            .map(|i| {
                                if i < max_p {
                                    i as u32
                                } else {
                                    (rng(&mut state) % max_p as u64) as u32
                                }
                            })
                            .collect();
                        (pal, ids)
                    }
                    // Reversed order (forces full remap).
                    1 => {
                        let pal = (0..max_p).map(|i| 200 + i as u32 * 5).collect();
                        let ids = (0..entries)
                            .map(|i| (max_p - 1 - (i % max_p)) as u32)
                            .collect();
                        (pal, ids)
                    }
                    // Orphaned middle entry (id 1 never used).
                    2 => {
                        let pal = (0..max_p).map(|i| 300 + i as u32 * 7).collect();
                        let ids = (0..entries)
                            .map(|i| {
                                let x = i % max_p;
                                if x == 1 { 0 } else { x as u32 }
                            })
                            .collect();
                        (pal, ids)
                    }
                    // Trailing orphaned entries forcing bit-width shrink (e.g., 8->4, 6->4, 3->1).
                    3 => {
                        let used = 2.min(max_p);
                        let pal = (0..max_p).map(|i| 400 + i as u32 * 11).collect();
                        let ids = (0..entries)
                            .map(|i| {
                                if i < used {
                                    i as u32
                                } else {
                                    (rng(&mut state) % used as u64) as u32
                                }
                            })
                            .collect();
                        (pal, ids)
                    }
                    // Uniform via last palette slot (collapses to Single).
                    4 => {
                        let pal = (0..max_p).map(|i| 500 + i as u32 * 13).collect();
                        let ids = vec![(max_p - 1) as u32; entries];
                        (pal, ids)
                    }
                    // Duplicate values in resident palette (deduplicated on pack).
                    5 => {
                        let half = (max_p / 2).max(1);
                        let pal = (0..max_p).map(|i| 600 + (i % half) as u32 * 17).collect();
                        let ids = (0..entries).map(|i| (i % max_p) as u32).collect();
                        (pal, ids)
                    }
                    // Canonical order with dirty padding bits in every word & tail word.
                    6 => {
                        let pal = (0..max_p).map(|i| 700 + i as u32 * 19).collect();
                        let ids = (0..entries)
                            .map(|i| {
                                if i < max_p {
                                    i as u32
                                } else {
                                    (rng(&mut state) % max_p as u64) as u32
                                }
                            })
                            .collect();
                        (pal, ids)
                    }
                    // Non-canonical order with dirty padding bits.
                    _ => {
                        let pal = (0..max_p).map(|i| 800 + i as u32 * 23).collect();
                        let ids = (0..entries)
                            .map(|i| (max_p - 1 - (i % max_p)) as u32)
                            .collect();
                        (pal, ids)
                    }
                };
                let mut words = baseline_pack_words(bits, entries, ids.iter().copied());
                if mode >= 6 {
                    let per = 64 / usize::from(bits);
                    for (word, positions) in words.iter_mut().zip((0..entries).step_by(per)) {
                        let used = (entries - positions).min(per) * usize::from(bits);
                        if used < 64 {
                            *word |= u64::MAX << used;
                        }
                    }
                }
                let container = PalettedContainer::from_wire(strategy, bits, pal, words).unwrap();
                assert_eq!(
                    container.pack(),
                    baseline_pack(&container),
                    "pack mismatch: kind={:?} bits={bits} mode={mode}",
                    strategy.kind
                );
            }
        }

        // 15-bit and 16-bit Global palettes (both dense and shrinking back to list).
        for global_bits in [15u8, 16] {
            let strat = Strategy::blocks(global_bits);
            let dense_vals: Vec<u32> = (0..4096)
                .map(|_| (rng(&mut state) % 400) as u32 * 37)
                .collect();
            let dense = PalettedContainer::from_values(strat, &dense_vals);
            assert!(dense.is_global());
            assert_eq!(dense.bits(), global_bits);
            assert_eq!(dense.pack(), baseline_pack(&dense));

            // Global container edited back down to 3 distinct values.
            let mut shrunk = dense.clone();
            for i in 0..4096 {
                shrunk.set(i, [5, 19, 77][i % 3]);
            }
            assert!(shrunk.is_global());
            assert_eq!(shrunk.pack(), baseline_pack(&shrunk));

            // Global SavedNbt cache invalidation.
            let budget = Arc::new(SavedPaletteBudget::new(1 << 20));
            let name = |id| Tag::Int(id as i32);
            let first = dense.pack_nbt(Some(&budget), name);
            assert!(cached_saved_nbt(&dense).is_some());
            assert_eq!(dense.pack_nbt(Some(&budget), name), first);
            let mut edited = dense.clone();
            edited.set(0, dense.get(0));
            assert!(cached_saved_nbt(&edited).is_some());
            edited.set(0, dense.get(0).wrapping_add(1) & mask(global_bits) as u32);
            assert!(cached_saved_nbt(&edited).is_none());
        }
    }

    #[test]
    fn table_driven_resize_matches_vanilla_entrywise_reference() {
        /// Independent entry-by-entry vanilla `PalettedContainer.onResize` reference.
        fn reference_resize(
            container: &PalettedContainer,
            target_bits: u8,
            last: u32,
        ) -> (u8, Vec<u32>, Vec<u64>, usize) {
            let strategy = container.strategy();
            let entries = strategy.entries();
            let config = strategy.config(target_bits);
            let storage_bits = strategy.storage_bits(config);
            let mut words = vec![0u64; storage_len(storage_bits, entries)];
            let mut palette = Vec::new();
            for i in 0..entries {
                let value = container.get(i);
                let id = match config {
                    Config::Single => 0,
                    Config::List(_) => match palette.iter().position(|&v| v == value) {
                        Some(pos) => pos as u32,
                        None => {
                            palette.push(value);
                            (palette.len() - 1) as u32
                        }
                    },
                    Config::Global => value,
                };
                if storage_bits > 0 {
                    write(&mut words, storage_bits, i, id);
                }
            }
            let last_id = match config {
                Config::Single => {
                    palette.push(last);
                    0
                }
                Config::List(_) => match palette.iter().position(|&v| v == last) {
                    Some(pos) => pos,
                    None => {
                        palette.push(last);
                        palette.len() - 1
                    }
                },
                Config::Global => last as usize,
            };
            (storage_bits, palette, words, last_id)
        }

        let mut state = 0xE703_7ED1_A0B4_28DBu64;
        // 1. Single -> InlineBiomes1 and Single -> List(4).
        for (strat, target_bits, last) in [(BIOMES, 1u8, 42u32), (BLOCKS, 1, 99)] {
            let mut c = PalettedContainer::new(strat, 7);
            let expected = reference_resize(&c, target_bits, last);
            let got_id = c.resize(target_bits, last);
            let (bits, pal, words) = c.to_wire();
            assert_eq!((bits, pal, words.to_vec(), got_id), expected);
        }

        // 2. InlineBiomes1 -> List(2) for word==0, word==u64::MAX, bit0==0, bit0==1, dup palette.
        for (pal, word) in [
            (vec![10u32, 20], 0u64),
            (vec![10, 20], u64::MAX),
            (vec![10, 20], 0xF0F0_1234_5678_9AB0),
            (vec![10, 20], 0xF0F0_1234_5678_9AB1),
            (vec![10, 10], 0xAAAA_5555_0F0F_F0F0),
        ] {
            let mut c = PalettedContainer::from_wire(BIOMES, 1, pal, vec![word]).unwrap();
            assert!(matches!(c.inner, Inner::InlineBiomes1 { .. }));
            let expected = reference_resize(&c, 2, 77);
            let got_id = c.resize(2, 77);
            let (bits, got_pal, got_words) = c.to_wire();
            assert_eq!((bits, got_pal, got_words.to_vec(), got_id), expected);
        }

        // 3. List -> List and List -> Global across every width (4->5->6->7->8->16, 2->3->7),
        // including reversed palettes, orphaned entries, and duplicate wire palettes.
        for (strategy, old_bits, target_bits) in [
            (BIOMES, 2u8, 3u8),
            (BIOMES, 3, 4),
            (BLOCKS, 4, 5),
            (BLOCKS, 5, 6),
            (BLOCKS, 6, 7),
            (BLOCKS, 7, 8),
            (BLOCKS, 8, 9),
        ] {
            let entries = strategy.entries();
            let max_p = 1usize << old_bits;
            for mode in 0..4 {
                let pal: Vec<u32> = if mode == 3 {
                    let half = (max_p / 2).max(1);
                    (0..max_p).map(|i| 10 + (i % half) as u32 * 5).collect()
                } else {
                    (0..max_p).map(|i| 10 + i as u32 * 5).collect()
                };
                let ids: Vec<u32> = match mode {
                    0 => (0..entries)
                        .map(|_| (rng(&mut state) % max_p as u64) as u32)
                        .collect(),
                    1 => (0..entries)
                        .map(|i| (max_p - 1 - (i % max_p)) as u32)
                        .collect(),
                    2 => (0..entries)
                        .map(|i| {
                            let x = i % max_p;
                            if x == 1 { 0 } else { x as u32 }
                        })
                        .collect(),
                    _ => (0..entries).map(|i| (i % max_p) as u32).collect(),
                };
                let words = baseline_pack_words(old_bits, entries, ids.iter().copied());
                let mut c = PalettedContainer::from_wire(strategy, old_bits, pal, words).unwrap();
                let last = 55_555u32;
                let expected = reference_resize(&c, target_bits, last);
                let got_id = c.resize(target_bits, last);
                let (bits, got_pal, got_words) = c.to_wire();
                assert_eq!(
                    (bits, got_pal, got_words.to_vec(), got_id),
                    expected,
                    "resize mismatch: {:?} {old_bits}->{target_bits} mode={mode}",
                    strategy.kind
                );
            }
        }
    }

    #[test]
    fn word_stepping_get_run_matches_get_across_all_widths() {
        let mut state = 0x94D0_49BB_1331_11EBu64;
        for (strategy, distinct, want_bits) in [
            (BIOMES, 1usize, 0u8),
            (BIOMES, 2, 1),
            (BIOMES, 4, 2),
            (BIOMES, 8, 3),
            (BIOMES, 12, 7),
            (BLOCKS, 1, 0),
            (BLOCKS, 12, 4),
            (BLOCKS, 25, 5),
            (BLOCKS, 50, 6),
            (BLOCKS, 100, 7),
            (BLOCKS, 200, 8),
            (Strategy::blocks(15), 350, 15),
            (BLOCKS, 350, 16),
        ] {
            let entries = strategy.entries();
            let values: Vec<u32> = (0..entries)
                .map(|i| {
                    let slot = if i < distinct {
                        i
                    } else {
                        (rng(&mut state) as usize) % distinct
                    };
                    50 + slot as u32 * 3
                })
                .collect();
            let container = PalettedContainer::from_values(strategy, &values);
            assert_eq!(container.bits(), want_bits);

            for len in [0usize, 1, 2, 3, 7, 12, 15, 16, 17, 32, entries] {
                if len > entries {
                    continue;
                }
                for start in [0usize, 1, 11, 12, 15, 16, entries - len] {
                    if start + len <= entries {
                        let mut buf = vec![u32::MAX; len];
                        container.get_run(start, &mut buf);
                        assert_eq!(
                            buf,
                            values[start..start + len],
                            "get_run bits={want_bits} {start}+{len}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn from_values_uniform_and_two_biome_fast_paths_match_reference() {
        // Uniform blocks and biomes.
        for (strategy, val) in [(BLOCKS, 0u32), (BLOCKS, 4097), (BIOMES, 17)] {
            let values = vec![val; strategy.entries()];
            let fast = PalettedContainer::from_values(strategy, &values);
            let (pal, ids) = palette_ids(&values);
            let reference = PalettedContainer::build(strategy, pal, &ids);
            assert_eq!(fast.to_wire(), reference.to_wire());
            assert_eq!(fast.heap_bytes(), 0);
        }
        // Two-biome fast path with first difference at index 1, middle, and 63.
        for diff_at in [1usize, 31, 63] {
            let mut values = [11u32; 64];
            for (i, cell) in values.iter_mut().enumerate().skip(diff_at) {
                if i == diff_at || i.is_multiple_of(3) {
                    *cell = 29;
                }
            }
            let fast = PalettedContainer::from_values(BIOMES, &values);
            assert!(matches!(fast.inner, Inner::InlineBiomes1 { .. }));
            assert_eq!(fast.heap_bytes(), 0);
            let (pal, ids) = palette_ids(&values);
            let reference = PalettedContainer::build(BIOMES, pal, &ids);
            assert_eq!(fast.to_wire(), reference.to_wire());
            assert_eq!(fast.pack(), baseline_pack(&fast));
        }
    }

    /// Microbenchmark of bulk `pack()`, `from_values()`, `resize()`, and
    /// 5/6-bit `get_run()` after compact storage and word-wise optimizations.
    #[test]
    #[ignore = "microbenchmark; run with --ignored --nocapture"]
    fn benchmark_bulk_ops() {
        use std::{hint::black_box, time::Instant};
        let mut state = 0xCAFE_BABE_1234_5678u64;
        let uniform_blocks = [1u32; 4096];
        let two_biomes: Vec<u32> = (0..64).map(|i| if i < 30 { 1 } else { 4 }).collect();
        let blocks_4: Vec<u32> = (0..4096)
            .map(|i| {
                if i < 12 {
                    i as u32
                } else {
                    (rng(&mut state) % 12) as u32
                }
            })
            .collect();
        let blocks_5: Vec<u32> = (0..4096)
            .map(|i| {
                if i < 24 {
                    i as u32
                } else {
                    (rng(&mut state) % 24) as u32
                }
            })
            .collect();
        let blocks_6: Vec<u32> = (0..4096)
            .map(|i| {
                if i < 48 {
                    i as u32
                } else {
                    (rng(&mut state) % 48) as u32
                }
            })
            .collect();

        let c4 = PalettedContainer::from_values(BLOCKS, &blocks_4);
        let c5 = PalettedContainer::from_values(BLOCKS, &blocks_5);
        let mut c6_orphan = PalettedContainer::from_values(BLOCKS, &blocks_6);
        for i in 0..4096 {
            if c6_orphan.get(i) == 1 {
                c6_orphan.set(i, 0);
            }
        }
        assert_eq!(c4.pack(), baseline_pack(&c4));
        assert_eq!(c5.pack(), baseline_pack(&c5));
        assert_eq!(c6_orphan.pack(), baseline_pack(&c6_orphan));

        let reps = 2000u32;
        for (name, c) in [
            ("4-bit canon", &c4),
            ("5-bit canon", &c5),
            ("6-bit orphan", &c6_orphan),
        ] {
            let t0 = Instant::now();
            for _ in 0..reps {
                black_box(c.pack());
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
            println!("pack {name}: {us:.2} us");
        }
        for (name, strat, vals) in [
            ("uniform blocks", BLOCKS, uniform_blocks.as_slice()),
            ("2-biome", BIOMES, two_biomes.as_slice()),
            ("4-bit blocks", BLOCKS, blocks_4.as_slice()),
            ("6-bit blocks", BLOCKS, blocks_6.as_slice()),
        ] {
            let t0 = Instant::now();
            for _ in 0..reps {
                black_box(PalettedContainer::from_values(strat, black_box(vals)));
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
            println!("from_values {name}: {us:.2} us");
        }
        let mut run = [0u32; 16];
        for (name, c) in [("5-bit get_run", &c5), ("6-bit get_run", &c6_orphan)] {
            let t0 = Instant::now();
            for _ in 0..reps {
                for row in 0..256 {
                    c.get_run(row * 16, black_box(&mut run));
                }
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
            println!("{name} (256x16): {us:.2} us");
        }
    }
}
