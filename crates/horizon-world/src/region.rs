//! Anvil region reading, export to a new region, and in-place chunk writes
//! as vanilla's `RegionFile` makes them (a file opened with `open_writable`).
//! Exports never overwrite a file: use a separate directory for offline
//! edits and copied .mcc data.
use horizon_core::position::ChunkPos;
use std::{
    cell::RefCell,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU32, Ordering},
};
use {
    crate::compression::{Compression, invalid},
    horizon_core::{
        nbt,
        nbt::{Limits, NamedTag},
    },
};

const SECTOR: usize = 4096;
const HEADER: usize = SECTOR * 2;
const MAX_COMPRESSED: usize = 64 * 1024 * 1024;
/// Maximum capacity retained by thread-local NBT encode/decode scratch buffers
/// between calls so oversized chunks do not pin multi-megabyte buffers.
const MAX_RETAINED_NBT_BUFFER: usize = 512 * 1024;
/// Maximum contiguous sector run buffered in a single `write_all` during
/// [`RegionFile::write_batch`].
const MAX_COALESCED_WRITE: usize = 512 * 1024;

thread_local! {
    /// Decompression storage retained by each region-reading worker.
    static NBT_DECODE_BUFFER: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// Serialization storage retained by each chunk-encoding worker.
    static NBT_ENCODE_BUFFER: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn release_nbt_buffer(bytes: &mut Vec<u8>) {
    bytes.clear();
    if bytes.capacity() > MAX_RETAINED_NBT_BUFFER {
        bytes.shrink_to(MAX_RETAINED_NBT_BUFFER);
    }
}

#[derive(Clone, Debug)]
pub struct ChunkRecord {
    pub compression: Compression,
    pub compressed: Vec<u8>,
    pub timestamp: u32,
}
impl ChunkRecord {
    pub fn from_nbt(
        root: &NamedTag,
        compression: Compression,
        timestamp: u32,
        limits: Limits,
    ) -> io::Result<Self> {
        NBT_ENCODE_BUFFER.with_borrow_mut(|bytes| {
            bytes.clear();
            let result = (|| {
                nbt::write_named_into(root, limits, bytes).map_err(|e| invalid(e.to_string()))?;
                let compressed = compression.encode(bytes)?;
                Ok(Self {
                    compression,
                    compressed,
                    timestamp,
                })
            })();
            release_nbt_buffer(bytes);
            result
        })
    }
    /// `RegionFileStorage.read`: the root read straight off the decompressing
    /// stream, so bytes after it, and a damaged stream past it, are never read.
    pub fn nbt(&self, limits: Limits) -> io::Result<NamedTag> {
        NBT_DECODE_BUFFER.with_borrow_mut(|bytes| {
            bytes.clear();
            let decoded = self
                .compression
                .decode_into(&self.compressed, limits.max_bytes, bytes);
            let result = match nbt::read_named(bytes, limits) {
                Ok((root, _)) => Ok(root),
                Err(e) => Err(decoded.err().unwrap_or_else(|| invalid(e.to_string()))),
            };
            release_nbt_buffer(bytes);
            result
        })
    }
}

pub struct RegionFile {
    file: File,
    path: PathBuf,
    position: ChunkPos,
    offsets: [u32; 1024],
    timestamps: [u32; 1024],
    /// The sectors in use (`RegionBitmap`): the header's two and every
    /// chunk's.
    used: Vec<bool>,
    /// The first free sector (`RegionBitmap`'s low watermark): allocation
    /// starts its first-fit search here instead of at sector 0.
    first_free: usize,
    /// Bitset of slots that may currently have an external `c.X.Z.mcc` file on
    /// disk (any 1-sector slot loaded from disk, or any slot written externally
    /// during this session). Avoids redundant `unlink` syscalls for inline slots.
    maybe_external: [u64; 16],
}

pub fn position_from_path(path: &Path) -> io::Result<ChunkPos> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| invalid("invalid region filename"))?;
    let parts: Vec<_> = name.split('.').collect();
    if parts.len() != 4 || parts[0] != "r" || parts[3] != "mca" {
        return Err(invalid("expected r.<x>.<z>.mca"));
    }
    let x: i32 = parts[1]
        .parse()
        .map_err(|_| invalid("invalid region x coordinate"))?;
    let z: i32 = parts[2]
        .parse()
        .map_err(|_| invalid("invalid region z coordinate"))?;
    // Regions outside this range cannot be addressed with signed chunk coordinates.
    if !(-67_108_864..=67_108_863).contains(&x) || !(-67_108_864..=67_108_863).contains(&z) {
        return Err(invalid("region coordinate outside chunk coordinate range"));
    }
    Ok(ChunkPos::new(x, z))
}

fn slot(region: ChunkPos, chunk: ChunkPos) -> io::Result<usize> {
    if chunk.region() != region {
        return Err(invalid("chunk does not belong to this region"));
    }
    let (x, z) = chunk.region_local();
    Ok(usize::from(x) + usize::from(z) * 32)
}
fn position(region: ChunkPos, index: usize) -> ChunkPos {
    ChunkPos::new(
        region.x * 32 + (index % 32) as i32,
        region.z * 32 + (index / 32) as i32,
    )
}
fn external_path(directory: &Path, chunk: ChunkPos) -> PathBuf {
    directory.join(format!("c.{}.{}.mcc", chunk.x, chunk.z))
}

fn write_region_header(
    file: &mut File,
    offsets: &[u32; 1024],
    timestamps: &[u32; 1024],
) -> io::Result<()> {
    let mut header = [0; HEADER];
    for i in 0..1024 {
        header[i * 4..i * 4 + 4].copy_from_slice(&offsets[i].to_be_bytes());
        header[SECTOR + i * 4..SECTOR + i * 4 + 4].copy_from_slice(&timestamps[i].to_be_bytes());
    }
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header)
}

/// The header of a region file, from its start: each chunk's place (sector
/// and sector count) and timestamp.
fn read_header(file: &mut File) -> io::Result<([u32; 1024], [u32; 1024])> {
    let len = file.metadata()?.len();
    // RegionFile's constructor: a short header (an empty or truncated file)
    // reads as far as it goes, the rest stays zero (no chunks).
    let mut header = [0; HEADER];
    let mut read = 0;
    while read < HEADER {
        match file.read(&mut header[read..])? {
            0 => break,
            n => read += n,
        }
    }
    let mut offsets = [0; 1024];
    let mut timestamps = [0; 1024];
    for i in 0..1024 {
        offsets[i] = u32::from_be_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
        timestamps[i] = u32::from_be_bytes(
            header[SECTOR + i * 4..SECTOR + i * 4 + 4]
                .try_into()
                .unwrap(),
        );
        if offsets[i] == 0 {
            continue;
        }
        let start = offsets[i] >> 8;
        let count = offsets[i] & 255;
        // Vanilla drops entries that overlap the header, are empty or start past
        // the end of the file (logged as invalid sectors).
        if start < 2 || count == 0 || u64::from(start) * SECTOR as u64 > len {
            offsets[i] = 0;
        }
    }
    Ok((offsets, timestamps))
}
/// `RegionFile.getChunkDataInputStream` for the chunk the header puts at
/// `offset`: a chunk the file cannot deliver is absent (`null`), not an
/// error, so a reader skips it the way vanilla does instead of giving up on
/// the whole region.
fn read_record(
    file: &mut File,
    directory: &Path,
    chunk: ChunkPos,
    offset: u32,
    timestamp: u32,
) -> io::Result<Option<ChunkRecord>> {
    if offset == 0 {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(u64::from(offset >> 8) * SECTOR as u64))?;
    let mut header = [0; 5];
    if file.read(&mut header)? < header.len() {
        return Ok(None); // "Chunk header is truncated"
    }
    let len = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
    if len == 0 {
        return Ok(None); // "Chunk is allocated, but stream is missing"
    }
    if len as u64 + 4 > (u64::from(offset & 255) * SECTOR as u64) {
        return Ok(None); // "Chunk stream is truncated"
    }
    let compression = Compression::try_from(header[4] & 127)?;
    let compressed = if header[4] & 128 != 0 {
        // `createExternalChunkInputStream`: the stub's length says nothing
        // about the stream in the `.mcc`, so a stub that also claims inline
        // data only warns; and a stub whose file is gone is a chunk that is
        // not there (`null`), not a damaged one.
        let path = external_path(directory, chunk);
        if !path.is_file() {
            return Ok(None);
        }
        let file = File::open(path)?;
        let mut data = Vec::new();
        file.take((MAX_COMPRESSED + 1) as u64)
            .read_to_end(&mut data)?;
        if data.len() > MAX_COMPRESSED {
            return Err(invalid("external chunk compressed limit exceeded"));
        }
        data
    } else {
        let mut data = vec![0; len - 1];
        match file.read_exact(&mut data) {
            Ok(()) => data,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
    };
    Ok(Some(ChunkRecord {
        compression,
        compressed,
        timestamp,
    }))
}
fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

fn flush_sector_run(file: &mut File, run_start: usize, buf: &mut Vec<u8>) -> io::Result<()> {
    file.seek(SeekFrom::Start((run_start as u64) * SECTOR as u64))?;
    file.write_all(buf)?;
    buf.clear();
    Ok(())
}

struct PendingBatchOp<'a> {
    slot: usize,
    chunk: ChunkPos,
    record: Option<&'a ChunkRecord>,
    old_offset: u32,
    old_timestamp: u32,
    new_offset: u32,
    new_timestamp: u32,
    allocated: Option<(usize, usize)>,
    new_external: bool,
    remove_old_external: bool,
}

impl RegionFile {
    pub fn position(&self) -> ChunkPos {
        self.position
    }
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_owned();
        let file = File::open(&path)?;
        Self::from_file(path, file)
    }

    /// Open for reading and in-place writes (`write`, `clear`), creating an
    /// empty region with `create`.
    pub fn open_writable(path: impl AsRef<Path>, create: bool) -> io::Result<Self> {
        let path = path.as_ref().to_owned();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(create)
            .truncate(false)
            .open(&path)?;
        Self::from_file(path, file)
    }

    fn from_file(path: PathBuf, mut file: File) -> io::Result<Self> {
        let position = position_from_path(&path)?;
        let (offsets, timestamps) = read_header(&mut file)?;
        let mut maybe_external = [0u64; 16];
        for (i, &offset) in offsets.iter().enumerate() {
            if offset != 0 && (offset & 255) == 1 {
                maybe_external[i >> 6] |= 1u64 << (i & 63);
            }
        }
        let mut region = Self {
            file,
            path,
            position,
            offsets,
            timestamps,
            used: Vec::new(),
            first_free: 0,
            maybe_external,
        };
        region.force(0, 2);
        for offset in offsets.into_iter().filter(|&o| o != 0) {
            region.force((offset >> 8) as usize, (offset & 255) as usize);
        }
        region.first_free = region
            .used
            .iter()
            .position(|&u| !u)
            .unwrap_or(region.used.len());
        Ok(region)
    }

    /// `RegionBitmap.force`.
    fn force(&mut self, start: usize, count: usize) {
        if self.used.len() < start + count {
            self.used.resize(start + count, false);
        }
        self.used[start..start + count].fill(true);
        if self.first_free >= start && self.first_free < start + count {
            self.first_free = self.used[start + count..]
                .iter()
                .position(|&u| !u)
                .map_or(self.used.len(), |i| start + count + i);
        }
    }
    /// `RegionBitmap.free`.
    fn free(&mut self, start: usize, count: usize) {
        let end = (start + count).min(self.used.len());
        if start < end {
            self.used[start..end].fill(false);
            self.first_free = self.first_free.min(start);
        }
    }
    /// `RegionBitmap.allocate`: the first run of free sectors long enough
    /// (the end of the file is free).
    fn allocate(&mut self, count: usize) -> usize {
        let mut current = self.first_free;
        loop {
            let start = (current..self.used.len())
                .find(|&s| !self.used[s])
                .unwrap_or(self.used.len());
            let end = (start..self.used.len()).find(|&s| self.used[s]);
            if end.is_none_or(|end| end - start >= count) {
                self.force(start, count);
                return start;
            }
            current = end.unwrap_or(start);
        }
    }

    /// `RegionFile.write`: the chunk goes to the first free run of sectors
    /// that holds it (an oversized one, 256 sectors or more, to its external
    /// `c.X.Z.mcc` file with a one-sector stub), the header points to it,
    /// and only then are its old sectors freed. The other chunks stay where
    /// they are. A chunk's last sector is padded with zeros.
    pub fn write(&mut self, chunk: ChunkPos, record: &ChunkRecord) -> io::Result<()> {
        let i = slot(self.position, chunk)?;
        if record.compressed.len() > MAX_COMPRESSED {
            return Err(invalid("compressed chunk limit exceeded"));
        }
        let required = (record.compressed.len() + 5).div_ceil(SECTOR);
        let external = required >= 256;
        let sectors = if external { 1 } else { required };
        let start = self.allocate(sectors);
        if start + sectors > 0x100_0000 {
            self.free(start, sectors);
            return Err(invalid("region sector address limit exceeded"));
        }
        let mut data = Vec::with_capacity(sectors * SECTOR);
        if external {
            let directory = self.path.parent().unwrap_or(Path::new("."));
            let external_file = external_path(directory, chunk);
            let temporary = directory.join(format!(".c.{}.{}.mcc.tmp", chunk.x, chunk.z));
            let written = File::create(&temporary)
                .and_then(|mut f| {
                    f.write_all(&record.compressed)?;
                    // The region's sync only covers the stub, not this external payload.
                    f.sync_all()
                })
                .and_then(|_| fs::rename(&temporary, &external_file));
            if let Err(e) = written {
                let _ = fs::remove_file(&temporary);
                self.free(start, sectors);
                return Err(e);
            }
            self.set_maybe_external(i, true);
            data.extend_from_slice(&1u32.to_be_bytes());
            data.push(record.compression as u8 | 128);
        } else {
            data.extend_from_slice(&(record.compressed.len() as u32 + 1).to_be_bytes());
            data.push(record.compression as u8);
            data.extend_from_slice(&record.compressed);
        }
        data.resize(sectors * SECTOR, 0);
        let written = self
            .file
            .seek(SeekFrom::Start((start as u64) * SECTOR as u64))
            .and_then(|_| self.file.write_all(&data));
        if let Err(e) = written {
            self.free(start, sectors);
            return Err(e);
        }
        let old = self.offsets[i];
        let old_timestamp = self.timestamps[i];
        let had_maybe_external = self.maybe_external(i);
        self.offsets[i] = (start as u32) << 8 | sectors as u32;
        self.timestamps[i] = record.timestamp;
        if let Err(e) = self.write_header() {
            self.offsets[i] = old;
            self.timestamps[i] = old_timestamp;
            self.free(start, sectors);
            return Err(e);
        }
        if old != 0 {
            self.free((old >> 8) as usize, (old & 255) as usize);
        }
        if !external && had_maybe_external {
            let path = external_path(self.path.parent().unwrap_or(Path::new(".")), chunk);
            remove_if_exists(&path)?;
            self.set_maybe_external(i, false);
        }
        Ok(())
    }

    /// Commit a batch of chunk writes (`Some`) and clears (`None`) with a
    /// single 8 KiB header rewrite, preserving the crash-safety invariant that
    /// old sectors remain reserved until the new header is written.
    ///
    /// Duplicate positions in `ops` coalesce to their last entry after every
    /// entry is validated. Returns whether the region changed.
    pub fn write_batch(
        &mut self,
        ops: &[(ChunkPos, Option<&ChunkRecord>)],
        clear_timestamp: u32,
    ) -> io::Result<bool> {
        if ops.is_empty() {
            return Ok(false);
        }

        // Phase 0: validate every op and coalesce duplicate positions (last op
        // wins) before mutating the sector bitmap or writing to disk.
        let mut effective: Vec<(usize, ChunkPos, Option<&ChunkRecord>)> =
            Vec::with_capacity(ops.len().min(1024));
        let mut seen = [u16::MAX; 1024];
        for &(chunk, record) in ops {
            let i = slot(self.position, chunk)?;
            if let Some(record) = record
                && record.compressed.len() > MAX_COMPRESSED
            {
                return Err(invalid("compressed chunk limit exceeded"));
            }
            if seen[i] == u16::MAX {
                seen[i] = effective.len() as u16;
                effective.push((i, chunk, record));
            } else {
                effective[usize::from(seen[i])] = (i, chunk, record);
            }
        }

        // Phase 1a: allocate all new sectors while keeping every old sector
        // reserved so a crash or error before the header write cannot corrupt
        // existing chunks.
        let mut pending: Vec<PendingBatchOp<'_>> = Vec::with_capacity(effective.len());
        for (i, chunk, record) in effective {
            let old_offset = self.offsets[i];
            let old_timestamp = self.timestamps[i];
            match record {
                Some(record) => {
                    let required = (record.compressed.len() + 5).div_ceil(SECTOR);
                    let new_external = required >= 256;
                    let sectors = if new_external { 1 } else { required };
                    let start = self.allocate(sectors);
                    if start + sectors > 0x100_0000 {
                        self.free(start, sectors);
                        self.free_pending(&pending);
                        return Err(invalid("region sector address limit exceeded"));
                    }
                    pending.push(PendingBatchOp {
                        slot: i,
                        chunk,
                        record: Some(record),
                        old_offset,
                        old_timestamp,
                        new_offset: (start as u32) << 8 | sectors as u32,
                        new_timestamp: record.timestamp,
                        allocated: Some((start, sectors)),
                        new_external,
                        remove_old_external: !new_external && self.maybe_external(i),
                    });
                }
                None => {
                    if old_offset == 0 {
                        continue;
                    }
                    pending.push(PendingBatchOp {
                        slot: i,
                        chunk,
                        record: None,
                        old_offset,
                        old_timestamp,
                        new_offset: 0,
                        new_timestamp: clear_timestamp,
                        allocated: None,
                        new_external: false,
                        remove_old_external: self.maybe_external(i),
                    });
                }
            }
        }
        if pending.is_empty() {
            return Ok(false);
        }

        // Phase 1b: write all newly allocated sector payloads (coalescing
        // contiguous sector runs up to MAX_COALESCED_WRITE).
        let mut run_buf = Vec::new();
        let mut run_start_sector: Option<usize> = None;
        let mut run_next_sector: usize = 0;
        for op in &pending {
            let (Some((start, sectors)), Some(record)) = (op.allocated, op.record) else {
                continue;
            };
            let bytes_len = sectors * SECTOR;
            if let Some(run_start) = run_start_sector
                && (start != run_next_sector || run_buf.len() + bytes_len > MAX_COALESCED_WRITE)
            {
                if let Err(e) = flush_sector_run(&mut self.file, run_start, &mut run_buf) {
                    self.free_pending(&pending);
                    return Err(e);
                }
                run_start_sector = None;
            }
            if run_start_sector.is_none() {
                run_start_sector = Some(start);
            }
            let chunk_start = run_buf.len();
            if op.new_external {
                run_buf.extend_from_slice(&1u32.to_be_bytes());
                run_buf.push(record.compression as u8 | 128);
            } else {
                run_buf.extend_from_slice(&(record.compressed.len() as u32 + 1).to_be_bytes());
                run_buf.push(record.compression as u8);
                run_buf.extend_from_slice(&record.compressed);
            }
            run_buf.resize(chunk_start + bytes_len, 0);
            run_next_sector = start + sectors;
        }
        if let Some(run_start) = run_start_sector
            && let Err(e) = flush_sector_run(&mut self.file, run_start, &mut run_buf)
        {
            self.free_pending(&pending);
            return Err(e);
        }

        // Phase 1c: write and sync temporary external `.mcc` files for any
        // oversized chunks before publishing them.
        let mut staged_externals: Vec<(usize, PathBuf, PathBuf)> = Vec::new();
        for op in &pending {
            if !op.new_external {
                continue;
            }
            let Some(record) = op.record else {
                continue;
            };
            let directory = self.path.parent().unwrap_or(Path::new("."));
            let temporary = directory.join(format!(".c.{}.{}.mcc.tmp", op.chunk.x, op.chunk.z));
            let external_file = external_path(directory, op.chunk);
            let written = File::create(&temporary).and_then(|mut f| {
                f.write_all(&record.compressed)?;
                // The region's sync only covers the stub, not this external payload.
                f.sync_all()
            });
            if let Err(e) = written {
                let _ = fs::remove_file(&temporary);
                for (_, staged_tmp, _) in &staged_externals {
                    let _ = fs::remove_file(staged_tmp);
                }
                self.free_pending(&pending);
                return Err(e);
            }
            staged_externals.push((op.slot, temporary, external_file));
        }

        // Phase 1d: atomically publish staged `.mcc` files before the header points to them.
        for (idx, (slot_idx, temporary, external_file)) in staged_externals.iter().enumerate() {
            if let Err(e) = fs::rename(temporary, external_file) {
                for (_, remaining_tmp, _) in &staged_externals[idx..] {
                    let _ = fs::remove_file(remaining_tmp);
                }
                self.free_pending(&pending);
                return Err(e);
            }
            self.set_maybe_external(*slot_idx, true);
        }

        // Phase 2: update header tables and commit the 8 KiB header once.
        for op in &pending {
            self.offsets[op.slot] = op.new_offset;
            self.timestamps[op.slot] = op.new_timestamp;
        }
        if let Err(e) = self.write_header() {
            for op in &pending {
                self.offsets[op.slot] = op.old_offset;
                self.timestamps[op.slot] = op.old_timestamp;
            }
            self.free_pending(&pending);
            return Err(e);
        }

        // Phase 3: free all superseded sectors and unlink obsolete `.mcc` files
        // only for slots that could have held an external payload.
        for op in &pending {
            if op.old_offset != 0 {
                self.free(
                    (op.old_offset >> 8) as usize,
                    (op.old_offset & 255) as usize,
                );
            }
        }
        let mut first_unlink_err: Option<io::Error> = None;
        for op in &pending {
            if !op.remove_old_external {
                continue;
            }
            let path = external_path(self.path.parent().unwrap_or(Path::new(".")), op.chunk);
            match remove_if_exists(&path) {
                Ok(()) => self.set_maybe_external(op.slot, false),
                Err(e) => {
                    if first_unlink_err.is_none() {
                        first_unlink_err = Some(e);
                    }
                }
            }
        }
        if let Some(e) = first_unlink_err {
            return Err(e);
        }
        Ok(true)
    }

    fn maybe_external(&self, index: usize) -> bool {
        (self.maybe_external[index >> 6] & (1u64 << (index & 63))) != 0
    }

    fn set_maybe_external(&mut self, index: usize, value: bool) {
        let mask = 1u64 << (index & 63);
        if value {
            self.maybe_external[index >> 6] |= mask;
        } else {
            self.maybe_external[index >> 6] &= !mask;
        }
    }

    fn free_pending(&mut self, pending: &[PendingBatchOp<'_>]) {
        for op in pending {
            if let Some((start, sectors)) = op.allocated {
                self.free(start, sectors);
            }
        }
    }

    /// `RegionFile.clear`: the chunk is gone; `timestamp` marks when.
    /// Returns whether the region held it.
    pub fn clear(&mut self, chunk: ChunkPos, timestamp: u32) -> io::Result<bool> {
        let i = slot(self.position, chunk)?;
        let old = self.offsets[i];
        if old == 0 {
            return Ok(false);
        }
        let old_timestamp = self.timestamps[i];
        let had_maybe_external = self.maybe_external(i);
        self.offsets[i] = 0;
        self.timestamps[i] = timestamp;
        if let Err(e) = self.write_header() {
            self.offsets[i] = old;
            self.timestamps[i] = old_timestamp;
            return Err(e);
        }
        self.free((old >> 8) as usize, (old & 255) as usize);
        if had_maybe_external {
            let path = external_path(self.path.parent().unwrap_or(Path::new(".")), chunk);
            remove_if_exists(&path)?;
            self.set_maybe_external(i, false);
        }
        Ok(true)
    }

    /// Flush the writes to the disk.
    pub fn sync(&self) -> io::Result<()> {
        self.file.sync_data()
    }

    /// `RegionFile.writeHeader`: both header sectors from memory, so the
    /// entries dropped as invalid when the file was opened are zero on disk too.
    fn write_header(&mut self) -> io::Result<()> {
        write_region_header(&mut self.file, &self.offsets, &self.timestamps)
    }

    pub fn chunks(&self) -> impl Iterator<Item = (ChunkPos, u32)> + '_ {
        self.offsets
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(i, _)| (position(self.position, i), self.timestamps[i]))
    }

    /// The chunk's record, `None` for a chunk the file does not hold or cannot
    /// deliver (`read_record`).
    pub fn read(&mut self, chunk: ChunkPos) -> io::Result<Option<ChunkRecord>> {
        let i = slot(self.position, chunk)?;
        let directory = self.path.parent().unwrap_or(Path::new("."));
        read_record(
            &mut self.file,
            directory,
            chunk,
            self.offsets[i],
            self.timestamps[i],
        )
    }

    /// The chunk as [`read`](Self::read) would give it with its `c.X.Z.mcc`
    /// taken from `directory`: `None` unless its stub marks it as stored in
    /// that file (and that file is there).
    pub fn read_external_from(
        &mut self,
        chunk: ChunkPos,
        directory: &Path,
    ) -> io::Result<Option<ChunkRecord>> {
        if !self.is_external(chunk)? {
            return Ok(None);
        }
        let i = slot(self.position, chunk)?;
        read_record(
            &mut self.file,
            directory,
            chunk,
            self.offsets[i],
            self.timestamps[i],
        )
    }

    /// Whether the chunk's stub marks it as stored in its own `c.X.Z.mcc`.
    pub fn is_external(&mut self, chunk: ChunkPos) -> io::Result<bool> {
        let offset = self.offsets[slot(self.position, chunk)?];
        if offset == 0 {
            return Ok(false);
        }
        self.file
            .seek(SeekFrom::Start(u64::from(offset >> 8) * SECTOR as u64))?;
        let mut header = [0; 5];
        Ok(self.file.read(&mut header)? == header.len() && header[4] & 128 != 0)
    }

    /// The `c.X.Z.mcc` files this region's oversized chunks are stored in
    /// (`createExternalChunkPath`): one per chunk whose stub marks it as
    /// external ([`is_external`](Self::is_external)), in the folder this
    /// region was opened from. Empty when none of its chunks is oversized.
    pub fn external_payloads(&mut self) -> io::Result<Vec<PathBuf>> {
        let directory = self.path.parent().unwrap_or(Path::new(".")).to_owned();
        let positions: Vec<ChunkPos> = self.chunks().map(|(pos, _)| pos).collect();
        let mut out = Vec::new();
        for pos in positions {
            if self.is_external(pos)? {
                out.push(external_path(&directory, pos));
            }
        }
        Ok(out)
    }

    /// Export all chunks, preserving untouched compressed bytes and timestamps.
    /// Overrides are keyed by chunk position; None deletes a chunk in the export.
    pub fn export(
        &mut self,
        destination: impl AsRef<Path>,
        changes: &[(ChunkPos, Option<ChunkRecord>)],
    ) -> io::Result<()> {
        let destination = destination.as_ref();
        if position_from_path(destination)? != self.position {
            return Err(invalid("export region coordinates differ"));
        }
        let mut replacements = BTreeChanges::new();
        for (chunk, record) in changes {
            let i = slot(self.position, *chunk)?;
            if replacements.insert(i, record.as_ref()).is_some() {
                return Err(invalid("duplicate chunk override"));
            }
        }
        let mut writer = RegionWriter::create(destination)?;
        for i in 0..1024 {
            let chunk = position(self.position, i);
            if let Some(record) = replacements.get(&i) {
                if let Some(record) = record {
                    writer.write(chunk, record)?;
                }
            } else if let Some(record) = self.read(chunk)? {
                writer.write(chunk, &record)?;
            }
        }
        writer.finish()
    }
}
type BTreeChanges<'a> = std::collections::BTreeMap<usize, Option<&'a ChunkRecord>>;

/// A region file's header kept without the file open: which chunks the file
/// holds and where, as of when it was read. For a reader of many chunks that
/// must not keep every file open (far terrain: a region file for each 32
/// chunks of the view), so that the header is read once and not once a chunk.
///
/// It goes stale when the file is written: `RegionFile::write` moves the
/// chunk, and frees its old sectors for the next chunk written. Whoever keeps
/// a header must know when that happens.
pub struct RegionHeader {
    path: PathBuf,
    position: ChunkPos,
    offsets: [u32; 1024],
    timestamps: [u32; 1024],
}
impl RegionHeader {
    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_owned();
        let position = position_from_path(&path)?;
        let (offsets, timestamps) = read_header(&mut File::open(&path)?)?;
        Ok(Self {
            path,
            position,
            offsets,
            timestamps,
        })
    }
    /// `RegionFile::read` at the place this header gives the chunk, with the
    /// file opened for this one record.
    pub fn read_chunk(&self, chunk: ChunkPos) -> io::Result<Option<ChunkRecord>> {
        let i = slot(self.position, chunk)?;
        if self.offsets[i] == 0 {
            return Ok(None);
        }
        let directory = self.path.parent().unwrap_or(Path::new("."));
        read_record(
            &mut File::open(&self.path)?,
            directory,
            chunk,
            self.offsets[i],
            self.timestamps[i],
        )
    }
}

static NEXT_TEMP: AtomicU32 = AtomicU32::new(0);

/// Writes a complete new region to a temporary sibling, then publishes it with
/// an atomic, non-overwriting hard link. Drop rolls back unpublished files.
pub struct RegionWriter {
    file: File,
    temporary: PathBuf,
    destination: PathBuf,
    position: ChunkPos,
    offsets: [u32; 1024],
    timestamps: [u32; 1024],
    next_sector: u32,
    externals: Vec<PathBuf>,
    committed: bool,
    failed: bool,
}
impl RegionWriter {
    pub fn create(destination: impl AsRef<Path>) -> io::Result<Self> {
        let destination = destination.as_ref().to_owned();
        let position = position_from_path(&destination)?;
        if destination.try_exists()? {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "destination region already exists",
            ));
        }
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let (temporary, file) = loop {
            let path = parent.join(format!(
                ".horizon-{}-{}.tmp",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => break (path, file),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        };
        let mut writer = Self {
            file,
            temporary,
            destination,
            position,
            offsets: [0; 1024],
            timestamps: [0; 1024],
            next_sector: 2,
            externals: Vec::new(),
            committed: false,
            failed: false,
        };
        writer.file.write_all(&[0; HEADER])?;
        Ok(writer)
    }

    pub fn write(&mut self, chunk: ChunkPos, record: &ChunkRecord) -> io::Result<()> {
        if self.failed {
            return Err(invalid("region writer failed previously"));
        }
        let result = self.write_record(chunk, record);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn write_record(&mut self, chunk: ChunkPos, record: &ChunkRecord) -> io::Result<()> {
        let i = slot(self.position, chunk)?;
        if self.offsets[i] != 0 {
            return Err(invalid("chunk already written"));
        }
        if record.compressed.len() > MAX_COMPRESSED {
            return Err(invalid("compressed chunk limit exceeded"));
        }
        let required = (record.compressed.len() + 5).div_ceil(SECTOR);
        let external = required >= 256;
        let sectors = if external { 1 } else { required as u32 };
        if self.next_sector + sectors > 0x100_0000 {
            return Err(invalid("region sector address limit exceeded"));
        }
        if external {
            let path = external_path(self.destination.parent().unwrap_or(Path::new(".")), chunk);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            self.externals.push(path);
            file.write_all(&record.compressed)?;
            file.sync_all()?;
        }
        self.file
            .seek(SeekFrom::Start(u64::from(self.next_sector) * SECTOR as u64))?;
        let len = if external {
            1
        } else {
            record.compressed.len() as u32 + 1
        };
        let [a, b, c, d] = len.to_be_bytes();
        let compression = record.compression as u8 | if external { 128 } else { 0 };
        self.file.write_all(&[a, b, c, d, compression])?;
        if !external {
            self.file.write_all(&record.compressed)?;
        }
        self.offsets[i] = self.next_sector << 8 | sectors;
        self.timestamps[i] = record.timestamp;
        self.next_sector += sectors;
        Ok(())
    }

    /// The `c.X.Z.mcc` files written so far beside the destination, one per
    /// oversized chunk.
    pub fn external_files(&self) -> &[PathBuf] {
        &self.externals
    }

    pub fn finish(mut self) -> io::Result<()> {
        if self.failed {
            return Err(invalid("cannot publish a failed region writer"));
        }
        self.file
            .set_len(u64::from(self.next_sector) * SECTOR as u64)?;
        write_region_header(&mut self.file, &self.offsets, &self.timestamps)?;
        self.file.sync_all()?;
        // Hard-link publication fails atomically if another writer won the name.
        fs::hard_link(&self.temporary, &self.destination)?;
        self.committed = true;
        Ok(())
    }
}
impl Drop for RegionWriter {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.temporary);
        if !self.committed {
            for path in &self.externals {
                let _ = fs::remove_file(path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-hint allocator: first fit from sector 0. The hint must keep
    /// producing exactly its placements.
    fn naive_first_fit(used: &[bool], count: usize) -> usize {
        let mut current = 0;
        loop {
            let start = (current..used.len())
                .find(|&s| !used[s])
                .unwrap_or(used.len());
            let end = (start..used.len()).find(|&s| used[s]);
            if end.is_none_or(|end| end - start >= count) {
                return start;
            }
            current = end.unwrap_or(start);
        }
    }

    #[test]
    fn allocation_matches_first_fit_from_zero() {
        let dir =
            std::env::temp_dir().join(format!("horizon-region-first-free-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.0.0.mca");
        let mut region = RegionFile::open_writable(&path, true).unwrap();

        let mut seed = 0x1234_5678u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for op in 0..120 {
            let slot = next() % 1024;
            let size = match next() % 4 {
                0 => 1, // external: one stub sector
                _ => 2,
            };
            let pos = position(region.position, slot);
            if next() % 5 == 0 {
                assert!(region.clear(pos, op as u32).is_ok());
            } else {
                let record = ChunkRecord {
                    compression: Compression::None,
                    compressed: vec![0; size * SECTOR - 5],
                    timestamp: op as u32,
                };
                assert!(region.write(pos, &record).is_ok());
            }
            // The naive bitmap from what the region placed, under the old rule.
            let mut naive = vec![false; region.used.len()];
            naive[0..2].fill(true);
            for i in 0..1024 {
                let o = region.offsets[i];
                if o != 0 {
                    let s = (o >> 8) as usize;
                    let c = (o & 255) as usize;
                    if naive.len() < s + c {
                        naive.resize(s + c, false);
                    }
                    naive[s..s + c].fill(true);
                }
            }
            assert_eq!(region.used, naive);
            assert_eq!(region.first_free, naive_first_fit(&naive, 1));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_kept_header_reads_the_chunks_of_its_file() {
        let dir =
            std::env::temp_dir().join(format!("horizon-region-header-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.-1.2.mca");
        let record = |fill: u8, len: usize| ChunkRecord {
            compression: Compression::None,
            compressed: vec![fill; len],
            timestamp: u32::from(fill),
        };
        let (first, last) = (ChunkPos::new(-32, 64), ChunkPos::new(-1, 95));
        let missing = ChunkPos::new(-31, 64);
        let mut region = RegionFile::open_writable(&path, true).unwrap();
        region.write(first, &record(7, 100)).unwrap();
        // Two sectors, after the first chunk's one.
        region.write(last, &record(9, 5000)).unwrap();

        let header = RegionHeader::read(&path).unwrap();
        for (chunk, fill, len) in [(first, 7, 100), (last, 9, 5000)] {
            let read = header.read_chunk(chunk).unwrap().unwrap();
            assert_eq!(read.compressed, vec![fill; len]);
            assert_eq!(read.timestamp, u32::from(fill));
        }
        assert!(header.read_chunk(missing).unwrap().is_none());
        // The next region's chunk.
        assert!(header.read_chunk(ChunkPos::new(0, 64)).is_err());
        assert!(RegionHeader::read(dir.join("r.5.5.mca")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Private temporary directory created atomically with a non-recursive
    /// `DirBuilder` (`0o700` on Unix) so test fixtures never reuse or delete
    /// a directory they did not create.
    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn create(prefix: &str) -> Self {
            use std::{
                collections::hash_map::RandomState,
                hash::BuildHasher,
                sync::atomic::AtomicU64,
                time::{SystemTime, UNIX_EPOCH},
            };

            static SEQ: AtomicU64 = AtomicU64::new(0);

            let base = std::env::temp_dir();
            let pid = std::process::id();
            let builder = Self::private_dir_builder();
            for attempt in 0..1024u64 {
                let seq = SEQ.fetch_add(1, Ordering::Relaxed);
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                let nonce = RandomState::new().hash_one((pid, seq, attempt, nanos));
                let path = base.join(format!("{prefix}-{pid}-{seq}-{nanos:x}-{nonce:016x}"));
                match builder.create(&path) {
                    Ok(()) => return Self(path),
                    Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(err) => panic!(
                        "failed to create temporary directory {}: {err}",
                        path.display()
                    ),
                }
            }
            panic!(
                "failed to create a unique temporary directory under {} after 1024 attempts",
                base.display()
            );
        }

        fn private_dir_builder() -> fs::DirBuilder {
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                let mut builder = fs::DirBuilder::new();
                builder.mode(0o700);
                builder
            }
            #[cfg(not(unix))]
            {
                fs::DirBuilder::new()
            }
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn assert_bitmap_matches_naive(region: &RegionFile) {
        let mut naive = vec![false; region.used.len()];
        naive[0..2].fill(true);
        for i in 0..1024 {
            let o = region.offsets[i];
            if o != 0 {
                let s = (o >> 8) as usize;
                let c = (o & 255) as usize;
                if naive.len() < s + c {
                    naive.resize(s + c, false);
                }
                naive[s..s + c].fill(true);
            }
        }
        assert_eq!(region.used, naive);
        assert_eq!(region.first_free, naive_first_fit(&naive, 1));
    }

    fn sample_record(fill: u8, len: usize, timestamp: u32) -> ChunkRecord {
        ChunkRecord {
            compression: Compression::None,
            compressed: vec![fill; len],
            timestamp,
        }
    }

    #[test]
    fn chunk_record_nbt_buffers_preserve_parity_and_cap_capacity() {
        let small = NamedTag {
            name: nbt::JavaString::default(),
            tag: nbt::Tag::compound([
                ("xPos", nbt::Tag::Int(3)),
                ("zPos", nbt::Tag::Int(-7)),
                ("ints", nbt::Tag::IntArray((0..128).collect())),
                (
                    "longs",
                    nbt::Tag::LongArray((0..512).map(|i| i * 17 - 9).collect()),
                ),
            ]),
        };
        let oversized = NamedTag {
            name: nbt::JavaString::default(),
            tag: nbt::Tag::compound([(
                "blob",
                nbt::Tag::ByteArray(vec![0x5a; MAX_RETAINED_NBT_BUFFER + 128 * 1024]),
            )]),
        };
        let limits = Limits::default();

        for codec in [Compression::Zlib, Compression::None, Compression::Lz4] {
            for tag in [&small, &oversized] {
                let raw = nbt::write_named(tag, limits).unwrap();
                let expected_compressed = codec.encode(&raw).unwrap();
                let record = ChunkRecord::from_nbt(tag, codec, 42, limits).unwrap();
                assert_eq!(record.compression, codec);
                assert_eq!(record.timestamp, 42);
                assert_eq!(record.compressed, expected_compressed);
                assert_eq!(record.nbt(limits).unwrap(), *tag);
            }
        }

        NBT_ENCODE_BUFFER.with_borrow(|buf| {
            assert!(buf.is_empty());
            assert!(buf.capacity() <= MAX_RETAINED_NBT_BUFFER);
        });
        NBT_DECODE_BUFFER.with_borrow(|buf| {
            assert!(buf.is_empty());
            assert!(buf.capacity() <= MAX_RETAINED_NBT_BUFFER);
        });

        let tight = Limits {
            max_bytes: 16,
            ..Limits::default()
        };
        assert!(ChunkRecord::from_nbt(&oversized, Compression::Zlib, 1, tight).is_err());
        NBT_ENCODE_BUFFER.with_borrow(|buf| {
            assert!(buf.is_empty());
            assert!(buf.capacity() <= MAX_RETAINED_NBT_BUFFER);
        });
    }

    #[test]
    fn write_batch_matches_sequential_writes_clears_and_externals() {
        let dir_seq = TempDirGuard::create("hz-region-seq");
        let dir_batch = TempDirGuard::create("hz-region-batch");
        let path_seq = dir_seq.path().join("r.-1.0.mca");
        let path_batch = dir_batch.path().join("r.-1.0.mca");

        let mut seq = RegionFile::open_writable(&path_seq, true).unwrap();
        let mut batch = RegionFile::open_writable(&path_batch, true).unwrap();

        // Empty batch and clears of absent chunks are no-ops.
        assert!(!batch.write_batch(&[], 10).unwrap());
        let absent = ChunkPos::new(-32, 0);
        assert!(!batch.write_batch(&[(absent, None)], 10).unwrap());
        assert_eq!(fs::metadata(&path_batch).unwrap().len(), 0);

        // Batch 1: 64 contiguous inline chunks with varying sector counts (1..=4).
        let records1: Vec<(ChunkPos, ChunkRecord)> = (0..64i32)
            .map(|i| {
                let pos = ChunkPos::new(-32 + (i % 32), i / 32);
                let sectors = (i as usize % 4) + 1;
                (
                    pos,
                    sample_record(
                        (i as u8).wrapping_add(1),
                        sectors * SECTOR - 5,
                        100 + i as u32,
                    ),
                )
            })
            .collect();
        for (pos, rec) in &records1 {
            seq.write(*pos, rec).unwrap();
        }
        let ops1: Vec<(ChunkPos, Option<&ChunkRecord>)> = records1
            .iter()
            .map(|(pos, rec)| (*pos, Some(rec)))
            .collect();
        assert!(batch.write_batch(&ops1, 0).unwrap());
        assert_bitmap_matches_naive(&batch);

        // Batch 2: overwrite inline chunks, add two external chunks, and clear several.
        let ext_a = ChunkPos::new(-30, 0);
        let ext_b = ChunkPos::new(-1, 31);
        let rec_ext_a = sample_record(0xaa, 256 * SECTOR, 200);
        let rec_ext_b = sample_record(0xbb, 255 * SECTOR - 4, 201);
        let rec_grow = sample_record(0xcc, 5 * SECTOR - 5, 202);
        let clear_a = ChunkPos::new(-31, 0);
        let clear_absent = ChunkPos::new(-2, 31);
        let batch2_ops: [(ChunkPos, Option<&ChunkRecord>); 5] = [
            (ext_a, Some(&rec_ext_a)),
            (ext_b, Some(&rec_ext_b)),
            (ChunkPos::new(-32, 0), Some(&rec_grow)),
            (clear_a, None),
            (clear_absent, None),
        ];
        for &(pos, rec) in &batch2_ops {
            match rec {
                Some(r) => seq.write(pos, r).unwrap(),
                None => {
                    seq.clear(pos, 250).unwrap();
                }
            }
        }
        assert!(batch.write_batch(&batch2_ops, 250).unwrap());
        assert_bitmap_matches_naive(&batch);
        assert!(dir_batch.path().join("c.-30.0.mcc").is_file());
        assert!(dir_batch.path().join("c.-1.31.mcc").is_file());

        // Batch 3: shrink ext_a back inline (removes .mcc), update ext_b externally.
        let rec_inline_a = sample_record(0x11, 128, 300);
        let rec_ext_b2 = sample_record(0xdd, 257 * SECTOR, 301);
        let batch3_ops: [(ChunkPos, Option<&ChunkRecord>); 2] =
            [(ext_a, Some(&rec_inline_a)), (ext_b, Some(&rec_ext_b2))];
        for &(pos, rec) in &batch3_ops {
            seq.write(pos, rec.unwrap()).unwrap();
        }
        assert!(batch.write_batch(&batch3_ops, 0).unwrap());
        assert_bitmap_matches_naive(&batch);
        assert!(!dir_batch.path().join("c.-30.0.mcc").exists());
        assert!(dir_batch.path().join("c.-1.31.mcc").is_file());
        assert!(batch.is_external(ext_b).unwrap());

        // Reopen from disk so `maybe_external` is reconstructed from header stubs,
        // then clear ext_b and ensure its `.mcc` is deleted.
        batch.sync().unwrap();
        seq.sync().unwrap();
        drop(batch);
        drop(seq);

        let mut seq = RegionFile::open_writable(&path_seq, false).unwrap();
        let mut batch = RegionFile::open_writable(&path_batch, false).unwrap();
        assert!(seq.clear(ext_b, 400).unwrap());
        assert!(batch.write_batch(&[(ext_b, None)], 400).unwrap());
        assert_bitmap_matches_naive(&batch);
        assert!(!dir_batch.path().join("c.-1.31.mcc").exists());
        drop(seq);
        drop(batch);

        // Final verification after fresh reopen: exact chunk set, timestamps, and bytes.
        let mut seq = RegionFile::open(&path_seq).unwrap();
        let mut batch = RegionFile::open(&path_batch).unwrap();
        let seq_chunks: Vec<_> = seq.chunks().collect();
        let batch_chunks: Vec<_> = batch.chunks().collect();
        assert_eq!(batch_chunks, seq_chunks);
        for (pos, _) in batch_chunks {
            let a = seq.read(pos).unwrap().unwrap();
            let b = batch.read(pos).unwrap().unwrap();
            assert_eq!(b.compression, a.compression);
            assert_eq!(b.timestamp, a.timestamp);
            assert_eq!(b.compressed, a.compressed);
        }
    }

    #[test]
    fn write_batch_coalesces_duplicates_and_preserves_state_on_errors() {
        let dir = TempDirGuard::create("hz-region-edge");
        let path = dir.path().join("r.0.0.mca");
        let mut region = RegionFile::open_writable(&path, true).unwrap();

        let a = ChunkPos::new(0, 0);
        let b = ChunkPos::new(1, 0);
        let c = ChunkPos::new(2, 0);
        let r1 = sample_record(1, 300, 10);
        let r2 = sample_record(2, 5000, 20);
        let ext = sample_record(3, 256 * SECTOR, 30);

        // Duplicate write + clear on absent chunk coalesces to a net no-op.
        assert!(
            !region
                .write_batch(&[(c, Some(&r1)), (c, None)], 55)
                .unwrap()
        );
        assert_bitmap_matches_naive(&region);
        assert!(region.read(c).unwrap().is_none());

        // Duplicate writes coalesce to last op; overridden external never creates `.mcc`.
        assert!(
            region
                .write_batch(
                    &[
                        (a, Some(&r1)),
                        (a, Some(&r2)),
                        (b, Some(&ext)),
                        (b, Some(&r1))
                    ],
                    0
                )
                .unwrap()
        );
        assert_bitmap_matches_naive(&region);
        assert!(!dir.path().join("c.1.0.mcc").exists());
        assert_eq!(region.read(a).unwrap().unwrap().compressed, r2.compressed);
        assert_eq!(region.read(b).unwrap().unwrap().compressed, r1.compressed);

        // Write then clear an existing chunk clears it; clear then write updates it.
        assert!(
            region
                .write_batch(&[(a, Some(&r1)), (a, None), (b, None), (b, Some(&r2))], 77)
                .unwrap()
        );
        assert_bitmap_matches_naive(&region);
        assert!(region.read(a).unwrap().is_none());
        assert_eq!(region.read(b).unwrap().unwrap().compressed, r2.compressed);

        // Validation error (out-of-region or oversized) leaves region completely unchanged.
        let offsets_before = region.offsets;
        let timestamps_before = region.timestamps;
        let used_before = region.used.clone();
        let first_free_before = region.first_free;
        assert!(
            region
                .write_batch(&[(c, Some(&r1)), (ChunkPos::new(32, 0), Some(&r1))], 0)
                .is_err()
        );
        assert_eq!(region.offsets, offsets_before);
        assert_eq!(region.timestamps, timestamps_before);
        assert_eq!(region.used, used_before);
        assert_eq!(region.first_free, first_free_before);

        let too_large = ChunkRecord {
            compression: Compression::None,
            compressed: vec![0; MAX_COMPRESSED + 1],
            timestamp: 99,
        };
        assert!(
            region
                .write_batch(&[(c, Some(&r1)), (a, Some(&too_large))], 0)
                .is_err()
        );
        assert_eq!(region.offsets, offsets_before);
        assert_eq!(region.timestamps, timestamps_before);
        assert_eq!(region.used, used_before);
        assert_eq!(region.first_free, first_free_before);

        // Interleave `write` (external) with `write_batch` (inline) so `maybe_external` stays synced.
        region.write(a, &ext).unwrap();
        assert!(dir.path().join("c.0.0.mcc").is_file());
        assert!(region.write_batch(&[(a, Some(&r1))], 0).unwrap());
        assert!(!dir.path().join("c.0.0.mcc").exists());
        assert_bitmap_matches_naive(&region);

        // Read-only file: Phase 1b write failure and Phase 2 clear-only header write failure
        // both roll back in-memory state cleanly.
        region.sync().unwrap();
        drop(region);
        let mut ro = RegionFile::open(&path).unwrap();
        let ro_offsets = ro.offsets;
        let ro_timestamps = ro.timestamps;
        let ro_first_free = ro.first_free;
        assert!(ro.write_batch(&[(c, Some(&r1))], 0).is_err());
        assert_eq!(ro.offsets, ro_offsets);
        assert_eq!(ro.timestamps, ro_timestamps);
        assert_eq!(ro.first_free, ro_first_free);
        assert_bitmap_matches_naive(&ro);

        // Clear-only batch reaches Phase 2 `write_header()` on a read-only fd.
        assert!(ro.write_batch(&[(b, None)], 999).is_err());
        assert_eq!(ro.offsets, ro_offsets);
        assert_eq!(ro.timestamps, ro_timestamps);
        assert_eq!(ro.first_free, ro_first_free);
        assert_bitmap_matches_naive(&ro);
    }

    #[test]
    fn write_batch_coalesces_across_buffer_limit_and_fragmented_holes() {
        let dir = TempDirGuard::create("hz-region-coalesce");
        let path = dir.path().join("r.0.0.mca");
        let mut region = RegionFile::open_writable(&path, true).unwrap();

        // 20 contiguous 8-sector chunks (640 KiB > MAX_COALESCED_WRITE) plus one
        // max-inline 255-sector chunk (~1 MiB > MAX_COALESCED_WRITE).
        let mut records: Vec<(ChunkPos, ChunkRecord)> = (0..20i32)
            .map(|i| {
                (
                    ChunkPos::new(i, 0),
                    sample_record(i as u8 + 10, 8 * SECTOR - 5, 10 + i as u32),
                )
            })
            .collect();
        records.push((
            ChunkPos::new(25, 0),
            sample_record(0xfe, 255 * SECTOR - 5, 99),
        ));
        let ops: Vec<(ChunkPos, Option<&ChunkRecord>)> =
            records.iter().map(|(p, r)| (*p, Some(r))).collect();
        assert!(region.write_batch(&ops, 0).unwrap());
        assert_bitmap_matches_naive(&region);

        // Fragment the region by clearing every even chunk, then refill with mixed sizes.
        let clear_ops: Vec<(ChunkPos, Option<&ChunkRecord>)> = (0..20usize)
            .filter(|i| i.is_multiple_of(2))
            .map(|i| (ChunkPos::new(i as i32, 0), None))
            .collect();
        assert!(region.write_batch(&clear_ops, 500).unwrap());
        assert_bitmap_matches_naive(&region);

        let refill: Vec<(ChunkPos, ChunkRecord)> = (0..10usize)
            .map(|i| {
                let sectors = if i.is_multiple_of(2) { 3 } else { 10 };
                (
                    ChunkPos::new((i * 2) as i32, 0),
                    sample_record(0x80 + i as u8, sectors * SECTOR - 5, 600 + i as u32),
                )
            })
            .collect();
        let refill_ops: Vec<(ChunkPos, Option<&ChunkRecord>)> =
            refill.iter().map(|(p, r)| (*p, Some(r))).collect();
        assert!(region.write_batch(&refill_ops, 0).unwrap());
        assert_bitmap_matches_naive(&region);
        drop(region);

        let mut reopened = RegionFile::open(&path).unwrap();
        for (pos, expected) in &refill {
            let actual = reopened.read(*pos).unwrap().unwrap();
            assert_eq!(actual.compressed, expected.compressed);
            assert_eq!(actual.timestamp, expected.timestamp);
        }
        // Surviving odd chunks and the 255-sector chunk must be intact.
        for (pos, expected) in records.iter().filter(|(p, _)| p.x == 25 || (p.x & 1) != 0) {
            let actual = reopened.read(*pos).unwrap().unwrap();
            assert_eq!(actual.compressed, expected.compressed);
            assert_eq!(actual.timestamp, expected.timestamp);
        }
    }
}
