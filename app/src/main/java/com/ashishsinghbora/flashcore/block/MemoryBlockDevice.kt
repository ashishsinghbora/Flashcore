package com.ashishsinghbora.flashcore.block

import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive

/**
 * In-Memory Sparse Block Device for automated testing, benchmarking, and simulation.
 *
 * Implements [BlockDevice] using dynamic heap-allocated sector buffers.
 * Unwritten sectors return deterministic zeroes. Sectors are allocated dynamically on write.
 *
 * Concurrency Guarantees & Thread Safety:
 * - Thread-safe for concurrent read and write operations protected by an internal monitor lock.
 * - Allocation accounting (`maxAllocatedSectors`) is strictly atomic across concurrent writers.
 * - Multi-sector writes are preflight-validated for allocation capacity before modifying storage,
 *   preventing partial state mutation on allocation-limit failure.
 * - Multi-sector reads and writes are atomic against concurrent lifecycle events (`close`, `clear`).
 * - In-memory flush records invocations without durable persistence across process termination.
 *
 * Testing Purpose:
 * Provides a fast, deterministic block storage medium for unit and integration testing
 * (MBR, GPT, FAT32, flashing pipelines) without physical hardware or disk file dependencies.
 */
class MemoryBlockDevice(
    val totalSectors: Long = DEFAULT_TOTAL_SECTORS,
    override val sectorSizeBytes: Int = DEFAULT_SECTOR_SIZE,
    @Volatile override var isConnected: Boolean = true,
    val maxAllocatedSectors: Int = DEFAULT_MAX_ALLOCATED_SECTORS
) : BlockDevice {

    companion object {
        const val DEFAULT_TOTAL_SECTORS: Long = 2097152L // Default 1 GB (2097152 * 512 bytes)
        const val DEFAULT_SECTOR_SIZE: Int = 512
        const val DEFAULT_MAX_ALLOCATED_SECTORS: Int = 1048576 // Up to 512 MB of 512-byte sectors
    }

    init {
        require(sectorSizeBytes > 0) {
            "Sector size must be strictly positive: $sectorSizeBytes"
        }
        require(totalSectors >= 0L) {
            "Total sectors cannot be negative: $totalSectors"
        }
        require(maxAllocatedSectors > 0) {
            "maxAllocatedSectors must be strictly positive: $maxAllocatedSectors"
        }
        if (totalSectors > 0L && totalSectors > Long.MAX_VALUE / sectorSizeBytes.toLong()) {
            throw IllegalArgumentException(
                "Total capacity overflows 64-bit addressable range: $totalSectors sectors * $sectorSizeBytes bytes/sector"
            )
        }
    }

    private val sectors = HashMap<Long, ByteArray>()
    private val lock = Any()

    private val _writeCount = AtomicLong(0L)
    val writeCount: Long get() = _writeCount.get()

    private val _readCount = AtomicLong(0L)
    val readCount: Long get() = _readCount.get()

    private val _flushCount = AtomicLong(0L)
    val flushCount: Long get() = _flushCount.get()

    /**
     * Simulated fault injection: if set to true, read/write/flush operations fail with IOException.
     */
    @Volatile
    var simulateIoFailure: Boolean = false

    override suspend fun capacity(): DeviceCapacity {
        checkConnected()
        return DeviceCapacity(totalSectors, sectorSizeBytes)
    }

    override suspend fun read(lba: Long, blockCount: Int, dest: ByteArray, offset: Int): Boolean {
        currentCoroutineContext().ensureActive()
        checkConnected()
        if (simulateIoFailure) throw IOException("Simulated I/O failure during read")
        validateBounds(lba, blockCount)
        val totalBytes = calculateTotalBytes(blockCount)
        validateBufferBounds(dest.size, offset, totalBytes)

        synchronized(lock) {
            checkConnected()
            for (i in 0 until blockCount) {
                val sectorLba = lba + i.toLong()
                val destOffset = offset + i * sectorSizeBytes
                val sectorData = sectors[sectorLba]
                if (sectorData != null) {
                    System.arraycopy(sectorData, 0, dest, destOffset, sectorSizeBytes)
                } else {
                    dest.fill(0, destOffset, destOffset + sectorSizeBytes)
                }
            }
        }
        _readCount.addAndGet(blockCount.toLong())
        return true
    }

    override suspend fun write(lba: Long, blockCount: Int, src: ByteArray, offset: Int): Boolean {
        currentCoroutineContext().ensureActive()
        checkConnected()
        if (simulateIoFailure) throw IOException("Simulated I/O failure during write")
        validateBounds(lba, blockCount)
        val totalBytes = calculateTotalBytes(blockCount)
        validateBufferBounds(src.size, offset, totalBytes)

        return commitWrite(lba, blockCount, src, offset)
    }

    override suspend fun writeDirectBuffer(
        lba: Long,
        blockCount: Int,
        directBuffer: ByteBuffer,
        offset: Int,
        length: Int
    ): Boolean {
        currentCoroutineContext().ensureActive()
        checkConnected()
        if (simulateIoFailure) throw IOException("Simulated I/O failure during direct write")
        validateBounds(lba, blockCount)
        val totalBytes = calculateTotalBytes(blockCount)
        validateBufferBounds(directBuffer.capacity(), offset, length)

        if (length != totalBytes) {
            throw IOException(
                "Direct buffer transfer length $length does not match requested block count $blockCount ($totalBytes bytes required)"
            )
        }

        val temp = ByteArray(length)
        val slice = directBuffer.duplicate()
        slice.position(offset)
        slice.get(temp, 0, length)

        return commitWrite(lba, blockCount, temp, 0)
    }

    /**
     * Atomically validates allocation constraints and writes all blocks to memory.
     * Preflights new sector requirements to prevent partial state mutation on failure.
     */
    private fun commitWrite(lba: Long, blockCount: Int, src: ByteArray, offset: Int): Boolean {
        synchronized(lock) {
            checkConnected()

            // 1. Preflight allocation feasibility: count how many requested LBAs are genuinely new
            var newSectorsNeeded = 0
            for (i in 0 until blockCount) {
                val sectorLba = lba + i.toLong()
                if (!sectors.containsKey(sectorLba)) {
                    newSectorsNeeded++
                }
            }

            if (sectors.size + newSectorsNeeded > maxAllocatedSectors) {
                val remaining = maxOf(0, maxAllocatedSectors - sectors.size)
                throw IOException(
                    "MemoryBlockDevice allocation limit exceeded: operation requires $newSectorsNeeded new sector(s), " +
                    "but only $remaining sector(s) remaining (limit: $maxAllocatedSectors)"
                )
            }

            // 2. Prepare sector buffers before modifying storage (defensive copies)
            val preparedSectors = ArrayList<Pair<Long, ByteArray>>(blockCount)
            for (i in 0 until blockCount) {
                val sectorLba = lba + i.toLong()
                val srcOffset = offset + (i * sectorSizeBytes)
                val sectorData = ByteArray(sectorSizeBytes)
                System.arraycopy(src, srcOffset, sectorData, 0, sectorSizeBytes)
                preparedSectors.add(Pair(sectorLba, sectorData))
            }

            // 3. Commit prepared sectors into storage
            for ((sectorLba, sectorData) in preparedSectors) {
                sectors[sectorLba] = sectorData
            }

            // 4. Update metrics
            _writeCount.addAndGet(blockCount.toLong())
            return true
        }
    }

    /**
     * Simulation hook for storage cache flushing.
     *
     * In this in-memory test implementation, all sector updates are already committed to JVM heap memory.
     * This method does not provide physical durability across process boundaries or establish custom
     * memory barriers. It increments [flushCount] and returns true to satisfy the [BlockDevice] contract
     * and allow automated tests to verify that callers invoke flush at appropriate checkpoints.
     */
    override suspend fun flush(): Boolean {
        currentCoroutineContext().ensureActive()
        checkConnected()
        if (simulateIoFailure) throw IOException("Simulated I/O failure during flush")
        _flushCount.incrementAndGet()
        return true
    }

    /**
     * Returns a defensive copy of a written sector for assertions in unit tests.
     * Returns null if the sector has never been written (representing unallocated/zeroed sector).
     */
    fun getSector(lba: Long): ByteArray? = synchronized(lock) { sectors[lba]?.copyOf() }

    /**
     * Number of non-zero sectors currently stored in memory.
     */
    val allocatedSectorCount: Int
        get() = synchronized(lock) { sectors.size }

    fun clear() {
        synchronized(lock) {
            sectors.clear()
        }
    }

    private fun checkConnected() {
        if (!isConnected) {
            throw DeviceDisconnectedException("MemoryBlockDevice is disconnected")
        }
    }

    private fun validateBounds(lba: Long, blockCount: Int) {
        if (lba < 0L) {
            throw IOException("LBA cannot be negative: $lba")
        }
        if (blockCount <= 0) {
            throw IOException("Block count must be positive: $blockCount")
        }
        // Overflow-safe bounds check: lba > totalSectors - blockCount
        if (blockCount.toLong() > totalSectors || lba > totalSectors - blockCount.toLong()) {
            val endLba = if (lba > Long.MAX_VALUE - blockCount.toLong()) Long.MAX_VALUE else lba + blockCount.toLong() - 1L
            throw IOException("Requested LBA range [$lba..$endLba] exceeds total sectors $totalSectors")
        }
    }

    private fun calculateTotalBytes(blockCount: Int): Int {
        val totalBytesLong = blockCount.toLong() * sectorSizeBytes.toLong()
        if (totalBytesLong > Int.MAX_VALUE.toLong()) {
            throw IndexOutOfBoundsException("Requested byte transfer count $totalBytesLong exceeds 32-bit integer limit")
        }
        return totalBytesLong.toInt()
    }

    private fun validateBufferBounds(bufferCapacity: Int, offset: Int, totalBytes: Int) {
        if (offset < 0 || totalBytes < 0 || offset.toLong() + totalBytes.toLong() > bufferCapacity.toLong()) {
            throw IndexOutOfBoundsException("Buffer offset $offset + $totalBytes exceeds buffer size $bufferCapacity")
        }
    }

    override fun close() {
        synchronized(lock) {
            if (!isConnected) return
            isConnected = false
            sectors.clear()
        }
    }
}
