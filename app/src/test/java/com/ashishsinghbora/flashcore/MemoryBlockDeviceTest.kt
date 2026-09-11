package com.ashishsinghbora.flashcore

import com.ashishsinghbora.flashcore.block.DeviceCapacity
import com.ashishsinghbora.flashcore.block.DeviceDisconnectedException
import com.ashishsinghbora.flashcore.block.MemoryBlockDevice
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Comprehensive Unit Test Suite for [MemoryBlockDevice].
 *
 * Verifies all storage-abstraction contracts, input validation, memory safety,
 * arithmetic overflow handling, neighboring sector isolation, direct-buffer semantics,
 * multi-sector all-or-nothing allocation failure, and high-contention concurrency.
 */
class MemoryBlockDeviceTest {

    // ------------------------------------------------------------------------
    // 1. Constructor and Geometry Validation
    // ------------------------------------------------------------------------

    @Test
    fun testConstructorAcceptsValidGeometry() {
        runBlocking {
            val devDefault = MemoryBlockDevice()
            assertEquals(MemoryBlockDevice.DEFAULT_TOTAL_SECTORS, devDefault.totalSectors)
            assertEquals(512, devDefault.sectorSizeBytes)
            assertTrue(devDefault.isConnected)
            assertEquals(0, devDefault.allocatedSectorCount)

            val devCustom = MemoryBlockDevice(totalSectors = 4096L, sectorSizeBytes = 2048)
            assertEquals(4096L, devCustom.totalSectors)
            assertEquals(2048, devCustom.sectorSizeBytes)
            assertEquals(4096L * 2048L, devCustom.capacity().totalBytes)
        }
    }

    @Test
    fun testConfigurableSectorSizes() {
        runBlocking {
            val sectorSizes = listOf(512, 1024, 2048, 4096)
            for (size in sectorSizes) {
                val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = size)
                assertEquals(size, dev.sectorSizeBytes)
                assertEquals(size, dev.capacity().sectorSizeBytes)

                val payload = ByteArray(size) { (it % 256).toByte() }
                assertTrue(dev.write(lba = 5L, blockCount = 1, src = payload))

                val readBack = ByteArray(size)
                assertTrue(dev.read(lba = 5L, blockCount = 1, dest = readBack))
                assertArrayEquals("Mismatch for sector size $size", payload, readBack)
            }
        }
    }

    @Test
    fun testConfigurableSectorCount() {
        runBlocking {
            val smallDev = MemoryBlockDevice(totalSectors = 1L, sectorSizeBytes = 512)
            assertEquals(1L, smallDev.capacity().totalSectors)

            val mediumDev = MemoryBlockDevice(totalSectors = 65536L, sectorSizeBytes = 512)
            assertEquals(65536L, mediumDev.capacity().totalSectors)

            val largeDev = MemoryBlockDevice(totalSectors = 10000000L, sectorSizeBytes = 512)
            assertEquals(10000000L, largeDev.capacity().totalSectors)
        }
    }

    @Test
    fun testCapacityCalculation() {
        runBlocking {
            // 2,097,152 sectors * 512 = 1,073,741,824 bytes = 1.00 GB
            val dev = MemoryBlockDevice(totalSectors = 2097152L, sectorSizeBytes = 512)
            val cap = dev.capacity()
            assertEquals(2097152L, cap.totalSectors)
            assertEquals(512, cap.sectorSizeBytes)
            assertEquals(1073741824L, cap.totalBytes)
            assertEquals("1.00 GB", cap.formattedCapacity)
        }
    }

    @Test
    fun testInvalidConstructorArguments() {
        // sectorSizeBytes <= 0
        assertThrows(IllegalArgumentException::class.java) {
            MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 0)
        }
        assertThrows(IllegalArgumentException::class.java) {
            MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = -512)
        }

        // totalSectors < 0
        assertThrows(IllegalArgumentException::class.java) {
            MemoryBlockDevice(totalSectors = -1L, sectorSizeBytes = 512)
        }

        // maxAllocatedSectors <= 0
        assertThrows(IllegalArgumentException::class.java) {
            MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512, maxAllocatedSectors = 0)
        }

        // Arithmetic overflow in total capacity
        assertThrows(IllegalArgumentException::class.java) {
            MemoryBlockDevice(totalSectors = Long.MAX_VALUE / 200L, sectorSizeBytes = 512)
        }
    }

    @Test
    fun testZeroCapacityDevice() {
        runBlocking {
            val zeroDev = MemoryBlockDevice(totalSectors = 0L, sectorSizeBytes = 512)
            assertEquals(0L, zeroDev.capacity().totalSectors)
            assertEquals(0L, zeroDev.capacity().totalBytes)

            // Any read or write with blockCount > 0 should fail bounds check
            assertThrows(IOException::class.java) {
                runBlocking { zeroDev.read(0L, 1, ByteArray(512)) }
            }
            assertThrows(IOException::class.java) {
                runBlocking { zeroDev.write(0L, 1, ByteArray(512)) }
            }
        }
    }

    // ------------------------------------------------------------------------
    // 2. Read and Write Semantics
    // ------------------------------------------------------------------------

    @Test
    fun testInitialContentsAreZeroes() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512)
            val dest = ByteArray(512) { 0xFF.toByte() } // Pre-fill with non-zero
            assertTrue(dev.read(lba = 10L, blockCount = 1, dest = dest))
            assertArrayEquals(ByteArray(512), dest)
            assertEquals(0, dev.allocatedSectorCount)
        }
    }

    @Test
    fun testSingleSectorWriteAndRead() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 50L, sectorSizeBytes = 512)
            val payload = ByteArray(512) { (it * 3 % 256).toByte() }

            assertTrue(dev.write(lba = 7L, blockCount = 1, src = payload))
            assertEquals(1, dev.allocatedSectorCount)
            assertEquals(1L, dev.writeCount)

            val dest = ByteArray(512)
            assertTrue(dev.read(lba = 7L, blockCount = 1, dest = dest))
            assertArrayEquals(payload, dest)
            assertEquals(1L, dev.readCount)
        }
    }

    @Test
    fun testMultiSectorWriteAndRead() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512)
            val blockCount = 5
            val payload = ByteArray(blockCount * 512) { idx -> ((idx * 7 + 13) % 256).toByte() }

            assertTrue(dev.write(lba = 20L, blockCount = blockCount, src = payload))
            assertEquals(blockCount, dev.allocatedSectorCount)
            assertEquals(blockCount.toLong(), dev.writeCount)

            val dest = ByteArray(blockCount * 512)
            assertTrue(dev.read(lba = 20L, blockCount = blockCount, dest = dest))
            assertArrayEquals(payload, dest)
            assertEquals(blockCount.toLong(), dev.readCount)
        }
    }

    @Test
    fun testOverwriteExistingSector() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 50L, sectorSizeBytes = 512)
            val initialData = ByteArray(512) { 0x11 }
            val updatedData = ByteArray(512) { 0x22 }

            assertTrue(dev.write(lba = 5L, blockCount = 1, src = initialData))
            assertEquals(1, dev.allocatedSectorCount)

            // Overwrite same sector
            assertTrue(dev.write(lba = 5L, blockCount = 1, src = updatedData))
            assertEquals("Allocated sectors count should not increase on overwrite", 1, dev.allocatedSectorCount)

            val readBack = ByteArray(512)
            assertTrue(dev.read(lba = 5L, blockCount = 1, dest = readBack))
            assertArrayEquals(updatedData, readBack)
        }
    }

    @Test
    fun testLastValidSectorWriteAndRead() {
        runBlocking {
            val totalSectors = 1000L
            val dev = MemoryBlockDevice(totalSectors = totalSectors, sectorSizeBytes = 512)
            val lastLba = totalSectors - 1L
            val payload = ByteArray(512) { 0x7E }

            assertTrue(dev.write(lba = lastLba, blockCount = 1, src = payload))
            val dest = ByteArray(512)
            assertTrue(dev.read(lba = lastLba, blockCount = 1, dest = dest))
            assertArrayEquals(payload, dest)
        }
    }

    @Test
    fun testRepeatedReadConsistency() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 20L, sectorSizeBytes = 512)
            val payload = ByteArray(512) { (it * 17 % 256).toByte() }
            dev.write(10L, 1, payload)

            val read1 = ByteArray(512)
            val read2 = ByteArray(512)
            dev.read(10L, 1, read1)
            dev.read(10L, 1, read2)

            assertArrayEquals(payload, read1)
            assertArrayEquals(read1, read2)
        }
    }

    // ------------------------------------------------------------------------
    // 3. Bounds and Error Handling
    // ------------------------------------------------------------------------

    @Test
    fun testReadBeforeBeginningNegativeLbaThrows() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512)
            val ex = assertThrows(IOException::class.java) {
                runBlocking { dev.read(lba = -1L, blockCount = 1, dest = ByteArray(512)) }
            }
            assertTrue(ex.message!!.contains("negative"))
        }
    }

    @Test
    fun testWriteBeforeBeginningNegativeLbaThrows() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512)
            val ex = assertThrows(IOException::class.java) {
                runBlocking { dev.write(lba = -1L, blockCount = 1, src = ByteArray(512)) }
            }
            assertTrue(ex.message!!.contains("negative"))
        }
    }

    @Test
    fun testReadPastEndThrows() {
        runBlocking {
            val totalSectors = 100L
            val dev = MemoryBlockDevice(totalSectors = totalSectors, sectorSizeBytes = 512)
            val ex = assertThrows(IOException::class.java) {
                runBlocking { dev.read(lba = totalSectors, blockCount = 1, dest = ByteArray(512)) }
            }
            assertTrue(ex.message!!.contains("exceeds total sectors"))
        }
    }

    @Test
    fun testWritePastEndThrows() {
        runBlocking {
            val totalSectors = 100L
            val dev = MemoryBlockDevice(totalSectors = totalSectors, sectorSizeBytes = 512)
            val ex = assertThrows(IOException::class.java) {
                runBlocking { dev.write(lba = totalSectors, blockCount = 1, src = ByteArray(512)) }
            }
            assertTrue(ex.message!!.contains("exceeds total sectors"))
        }
    }

    @Test
    fun testRequestExtendingBeyondEndThrows() {
        runBlocking {
            val totalSectors = 100L
            val dev = MemoryBlockDevice(totalSectors = totalSectors, sectorSizeBytes = 512)
            val ex = assertThrows(IOException::class.java) {
                runBlocking { dev.write(lba = totalSectors - 1L, blockCount = 2, src = ByteArray(1024)) }
            }
            assertTrue(ex.message!!.contains("exceeds total sectors"))
        }
    }

    @Test
    fun testOverflowSafeRangeValidation() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512)

            // LBA near Long.MAX_VALUE should NOT bypass bounds check via arithmetic overflow
            val ex1 = assertThrows(IOException::class.java) {
                runBlocking { dev.read(lba = Long.MAX_VALUE - 5L, blockCount = 10, dest = ByteArray(10 * 512)) }
            }
            assertTrue(ex1.message!!.contains("exceeds total sectors"))

            val ex2 = assertThrows(IOException::class.java) {
                runBlocking { dev.write(lba = Long.MAX_VALUE - 5L, blockCount = 10, src = ByteArray(10 * 512)) }
            }
            assertTrue(ex2.message!!.contains("exceeds total sectors"))
        }
    }

    @Test
    fun testBufferBoundsValidation() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512)

            // Destination buffer too small
            assertThrows(IndexOutOfBoundsException::class.java) {
                runBlocking { dev.read(lba = 0L, blockCount = 2, dest = ByteArray(512)) }
            }

            // Source buffer too small
            assertThrows(IndexOutOfBoundsException::class.java) {
                runBlocking { dev.write(lba = 0L, blockCount = 2, src = ByteArray(512)) }
            }

            // Negative offset
            assertThrows(IndexOutOfBoundsException::class.java) {
                runBlocking { dev.read(lba = 0L, blockCount = 1, dest = ByteArray(512), offset = -1) }
            }

            // Offset + bytes exceeds buffer
            assertThrows(IndexOutOfBoundsException::class.java) {
                runBlocking { dev.write(lba = 0L, blockCount = 1, src = ByteArray(512), offset = 1) }
            }
        }
    }

    // ------------------------------------------------------------------------
    // 4. Flush and Persistence Behavior
    // ------------------------------------------------------------------------

    @Test
    fun testFlushBehavior() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 50L, sectorSizeBytes = 512)
            assertEquals(0L, dev.flushCount)

            val payload = ByteArray(512) { 0x5A }
            assertTrue(dev.write(lba = 0L, blockCount = 1, src = payload))

            assertTrue(dev.flush())
            assertEquals(1L, dev.flushCount)

            val dest = ByteArray(512)
            assertTrue(dev.read(lba = 0L, blockCount = 1, dest = dest))
            assertArrayEquals(payload, dest)
        }
    }

    // ------------------------------------------------------------------------
    // 5. Corruption and Boundary Isolation (PHASE 15)
    // ------------------------------------------------------------------------

    @Test
    fun testNeighboringSectorIsolation() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 10L, sectorSizeBytes = 512)
            val sectorN = 5L
            val payload = ByteArray(512) { 0xAB.toByte() }

            // Write only to sector N
            assertTrue(dev.write(lba = sectorN, blockCount = 1, src = payload))

            // Verify sector N is written
            val readN = ByteArray(512)
            dev.read(lba = sectorN, blockCount = 1, dest = readN)
            assertArrayEquals(payload, readN)

            // Verify sector N - 1 remains untouched zeroes
            val readPrev = ByteArray(512)
            dev.read(lba = sectorN - 1L, blockCount = 1, dest = readPrev)
            assertArrayEquals("Neighboring sector N-1 must be zero", ByteArray(512), readPrev)

            // Verify sector N + 1 remains untouched zeroes
            val readNext = ByteArray(512)
            dev.read(lba = sectorN + 1L, blockCount = 1, dest = readNext)
            assertArrayEquals("Neighboring sector N+1 must be zero", ByteArray(512), readNext)
        }
    }

    @Test
    fun testBufferMutationIsolation() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 20L, sectorSizeBytes = 512)
            val src = ByteArray(512) { 0x42 }

            // Write data
            dev.write(lba = 3L, blockCount = 1, src = src)

            // Mutate caller's source array after write
            src.fill(0x00)

            // Verify stored data inside device was NOT affected
            val readBack = ByteArray(512)
            dev.read(lba = 3L, blockCount = 1, dest = readBack)
            assertEquals("Stored sector must not be corrupted by caller mutating source buffer", 0x42.toByte(), readBack[0])

            // getSector returns defensive copy
            val directSector = dev.getSector(3L)
            assertNotNull(directSector)
            directSector!![0] = 0x99.toByte()

            // Verify device storage unchanged after mutating getSector result
            val readBack2 = ByteArray(512)
            dev.read(lba = 3L, blockCount = 1, dest = readBack2)
            assertEquals(0x42.toByte(), readBack2[0])
        }
    }

    // ------------------------------------------------------------------------
    // 6. Direct Buffer Operations & Partial-Write Rejection (Points 1 & 4)
    // ------------------------------------------------------------------------

    @Test
    fun testDirectBufferWriteAndReadBack() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 50L, sectorSizeBytes = 512)
            val directBuf = ByteBuffer.allocateDirect(1024)
            val pattern = ByteArray(1024) { (it % 256).toByte() }
            directBuf.put(pattern)
            directBuf.flip()

            val success = dev.writeDirectBuffer(lba = 10L, blockCount = 2, directBuffer = directBuf, offset = 0, length = 1024)
            assertTrue(success)

            // Original buffer position should be restored
            assertEquals(0, directBuf.position())

            val readBack = ByteArray(1024)
            dev.read(lba = 10L, blockCount = 2, dest = readBack)
            assertArrayEquals(pattern, readBack)
        }
    }

    @Test
    fun testDirectBufferInsufficientLengthRejected() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 50L, sectorSizeBytes = 512)
            val directBuf = ByteBuffer.allocateDirect(1024)
            directBuf.put(ByteArray(1024) { 0x44 })
            directBuf.flip()

            // Request 2 blocks (1024 bytes required), but pass length = 512
            val ex = assertThrows(IOException::class.java) {
                runBlocking {
                    dev.writeDirectBuffer(lba = 10L, blockCount = 2, directBuffer = directBuf, offset = 0, length = 512)
                }
            }
            assertTrue(ex.message!!.contains("does not match requested block count"))

            // Assert NO storage mutation occurred (prevent silent partial write)
            assertEquals(0, dev.allocatedSectorCount)
            assertNull(dev.getSector(10L))
            assertNull(dev.getSector(11L))

            val readBack = ByteArray(1024)
            dev.read(10L, 2, readBack)
            assertArrayEquals(ByteArray(1024), readBack)
        }
    }

    @Test
    fun testDirectBufferExcessLengthRejected() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 50L, sectorSizeBytes = 512)
            val directBuf = ByteBuffer.allocateDirect(1024)
            directBuf.put(ByteArray(1024) { 0x44 })
            directBuf.flip()

            // Request 1 block (512 bytes required), but pass length = 1024
            val ex = assertThrows(IOException::class.java) {
                runBlocking {
                    dev.writeDirectBuffer(lba = 5L, blockCount = 1, directBuffer = directBuf, offset = 0, length = 1024)
                }
            }
            assertTrue(ex.message!!.contains("does not match requested block count"))
            assertEquals(0, dev.allocatedSectorCount)
            assertNull(dev.getSector(5L))
        }
    }

    @Test
    fun testDirectBufferNonMultipleLengthRejected() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 50L, sectorSizeBytes = 512)
            val directBuf = ByteBuffer.allocateDirect(1024)
            directBuf.put(ByteArray(1024))
            directBuf.flip()

            // Request 1 block (512 bytes required), but pass length = 600
            val ex = assertThrows(IOException::class.java) {
                runBlocking {
                    dev.writeDirectBuffer(lba = 5L, blockCount = 1, directBuffer = directBuf, offset = 0, length = 600)
                }
            }
            assertTrue(ex.message!!.contains("does not match requested block count"))
            assertEquals(0, dev.allocatedSectorCount)
        }
    }

    @Test
    fun testDirectBufferBlockCountOverflowRejected() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = Long.MAX_VALUE / 512L, sectorSizeBytes = 512)
            val directBuf = ByteBuffer.allocateDirect(512)

            assertThrows(IndexOutOfBoundsException::class.java) {
                runBlocking {
                    dev.writeDirectBuffer(lba = 0L, blockCount = Int.MAX_VALUE, directBuffer = directBuf, offset = 0, length = 512)
                }
            }
            assertEquals(0, dev.allocatedSectorCount)
        }
    }

    @Test
    fun testDirectBufferBoundsValidation() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 50L, sectorSizeBytes = 512)
            val directBuf = ByteBuffer.allocateDirect(512)

            // Negative offset
            assertThrows(IndexOutOfBoundsException::class.java) {
                runBlocking {
                    dev.writeDirectBuffer(0L, 1, directBuf, offset = -1, length = 512)
                }
            }

            // Offset + length exceeds capacity
            assertThrows(IndexOutOfBoundsException::class.java) {
                runBlocking {
                    dev.writeDirectBuffer(0L, 1, directBuf, offset = 100, length = 512)
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // 7. Allocation Limit & Multi-Sector Non-Partial Failure (Points 2 & 3)
    // ------------------------------------------------------------------------

    @Test
    fun testAllocationLimitEnforcement() {
        runBlocking {
            // Configure tight limit: maximum 3 sectors
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512, maxAllocatedSectors = 3)
            val dummy = ByteArray(512)

            assertTrue(dev.write(0L, 1, dummy))
            assertTrue(dev.write(1L, 1, dummy))
            assertTrue(dev.write(2L, 1, dummy))
            assertEquals(3, dev.allocatedSectorCount)

            // Writing to a 4th distinct sector must exceed limit and throw IOException
            assertThrows(IOException::class.java) {
                runBlocking { dev.write(3L, 1, dummy) }
            }

            // Overwriting existing sector 0 is permitted (does not increase count)
            assertTrue(dev.write(0L, 1, dummy))
            assertEquals(3, dev.allocatedSectorCount)
        }
    }

    @Test
    fun testMultiSectorAllocationFailureDoesNotPartiallyMutateStorage() {
        runBlocking {
            // maxAllocatedSectors = 2
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512, maxAllocatedSectors = 2)

            val initialPayload = ByteArray(1024) { 0xAA.toByte() }
            assertTrue(dev.write(lba = 10L, blockCount = 2, src = initialPayload))
            assertEquals(2, dev.allocatedSectorCount)

            // Attempt to write sectors 10, 11, 12 (sectors 10 and 11 exist, 12 is new -> requires 1 new sector, exceeds limit 2)
            val newPayload = ByteArray(1536) { 0xBB.toByte() }
            val ex = assertThrows(IOException::class.java) {
                runBlocking {
                    dev.write(lba = 10L, blockCount = 3, src = newPayload)
                }
            }
            assertTrue(ex.message!!.contains("allocation limit exceeded"))

            // CRITICAL VERIFICATION:
            // Sector 12 was NOT created
            assertNull("Sector 12 must not be allocated", dev.getSector(12L))

            // Sectors 10 and 11 must still contain original 0xAA data (NOT 0xBB)
            val check10 = ByteArray(512)
            val check11 = ByteArray(512)
            dev.read(10L, 1, check10)
            dev.read(11L, 1, check11)
            assertTrue("Sector 10 must not be corrupted by failed write", check10.all { it == 0xAA.toByte() })
            assertTrue("Sector 11 must not be corrupted by failed write", check11.all { it == 0xAA.toByte() })

            // Allocated sector count remains 2
            assertEquals(2, dev.allocatedSectorCount)
        }
    }

    @Test
    fun testMultiSectorAllocationFailureOnAllNewSectorsLeavesZeroMutation() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512, maxAllocatedSectors = 2)

            // Pre-allocate 1 sector
            assertTrue(dev.write(0L, 1, ByteArray(512) { 0x11 }))
            assertEquals(1, dev.allocatedSectorCount)

            // Attempt write of 2 new sectors (5 and 6) when only 1 remaining capacity (1 + 2 = 3 > 2)
            val ex = assertThrows(IOException::class.java) {
                runBlocking {
                    dev.write(5L, 2, ByteArray(1024) { 0x22 })
                }
            }
            assertTrue(ex.message!!.contains("allocation limit exceeded"))

            // Neither sector 5 nor sector 6 was created
            assertNull(dev.getSector(5L))
            assertNull(dev.getSector(6L))
            assertEquals(1, dev.allocatedSectorCount)
        }
    }

    @Test
    fun testRewriteExistingSectorAtCapacityLimit() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512, maxAllocatedSectors = 2)

            assertTrue(dev.write(10L, 1, ByteArray(512) { 0x11 }))
            assertTrue(dev.write(11L, 1, ByteArray(512) { 0x22 }))
            assertEquals(2, dev.allocatedSectorCount)

            // Device is at capacity limit (2 of 2).
            // Rewriting existing sector 10 must succeed without consuming new allocation
            val updated = ByteArray(512) { 0x99.toByte() }
            assertTrue(dev.write(10L, 1, updated))
            assertEquals(2, dev.allocatedSectorCount)

            val readBack = ByteArray(512)
            dev.read(10L, 1, readBack)
            assertArrayEquals(updated, readBack)
        }
    }

    // ------------------------------------------------------------------------
    // 8. High-Contention Concurrency Tests (Points 2 & 5)
    // ------------------------------------------------------------------------

    @Test
    fun testConcurrentAllocationRespectsLimitUnderContention() {
        val maxAllocated = 20
        val threadCount = 40
        val dev = MemoryBlockDevice(totalSectors = 1000L, sectorSizeBytes = 512, maxAllocatedSectors = maxAllocated)
        val barrier = CyclicBarrier(threadCount)
        val executor = Executors.newFixedThreadPool(threadCount)
        val successCount = AtomicInteger(0)
        val failureCount = AtomicInteger(0)
        val unexpectedErrors = CopyOnWriteArrayList<Throwable>()

        for (i in 0 until threadCount) {
            val sectorLba = (i + 1) * 10L
            executor.submit {
                try {
                    barrier.await(5, TimeUnit.SECONDS)
                    runBlocking {
                        val payload = ByteArray(512) { (i + 1).toByte() }
                        val ok = dev.write(lba = sectorLba, blockCount = 1, src = payload)
                        if (ok) successCount.incrementAndGet()
                    }
                } catch (e: IOException) {
                    if (e.message?.contains("allocation limit exceeded") == true) {
                        failureCount.incrementAndGet()
                    } else {
                        unexpectedErrors.add(e)
                    }
                } catch (t: Throwable) {
                    unexpectedErrors.add(t)
                }
            }
        }

        executor.shutdown()
        assertTrue("Executor should terminate within 10s", executor.awaitTermination(10, TimeUnit.SECONDS))
        assertTrue("Unexpected errors: $unexpectedErrors", unexpectedErrors.isEmpty())

        // Invariants:
        assertEquals("Allocated sectors must exactly equal limit", maxAllocated, dev.allocatedSectorCount)
        assertEquals("Successful writes must equal allocation limit", maxAllocated, successCount.get())
        assertEquals("Failed writes must equal total - limit", threadCount - maxAllocated, failureCount.get())
    }

    @Test
    fun testConcurrentRewritesOnExistingSectors() {
        runBlocking {
            val sectorCount = 5
            val dev = MemoryBlockDevice(totalSectors = 100L, sectorSizeBytes = 512, maxAllocatedSectors = sectorCount)

            // Pre-allocate all 5 sectors
            for (s in 0 until sectorCount) {
                dev.write(lba = s.toLong(), blockCount = 1, src = ByteArray(512) { 0x00 })
            }
            assertEquals(sectorCount, dev.allocatedSectorCount)

            val threadCount = 20
            val iterationsPerThread = 50
            val barrier = CyclicBarrier(threadCount)
            val executor = Executors.newFixedThreadPool(threadCount)
            val errors = CopyOnWriteArrayList<Throwable>()

            for (t in 0 until threadCount) {
                val threadByte = (t + 1).toByte()
                executor.submit {
                    try {
                        barrier.await(5, TimeUnit.SECONDS)
                        for (iter in 0 until iterationsPerThread) {
                            val targetSector = ((t + iter) % sectorCount).toLong()
                            val payload = ByteArray(512) { threadByte }
                            runBlocking {
                                val ok = dev.write(targetSector, 1, payload)
                                assertTrue(ok)
                            }
                        }
                    } catch (th: Throwable) {
                        errors.add(th)
                    }
                }
            }

            executor.shutdown()
            assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS))
            assertTrue("Errors during concurrent rewrite: $errors", errors.isEmpty())

            // Allocation count must NOT grow
            assertEquals("Allocated sectors must remain unchanged on rewrites", sectorCount, dev.allocatedSectorCount)

            // Each sector must be internally consistent (no partial/torn byte updates)
            for (s in 0 until sectorCount) {
                val buf = ByteArray(512)
                dev.read(s.toLong(), 1, buf)
                val expectedByte = buf[0]
                assertTrue("All bytes in sector must be identical to single writer payload", buf.all { it == expectedByte })
            }
        }
    }

    @Test
    fun testConcurrentReadersAndWritersDataIntegrity() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 200L, sectorSizeBytes = 512, maxAllocatedSectors = 50)
            val writerCount = 10
            val readerCount = 10
            val totalThreads = writerCount + readerCount
            val barrier = CyclicBarrier(totalThreads)
            val executor = Executors.newFixedThreadPool(totalThreads)
            val errors = CopyOnWriteArrayList<Throwable>()

            // Writers update sectors 0..9 with uniform byte values
            for (w in 0 until writerCount) {
                val writerVal = (w + 1).toByte()
                executor.submit {
                    try {
                        barrier.await(5, TimeUnit.SECONDS)
                        for (i in 0 until 50) {
                            val lba = (i % 10).toLong()
                            val payload = ByteArray(512) { writerVal }
                            runBlocking {
                                dev.write(lba, 1, payload)
                            }
                        }
                    } catch (t: Throwable) {
                        errors.add(t)
                    }
                }
            }

            // Readers read sectors 0..9 and verify they contain a uniform byte value (no torn bytes)
            for (r in 0 until readerCount) {
                executor.submit {
                    try {
                        barrier.await(5, TimeUnit.SECONDS)
                        val buf = ByteArray(512)
                        for (i in 0 until 50) {
                            val lba = (i % 10).toLong()
                            runBlocking {
                                dev.read(lba, 1, buf)
                            }
                            val b0 = buf[0]
                            assertTrue("Sector must not have torn bytes", buf.all { it == b0 })
                        }
                    } catch (t: Throwable) {
                        errors.add(t)
                    }
                }
            }

            executor.shutdown()
            assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS))
            assertTrue("Errors during reader/writer concurrency: $errors", errors.isEmpty())
        }
    }

    // ------------------------------------------------------------------------
    // 9. Lifecycle and Fault Simulation
    // ------------------------------------------------------------------------

    @Test
    fun testSimulateIoFailure() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 50L, sectorSizeBytes = 512)
            dev.simulateIoFailure = true

            assertThrows(IOException::class.java) {
                runBlocking { dev.read(0L, 1, ByteArray(512)) }
            }
            assertThrows(IOException::class.java) {
                runBlocking { dev.write(0L, 1, ByteArray(512)) }
            }
            assertThrows(IOException::class.java) {
                runBlocking { dev.flush() }
            }
            assertThrows(IOException::class.java) {
                runBlocking {
                    dev.writeDirectBuffer(0L, 1, ByteBuffer.allocateDirect(512), 0, 512)
                }
            }
        }
    }

    @Test
    fun testDeviceDisconnectedExceptionOnClose() {
        runBlocking {
            val dev = MemoryBlockDevice(totalSectors = 50L, sectorSizeBytes = 512)
            assertTrue(dev.isConnected)

            dev.close()
            assertFalse(dev.isConnected)
            assertEquals(0, dev.allocatedSectorCount)

            assertThrows(DeviceDisconnectedException::class.java) {
                runBlocking { dev.capacity() }
            }
            assertThrows(DeviceDisconnectedException::class.java) {
                runBlocking { dev.read(0L, 1, ByteArray(512)) }
            }
            assertThrows(DeviceDisconnectedException::class.java) {
                runBlocking { dev.write(0L, 1, ByteArray(512)) }
            }
            assertThrows(DeviceDisconnectedException::class.java) {
                runBlocking { dev.flush() }
            }
        }
    }

    // ------------------------------------------------------------------------
    // 10. Property / Round-Trip Data-Driven Tests (PHASE 14)
    // ------------------------------------------------------------------------

    @Test
    fun testPropertyRoundTripAcrossSectorSizes() {
        runBlocking {
            val geometries = listOf(
                Pair(512, 100L),
                Pair(1024, 50L),
                Pair(2048, 25L),
                Pair(4096, 15L)
            )

            for ((sectorSize, sectorCount) in geometries) {
                val dev = MemoryBlockDevice(totalSectors = sectorCount, sectorSizeBytes = sectorSize)

                // Write pseudo-random deterministic pattern across 3 sectors
                val blocksToWrite = 3
                val totalBytes = blocksToWrite * sectorSize
                val pattern = ByteArray(totalBytes) { idx ->
                    ((idx * 31 + sectorSize * 17 + 7) % 256).toByte()
                }

                val startLba = sectorCount / 2L
                assertTrue(dev.write(startLba, blocksToWrite, pattern))
                assertTrue(dev.flush())

                val readBack = ByteArray(totalBytes)
                assertTrue(dev.read(startLba, blocksToWrite, readBack))
                assertArrayEquals("Round-trip failed for sectorSize=$sectorSize", pattern, readBack)
            }
        }
    }
}
