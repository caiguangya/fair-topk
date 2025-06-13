#ifndef FAIR_TOPK_MEMORY_H
#define FAIR_TOPK_MEMORY_H

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <type_traits>
#include <concepts>
#include <bit>

#include <boost/predef.h>

#if BOOST_OS_WINDOWS
#include <malloc.h>
#endif

namespace FairTopK {

#ifdef __cpp_lib_hardware_interference_size
    static constexpr std::size_t CacheLineAlign = std::hardware_destructive_interference_size;
#else
    static constexpr std::size_t CacheLineAlign = 64;
#endif

template <std::size_t Align, class T>
concept LegitAlignment = (std::has_single_bit(Align)) && (Align >= alignof(T));

template<std::size_t Align> requires (std::has_single_bit(Align))
inline void *allocAligned(std::size_t size) {
	if constexpr (Align > alignof(std::max_align_t)) {
#if BOOST_OS_WINDOWS
        return _aligned_malloc(size, Align);
#else
        return std::aligned_alloc(Align, size);
#endif
	}
	else {
		return std::malloc(size);
	}
}

template<class T, std::size_t Align = CacheLineAlign> requires LegitAlignment<Align, T>
inline T *allocAligned(std::size_t num = 1) {
	return (T *)allocAligned<Align>(num * sizeof(T));
}

template<std::size_t Align = CacheLineAlign>
inline void freeAligned(void *ptr) {
#if BOOST_OS_WINDOWS
   if constexpr (Align > alignof(std::max_align_t)) 
       _aligned_free(ptr);
   else 
       std::free(ptr);
#else
    std::free(ptr);
#endif
}

template<class T, class... Args> 
inline void Initiation(T *vals, std::size_t size, Args&&... inits) {
	for (std::size_t i = 0; i < size; i++){
		new (&vals[i]) T(std::forward<Args>(inits)...);
	}
}

template<class T> requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
inline void Initiation(T *vals, std::size_t size) {
	memset(vals, 0, sizeof(T) * size);
}

template <class T, std::size_t BlockAlign = CacheLineAlign, std::size_t ObjAlign = alignof(T)> requires
LegitAlignment<BlockAlign, T> && LegitAlignment<ObjAlign, T> && (BlockAlign >= ObjAlign)
class MemoryArena {
public:
    MemoryArena(std::size_t objSize = 16384) {
		blockSize = objSize * sizeof(T);
		curBlockPos = 0;
		curBlock = allocAligned<std::int8_t, BlockAlign>(blockSize);
	}
	MemoryArena(const MemoryArena&) = delete;
	MemoryArena(MemoryArena&&) = delete;
	MemoryArena& operator=(const MemoryArena&) = delete;
	MemoryArena& operator=(MemoryArena&&) = delete;
	
	template<bool runConstructor = true, class... Args>
	T* Alloc(std::size_t size = 1, std::bool_constant<runConstructor> = std::bool_constant<runConstructor>{}, Args&&... inits) {
		constexpr std::size_t complement = ObjAlign - 1;
		std::size_t bytes = ((size * sizeof(T) + complement) & (~complement));
		if (curBlockPos + bytes > blockSize) {
			usedBlocks.push_back(curBlock);
			curBlock = allocAligned<std::int8_t, BlockAlign>(std::max(bytes, blockSize));
			curBlockPos = 0;
		}
		T *ret = (T *)(curBlock + curBlockPos);
		curBlockPos += bytes;
		
		if constexpr (runConstructor) Initiation(ret, size, std::forward<Args>(inits)...);

		return ret;
	}

	~MemoryArena() {
		freeAligned<BlockAlign>(curBlock);
		std::size_t us = usedBlocks.size();
		for (std::size_t i = 0; i < us; i++)
			freeAligned<BlockAlign>(usedBlocks[i]);
	}

private:
	std::size_t curBlockPos, blockSize;
	std::int8_t *curBlock;
	std::vector<std::int8_t *> usedBlocks;
};

template<class T, class... Args> inline void Construct(T* p, Args&&... inits){
	new (p) T(std::forward<Args>(inits)...);
}

template<class T> requires std::is_trivially_destructible_v<T>
inline void Destory(T* p) {}

template<class T> inline void Destory(T* p) {
	p->~T();
}

template <class T, std::size_t Align = std::max({alignof(T), alignof(std::int8_t *), sizeof(std::int8_t *)})> requires
LegitAlignment<Align, T> && (Align >= std::max(alignof(std::int8_t *), sizeof(std::int8_t *))) && (Align <= CacheLineAlign)
class MemoryPool {
public:
	MemoryPool(std::size_t count = 16384) : curBlockPos(0) {
		constexpr std::size_t complement = Align - 1;
		constexpr std::size_t objectByteCounts = ((sizeof(T) + complement) & (~complement));
		blockSize = count * objectByteCounts;
		curBlock = allocAligned<int8_t>(blockSize);
		deadStack = nullptr;
	}
	MemoryPool(const MemoryPool&) = delete;
	MemoryPool(MemoryPool&&) = delete;
	MemoryPool& operator=(const MemoryPool&) = delete;
	MemoryPool& operator=(MemoryPool&&) = delete;
	
	template<class... Args> T* Alloc(Args&&... inits) {
		constexpr std::size_t complement = Align - 1;
		constexpr std::size_t objectByteCounts = ((sizeof(T) + complement) & (~complement));

		T *ret = nullptr;
		if (deadStack != nullptr) {
			ret = (T *)deadStack;
			deadStack = *(int8_t **)deadStack;
		}
		else {
			if (curBlockPos + objectByteCounts > blockSize){
				usedBlocks.push_back(curBlock);
				curBlock = allocAligned<int8_t>(blockSize);
				curBlockPos = 0;
			}
			ret = (T *)(curBlock + curBlockPos);
			curBlockPos += objectByteCounts;
		}
		Construct(ret, std::forward<Args>(inits)...);
		return ret;
	}

	void Dealloc(T *item) {
		Destory(item);
		*(int8_t **)item = deadStack;
		deadStack = (int8_t *)item;
	}

	~MemoryPool() {
		freeAligned(curBlock);
		std::size_t us = usedBlocks.size();
		for (std::size_t i = 0; i < us; i++)
			freeAligned(usedBlocks[i]);
	}

	private:
		std::size_t curBlockPos, blockSize;
		std::int8_t *curBlock;
		std::int8_t *deadStack;
		std::vector<int8_t *> usedBlocks;
};

}

#endif
